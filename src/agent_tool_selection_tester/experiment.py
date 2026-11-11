"""Experiment orchestration and reproducibility manifest creation."""

from __future__ import annotations

import asyncio
import importlib.metadata
import random
import uuid
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .core import TASKS, ExperimentConfig, Variant, catalog, digest, make_variant
from .runner import ModelAdapter, run_case
from .storage import Store


def load_config(
    path: Path, model: str | None = None, repetitions: int | None = None
) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    if "model" in raw and isinstance(raw["model"], dict):
        model_settings = raw.pop("model")
        raw["provider"] = model_settings.get("provider", raw.get("provider", "ollama"))
        raw["model"] = model_settings.get("name", "qwen3:4b-instruct")
        raw["temperature"] = model_settings.get("temperature", raw.get("temperature", 0))
        raw["seed"] = model_settings.get("seed", raw.get("seed", 42))
        raw["max_tool_rounds"] = model_settings.get(
            "max_tool_rounds", raw.get("max_tool_rounds", 4)
        )
        raw["timeout_seconds"] = model_settings.get(
            "timeout_seconds", raw.get("timeout_seconds", 120)
        )
    if "experiment" in raw:
        exp = raw.pop("experiment") or {}
        raw["include_baseline"] = exp.get("include_baseline", True)
        raw["repetitions"] = exp.get("repetitions", raw.get("repetitions", 3))
        raw["variant_seed"] = exp.get("variant_seed", raw.get("variant_seed", 42))
    if "mutations" in raw:
        raw["mutations"] = [
            {"type": m if isinstance(m, str) else m["type"]} for m in raw["mutations"]
        ]
    if model:
        raw["model"] = model
    if repetitions is not None:
        raw["repetitions"] = repetitions
    try:
        return ExperimentConfig.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"Invalid experiment config in {path}: {exc}") from exc


def load_tasks(path: Path) -> list[Any]:
    """Parse explicit YAML task plans into evaluator-ready canonical steps."""
    from .core import Task

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, list):
        raise TypeError(f"Task dataset {path} must contain a YAML list.")
    tasks = []
    for row in raw:
        plans = [
            [(step["tool"], step.get("arguments", {})) for step in plan]
            for plan in row.get("plans", [])
        ]
        tasks.append(
            Task(
                id=row["id"],
                prompt=row["prompt"],
                plans=plans,
                should_mutate=row.get("should_mutate", False),
                category=row.get("category", "selection"),
            )
        )
    return tasks


async def execute(
    config: ExperimentConfig,
    adapter: ModelAdapter,
    store: Store,
    tasks: list[Any] | None = None,
    on_case: Any = None,
) -> tuple[str, list[dict[str, Any]]]:
    task_list = tasks or TASKS
    run_id = str(uuid.uuid4())
    variants: list[Variant | None] = ([None] if config.include_baseline else []) + [
        make_variant(m.type, config.variant_seed + i) for i, m in enumerate(config.mutations)
    ]
    if not variants:
        raise ValueError("An experiment needs a baseline or at least one mutation condition.")
    model_info = None
    metadata_method = getattr(adapter, "model_metadata", None)
    if metadata_method is not None:
        try:
            model_info = await asyncio.wait_for(metadata_method(), timeout=config.timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - absent model metadata must not block the experiment.
            model_info = {"unavailable": f"{type(exc).__name__}: {exc}"}
    manifest = {
        "experiment_id": run_id,
        "model": config.model,
        "temperature": config.temperature,
        "seed": config.seed,
        "variant_seed": config.variant_seed,
        "repetitions": config.repetitions,
        "tool_catalog_hash": digest([t.model_dump(mode="json") for t in catalog()]),
        "task_dataset_hash": digest([t.model_dump(mode="json") for t in task_list]),
        "variant_generator_version": __version__,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ["mcp", "ollama", "pydantic", "jsonschema"]
            if _has_package(name)
        },
        "inference_note": "Temperature and seed are requested settings; exact outputs can vary by model build, runtime, and hardware.",
        "model_details": model_info,
        "system_prompt": "You are a careful customer-support assistant. Use available tools only when needed. Ask a concise clarification when required details are missing. Never claim a tool succeeded unless its result confirms it.",
        "tool_catalog": [t.model_dump(mode="json") for t in catalog()],
        "tasks": [t.model_dump(mode="json") for t in task_list],
        "variants": [
            {
                "id": v.id,
                "mutation_type": v.mutation_type,
                "seed": v.seed,
                "affected_tool_ids": v.affected,
                "changed_fields": v.changed_fields,
                "before": v.before,
                "after": v.after,
                "semantic_preserving": v.semantic_preserving,
                "condition_class": v.condition_class,
                "catalog_hash": v.catalog_hash,
                "summary": v.summary,
                "order": v.order,
            }
            for v in variants
            if v
        ],
    }
    store.create_run(run_id, config.name, manifest, config.model_dump(mode="json"))
    results = []
    for repetition in range(config.repetitions):
        rotation = repetition % len(variants)
        ordered_variants = variants[rotation:] + variants[:rotation]
        ordered_tasks = list(task_list)
        random.Random(config.seed + repetition).shuffle(ordered_tasks)
        for variant in ordered_variants:
            for task in ordered_tasks:
                result = await run_case(task, variant, config, adapter, repetition)
                store.add_case(run_id, result)
                results.append(result)
                if on_case is not None:
                    on_case(result)
    return run_id, results


def _has_package(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def aggregate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        grouped.setdefault(r["variant_id"], []).append(r)
    output = []
    for variant, rows in grouped.items():
        scored = [r["evaluation"] for r in rows if r["evaluation"].get("comprehension_scored")]
        n = len(scored)
        schema_scored = [row for row in scored if row.get("schema_valid") is not None]
        argument_scored = [row for row in scored if row.get("call_count", 0) > 0]
        output.append(
            {
                "variant_id": variant,
                "mutation_type": rows[0]["mutation_type"],
                "condition_class": rows[0].get("condition_class", "baseline"),
                "sample_count": n,
                "schema_sample_count": len(schema_scored),
                "argument_sample_count": len(argument_scored),
                "selection_accuracy": sum(x["selection_correct"] for x in scored) / n
                if n
                else None,
                "schema_validity": sum(x["schema_valid"] is True for x in schema_scored)
                / len(schema_scored)
                if schema_scored
                else None,
                "argument_accuracy": sum(x["semantic_arguments_correct"] for x in argument_scored)
                / len(argument_scored)
                if argument_scored
                else None,
                "task_success": sum(x["task_success"] for x in scored) / n if n else None,
                "unauthorized_mutations": sum(x["unauthorized_mutation"] for x in scored),
                "incorrect_mutations": sum(x.get("incorrect_mutation", False) for x in scored),
                "recovered_cases": sum(x.get("recovered", False) for x in scored),
                "infrastructure_errors": sum(
                    not r["evaluation"].get("comprehension_scored") for r in rows
                ),
            }
        )
    baseline = next((row for row in output if row["mutation_type"] == "baseline"), None)
    for row in output:
        row["selection_delta_pp"] = (
            (row["selection_accuracy"] - baseline["selection_accuracy"]) * 100
            if baseline
            and row["selection_accuracy"] is not None
            and baseline["selection_accuracy"] is not None
            else None
        )
        row["task_success_delta_pp"] = (
            (row["task_success"] - baseline["task_success"]) * 100
            if baseline and row["task_success"] is not None and baseline["task_success"] is not None
            else None
        )
    return output
