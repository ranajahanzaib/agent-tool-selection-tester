"""Create a clearly labeled synthetic dashboard fixture without calling any model."""

from __future__ import annotations

import uuid
from typing import Any

from . import __version__
from .core import TASKS, ExperimentConfig, catalog, digest, evaluate, make_variant
from .storage import Store


def seed_demo(store: Store) -> str:
    """Persist an illustrative, synthetic run so the dashboard works offline."""
    run_id = "synthetic-" + str(uuid.uuid4())
    mutations = [
        "description_shortened",
        "description_elaborated",
        "similar_tool_name",
        "parameter_alias",
        "tool_order_shuffle",
    ]
    variants = [None] + [make_variant(name, 42 + i) for i, name in enumerate(mutations)]
    manifest: dict[str, Any] = {
        "experiment_id": run_id,
        "synthetic": True,
        "disclosure": "Synthetic demonstration data. These records are hand-scripted interface examples, not model measurements.",
        "model": "SYNTHETIC DEMONSTRATION DATA",
        "temperature": None,
        "seed": 42,
        "repetitions": 1,
        "tool_catalog": [t.model_dump(mode="json") for t in catalog()],
        "tool_catalog_hash": digest([t.model_dump(mode="json") for t in catalog()]),
        "task_dataset_hash": digest([t.model_dump(mode="json") for t in TASKS]),
        "tasks": [t.model_dump(mode="json") for t in TASKS],
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
        "variant_generator_version": __version__,
    }
    config = ExperimentConfig(
        name="Synthetic demo · not model results", repetitions=1, mutations=[]
    )
    store.create_run(run_id, config.name, manifest, config.model_dump(mode="json"))
    for vi, variant in enumerate(variants):
        tools = variant.tools if variant else catalog()
        by_id = {tool.id: tool for tool in tools}
        for ti, task in enumerate(TASKS):
            plan = task.plans[0]
            steps = list(plan)
            calls: list[dict[str, Any]] = []
            if steps:
                degrade = variant is not None and (
                    variant.mutation_type == "description_shortened"
                    and ti in {2, 7, 12}
                    or variant.mutation_type == "similar_tool_name"
                    and ti in {1, 4, 8, 13}
                    or variant.mutation_type == "tool_order_shuffle"
                    and ti in {0, 5}
                )
                if degrade:
                    first_id, _ = steps[0]
                    confused = {
                        "customer.lookup_by_id": "customer.search",
                        "customer.search": "customer.lookup_by_id",
                        "customer.list_orders": "customer.lookup_by_id",
                        "customer.update": "customer.lookup_by_id",
                        "order.lookup": "order.cancel",
                        "order.cancel": "order.lookup",
                    }.get(first_id, "customer.lookup_by_id")
                    steps[0] = (
                        confused,
                        {"query": "Alex"}
                        if confused == "customer.search"
                        else {"customer_id": "cus_1001"}
                        if confused.startswith("customer.")
                        else {"order_id": "ord_2001"},
                    )
                for step_index, (tool_id, expected_args) in enumerate(steps):
                    tool = by_id[tool_id]
                    args = dict(expected_args)
                    if (
                        variant
                        and variant.mutation_type == "parameter_alias"
                        and ti in {1, 4, 6, 8, 14}
                    ):
                        args = {
                            "id" if key in {"customer_id", "order_id"} else key: value
                            for key, value in args.items()
                        }
                    # Some synthetic alias failures intentionally use canonical keys against the public alias schema.
                    if (
                        variant
                        and variant.mutation_type == "parameter_alias"
                        and ti in {1, 6}
                        and "customer_id" in args
                    ):
                        args = {"customer_id": args["customer_id"]}
                    calls.append(
                        {
                            "name": tool.name,
                            "arguments": args,
                            "is_error": task.category == "expected_rejection",
                        }
                    )
            final = (
                "Which order ID should I cancel?"
                if task.category == "clarification"
                else "That order has already been delivered and cannot be cancelled."
                if task.category == "expected_rejection"
                else "Synthetic illustration only."
            )
            evaluation = evaluate(task, calls, tools, final=final)
            trace = [
                {"event": "synthetic_fixture", "disclosure": manifest["disclosure"]},
                {"event": "task_submitted", "prompt": task.prompt},
            ]
            trace += [
                {"event": "tool_call_requested", "name": c["name"], "arguments": c["arguments"]}
                for c in calls
            ]
            trace.append({"event": "evaluation", "result": evaluation})
            result = {
                "task_id": task.id,
                "variant_id": variant.id if variant else "synthetic-baseline",
                "mutation_type": variant.mutation_type if variant else "baseline",
                "repetition": 0,
                "condition_class": variant.condition_class if variant else "baseline",
                "tool_order": [x.name for x in tools],
                "calls": calls,
                "final": final,
                "fixture_changes": [],
                "evaluation": evaluation,
                "trace": trace,
                "duration_ms": 0.0,
            }
            store.add_case(run_id, result)
    return run_id
