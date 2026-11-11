"""Agent Tool Selection Tester command line interface."""

from __future__ import annotations

import asyncio
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from . import __version__
from .experiment import aggregate, execute, load_config, load_tasks
from .runner import OllamaAdapter
from .storage import DEFAULT_DB, Store

app = typer.Typer(
    no_args_is_help=True, help="Run local, controlled LLM tool-comprehension experiments."
)
console = Console()


def _example_config() -> Path:
    return Path(__file__).parent / "examples" / "customer-tools.yaml"


@app.command()
def doctor() -> None:
    """Check local runtime prerequisites without downloading a model."""
    console.print(f"Agent Tool Selection Tester {__version__} · Python {sys.version.split()[0]}")
    try:
        import mcp  # noqa: F401

        console.print("[green]✓[/green] MCP Python SDK installed")
    except ImportError:
        console.print("[red]✗[/red] MCP SDK unavailable; run `uv sync`")
    if shutil.which("ollama"):
        console.print("[green]✓[/green] Ollama CLI found")
        console.print(
            "Tool-call capability depends on the selected local model and is exercised during a run."
        )
        try:
            import ollama

            models = ollama.list().models
            model_names = [m.model for m in models if m.model]
            console.print(
                "Local models: " + (", ".join(model_names) if model_names else "none installed")
            )
            if not model_names:
                console.print("Install the default model with `ollama pull qwen3:4b-instruct`.")
            elif "qwen3:4b-instruct" not in model_names:
                console.print(
                    "The default model is missing; install it with `ollama pull qwen3:4b-instruct` or pass `--model`."
                )
        except Exception as exc:  # noqa: BLE001 - Ollama failures are doctor diagnostics.
            console.print(f"[yellow]![/yellow] Ollama did not respond: {exc}")
    else:
        console.print("[yellow]![/yellow] Ollama is not installed or is not on PATH")
    try:
        DEFAULT_DB.parent.mkdir(parents=True, exist_ok=True)
        with (DEFAULT_DB.parent / ".write-check").open("w") as f:
            f.write("ok")
        (DEFAULT_DB.parent / ".write-check").unlink()
        console.print(f"[green]✓[/green] Results directory writable: {DEFAULT_DB.parent}")
    except OSError as exc:
        console.print(f"[red]✗[/red] Results directory unavailable: {exc}")


@app.command()
def examples() -> None:
    """List bundled experiment suites."""
    console.print(
        "customer-tools — 6 customer/order tools, 17 tasks, nine controlled interface variants"
    )
    console.print(f"Config: {_example_config()}")


@app.command()
def run(
    config: Path | None = typer.Argument(  # noqa: B008 - Typer binds argument metadata here.
        None, help="Experiment YAML (defaults to customer-tools)."
    ),
    model: str | None = typer.Option(None, "--model"),
    repetitions: int | None = typer.Option(None, "--repetitions", min=1, max=100),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Run the configured tasks against Ollama and store every trace."""
    path = config or _example_config()
    try:
        settings = load_config(path, model, repetitions)
        adapter = OllamaAdapter(settings.model)
        tasks_path = (
            path.parent / settings.tasks_file
            if settings.tasks_file
            else _example_config().with_name("customer-tasks.yaml")
        )
        tasks = load_tasks(tasks_path)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    store = Store()
    condition_count = len(settings.mutations) + int(settings.include_baseline)
    total = condition_count * len(tasks) * settings.repetitions
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            "{task.completed}/{task.total}",
            TimeElapsedColumn(),
        ) as progress:
            task_progress = progress.add_task("Running paired task cases", total=total)

            def on_case(_result: dict[str, object]) -> None:
                progress.advance(task_progress)

            run_id, results = asyncio.run(
                execute(settings, adapter, store, tasks=tasks, on_case=on_case)
            )
    except Exception as exc:
        console.print(f"[red]Experiment could not be completed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        store.close()
    table = Table(title=f"Run {run_id[:8]} · {settings.name}")
    for col in [
        "Condition",
        "Class",
        "Selection",
        "Δ selection pp",
        "Schema valid",
        "Task success",
        "n",
        "schema n",
    ]:
        table.add_column(col)
    for item in aggregate(results):
        table.add_row(
            item["mutation_type"],
            item["condition_class"],
            _pct(item["selection_accuracy"]),
            "—" if item["selection_delta_pp"] is None else f"{item['selection_delta_pp']:+.1f}",
            _pct(item["schema_validity"]),
            _pct(item["task_success"]),
            str(item["sample_count"]),
            str(item["schema_sample_count"]),
        )
    console.print(table)
    console.print(f"Saved run ID: {run_id}")
    infrastructure_errors = sum(item["infrastructure_errors"] for item in aggregate(results))
    if infrastructure_errors:
        console.print(
            f"[yellow]{infrastructure_errors} cases had provider/runtime errors and were excluded from comprehension metrics.[/yellow]"
        )
        if infrastructure_errors == len(results):
            console.print(
                f"Check `agent-tool-selection-tester doctor`; if `{settings.model}` is missing, install it with `ollama pull {settings.model}`. Agent Tool Selection Tester does not download models."
            )
    if verbose:
        console.print_json(json.dumps(results[-1]["trace"], indent=2))


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


@app.command()
def runs() -> None:
    """List locally stored runs."""
    store = Store()
    try:
        data = store.runs()
    finally:
        store.close()
    table = Table(title="Stored runs")
    for col in ["Run ID", "Name", "Started (UTC)"]:
        table.add_column(col)
    for x in data:
        table.add_row(x["id"], x["name"], x["started_at"])
    console.print(table)


@app.command()
def report(run_id: str) -> None:
    """Summarize a stored experiment without rerunning the model."""
    store = Store()
    try:
        run_info, results = store.get_run(run_id)
    finally:
        store.close()
    console.print(f"[bold]{run_info['name']}[/bold] · {run_info['id']} · {len(results)} cases")
    manifest = json.loads(run_info["manifest"])
    console.print_json(
        json.dumps(
            {
                key: manifest.get(key)
                for key in [
                    "model",
                    "model_details",
                    "temperature",
                    "seed",
                    "repetitions",
                    "tool_catalog_hash",
                    "task_dataset_hash",
                    "variant_generator_version",
                    "inference_note",
                    "synthetic",
                    "disclosure",
                ]
                if key in manifest
            }
        )
    )
    table = Table()
    for c in [
        "Condition",
        "Class",
        "Selection",
        "Δ pp",
        "Schema valid",
        "Task completion",
        "n",
        "Schema n",
        "Infra errors",
    ]:
        table.add_column(c)
    for item in aggregate(results):
        table.add_row(
            item["mutation_type"],
            item["condition_class"],
            _pct(item["selection_accuracy"]),
            "—" if item["selection_delta_pp"] is None else f"{item['selection_delta_pp']:+.1f}",
            _pct(item["schema_validity"]),
            _pct(item["task_success"]),
            str(item["sample_count"]),
            str(item["schema_sample_count"]),
            str(item["infrastructure_errors"]),
        )
    console.print(table)
    confusions: dict[str, int] = {}
    for r in results:
        e = r["evaluation"]
        expected = e.get("expected_first")
        selected = e.get("selected_first")
        correct_first = selected in expected if isinstance(expected, list) else selected == expected
        if e.get("comprehension_scored") and not correct_first:
            key = f"{expected} → {e.get('selected_first')}"
            confusions[key] = confusions.get(key, 0) + 1
    console.print(
        "Observed first-tool mismatches: "
        + (json.dumps(confusions, sort_keys=True) if confusions else "none")
    )
    error_table = Table(title="Argument, mutation, and recovery counts")
    for col in [
        "Condition",
        "Schema errors",
        "Argument errors",
        "Unauthorized writes",
        "Incorrect writes",
        "Recovered cases",
    ]:
        error_table.add_column(col)
    for item in aggregate(results):
        condition_cases = [
            r["evaluation"]
            for r in results
            if r["variant_id"] == item["variant_id"] and r["evaluation"].get("comprehension_scored")
        ]
        error_table.add_row(
            item["mutation_type"],
            str(sum(e.get("schema_valid") is False for e in condition_cases)),
            str(sum(e.get("semantic_arguments_correct") is False for e in condition_cases)),
            str(item["unauthorized_mutations"]),
            str(item["incorrect_mutations"]),
            str(item["recovered_cases"]),
        )
    console.print(error_table)
    failures = [
        r
        for r in results
        if r["evaluation"].get("comprehension_scored") and not r["evaluation"].get("task_success")
    ]
    if failures:
        console.print("Representative unsuccessful traces:")
        for case in failures[:3]:
            observed = [
                f"{event.get('name')} {event.get('arguments')}"
                for event in case["trace"]
                if event.get("event") == "tool_call_requested"
            ]
            console.print(
                f"• {case['task_id']} · {case['mutation_type']}: {', '.join(observed) or 'no tool call'}"
            )
    improvements = [
        item
        for item in aggregate(results)
        if item["selection_delta_pp"] is not None and item["selection_delta_pp"] > 0
    ]
    console.print(
        "Observed first-selection improvements: "
        + (
            ", ".join(
                f"{x['mutation_type']} (+{x['selection_delta_pp']:.1f} pp, n={x['sample_count']})"
                for x in improvements
            )
            if improvements
            else "none in this run"
        )
    )
    console.print(
        "Interpretation: descriptive measurements from this local model and task sample; no claim of statistical significance or universal behavior."
    )


@app.command()
def export(
    run_id: str,
    format: str = typer.Option("json", "--format", case_sensitive=False),
    output: Path | None = typer.Option(None, "--output", "-o"),  # noqa: B008
) -> None:
    """Export stored raw case results as JSON or CSV."""
    if format not in {"json", "csv"}:
        raise typer.BadParameter("format must be json or csv")
    store = Store()
    try:
        run_info, results = store.get_run(run_id)
    finally:
        store.close()
    dest = output or Path(f"agent_tool_selection_tester-{run_id[:8]}.{format}")
    if format == "json":
        dest.write_text(json.dumps({"run": run_info, "cases": results}, indent=2))
    else:
        with dest.open("w", newline="") as f:
            fields = [
                "variant_id",
                "mutation_type",
                "condition_class",
                "task_id",
                "repetition",
                "selected_first",
                "expected_first",
                "selection_correct",
                "schema_valid",
                "semantic_arguments_correct",
                "task_success",
                "call_count",
                "unnecessary_calls",
                "recovered",
                "unauthorized_mutation",
                "incorrect_mutation",
                "infrastructure_error",
                "duration_ms",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in results:
                e = r["evaluation"]
                writer.writerow(
                    {
                        "variant_id": r["variant_id"],
                        "mutation_type": r["mutation_type"],
                        "condition_class": r.get("condition_class"),
                        "task_id": r["task_id"],
                        "repetition": r["repetition"],
                        "duration_ms": r["duration_ms"],
                        **{k: e.get(k) for k in fields[5:-1]},
                    }
                )
    console.print(f"Exported {len(results)} cases to {dest}")


@app.command()
def ui() -> None:
    """Open the local dashboard for stored runs."""
    app_path = Path(__file__).parent / "dashboard.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=False)


@app.command()
def demo() -> None:
    """Open the dashboard and clearly label the bundled synthetic illustration."""
    from .demo import seed_demo

    store = Store()
    try:
        run_id = seed_demo(store)
    finally:
        store.close()
    console.print(
        f"[yellow]Created synthetic demonstration run {run_id}. It is not model measurement.[/yellow]"
    )
    ui()


if __name__ == "__main__":
    app()
