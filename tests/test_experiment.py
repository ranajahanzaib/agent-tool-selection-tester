from pathlib import Path

import pytest

from agent_tool_selection_tester.core import TASKS, ExperimentConfig
from agent_tool_selection_tester.experiment import aggregate, execute, load_config, load_tasks
from agent_tool_selection_tester.runner import FakeAdapter
from agent_tool_selection_tester.storage import Store


@pytest.mark.asyncio
async def test_complete_fake_experiment_persists_manifest_and_trace(tmp_path: Path) -> None:
    task = TASKS[0]
    adapter = FakeAdapter(
        {
            task.prompt: [
                {"tool_calls": [{"name": "get_customer", "arguments": {"customer_id": "cus_1001"}}]}
            ]
        }
    )
    store = Store(tmp_path / "results.sqlite3")
    try:
        run_id, results = await execute(
            ExperimentConfig(name="offline-test", mutations=[], repetitions=1),
            adapter,
            store,
            tasks=[task],
        )
        metadata, stored = store.get_run(run_id)
        assert len(results) == len(stored) == 1
        assert metadata["name"] == "offline-test"
        assert "tool_catalog_hash" in metadata["manifest"]
        assert results[0]["evaluation"]["task_success"]
        assert results[0]["trace"][-1]["event"] == "evaluation"
        totals = aggregate(stored)
        assert totals[0]["sample_count"] == 1
        assert totals[0]["schema_sample_count"] == 1
    finally:
        store.close()


def test_bundled_yaml_config_and_tasks_parse() -> None:
    root = Path(__file__).parents[1] / "src/agent_tool_selection_tester/examples"
    config = load_config(root / "customer-tools.yaml")
    tasks = load_tasks(root / config.tasks_file)
    assert len(tasks) == 17
    assert config.repetitions == 3
    assert len(config.mutations) == 9
    assert config.seed == config.variant_seed == 42


def test_model_and_variant_seeds_are_independent(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "model:\n  seed: 9\nexperiment:\n  variant_seed: 71\nmutations: []\n"
    )
    config = load_config(config_path)
    assert config.seed == 9
    assert config.variant_seed == 71


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("modle: typo\n")
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_config(config_path)
