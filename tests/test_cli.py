from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from agent_tool_selection_tester.cli import app
from agent_tool_selection_tester.storage import Store, get_default_db_path

runner = CliRunner()


def test_cli_doctor() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Agent Tool Selection Tester" in result.stdout
    assert "MCP Python SDK installed" in result.stdout
    assert "Results directory writable" in result.stdout


def test_cli_examples() -> None:
    result = runner.invoke(app, ["examples"])
    assert result.exit_code == 0
    assert "customer-tools" in result.stdout


def test_cli_demo_and_runs_and_report(tmp_path: Path) -> None:
    db_file = tmp_path / "test_results.sqlite3"
    with patch.dict("os.environ", {"AGENT_TOOL_SELECTION_TESTER_DB": str(db_file)}):
        # 1. Seed demo without opening UI
        demo_result = runner.invoke(app, ["demo", "--no-ui"])
        assert demo_result.exit_code == 0
        assert "Created synthetic demonstration run" in demo_result.stdout
        assert "synthetic-" in demo_result.stdout

        # 2. List runs
        runs_result = runner.invoke(app, ["runs"])
        assert runs_result.exit_code == 0
        assert "synthetic-" in runs_result.stdout

        # Extract run_id
        store = Store(db_file)
        runs = store.runs()
        store.close()
        assert len(runs) >= 1
        run_id = runs[0]["id"]

        # 3. Report on the run
        report_result = runner.invoke(app, ["report", run_id])
        assert report_result.exit_code == 0
        assert "Synthetic demo" in report_result.stdout
        assert "baseline" in report_result.stdout

        # 4. Export as JSON
        json_out = tmp_path / "export.json"
        export_json = runner.invoke(app, ["export", run_id, "--format", "json", "-o", str(json_out)])
        assert export_json.exit_code == 0
        assert json_out.exists()
        assert "Exported" in export_json.stdout

        # 5. Export as CSV
        csv_out = tmp_path / "export.csv"
        export_csv = runner.invoke(app, ["export", run_id, "--format", "csv", "-o", str(csv_out)])
        assert export_csv.exit_code == 0
        assert csv_out.exists()
        assert "Exported" in export_csv.stdout


def test_cli_run_offline(tmp_path: Path) -> None:
    db_file = tmp_path / "offline_results.sqlite3"
    with patch.dict("os.environ", {"AGENT_TOOL_SELECTION_TESTER_DB": str(db_file)}):
        result = runner.invoke(app, ["run", "--offline", "--repetitions", "1"])
        assert result.exit_code == 0
        assert "customer-tool-comprehension" in result.stdout
        assert "Saved run ID:" in result.stdout

        store = Store(db_file)
        runs = store.runs()
        store.close()
        assert len(runs) == 1


def test_storage_fallback_when_home_restricted(tmp_path: Path) -> None:
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("pathlib.Path.home", return_value=tmp_path / "non_existent_readonly_home"),
    ):
        # The home dir candidate cannot be created if parent is uncreatable or restricted
        fallback_path = get_default_db_path()
        assert fallback_path.parent.exists()
