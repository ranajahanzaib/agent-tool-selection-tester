import pytest

from agent_tool_selection_tester.core import TASKS, ExperimentConfig
from agent_tool_selection_tester.runner import FakeAdapter, run_case


@pytest.mark.asyncio
async def test_fake_adapter_runs_through_mcp() -> None:
    task = TASKS[0]
    adapter = FakeAdapter(
        {
            task.prompt: [
                {"tool_calls": [{"name": "get_customer", "arguments": {"customer_id": "cus_1001"}}]}
            ]
        }
    )
    result = await run_case(task, None, ExperimentConfig(max_tool_rounds=2), adapter, 0)
    assert result["evaluation"]["selection_correct"]
    assert result["evaluation"]["task_success"]
    assert any(event["event"] == "tool_call_result" for event in result["trace"])


@pytest.mark.asyncio
async def test_malformed_arguments_are_recorded_without_crashing() -> None:
    task = TASKS[0]
    adapter = FakeAdapter(
        {task.prompt: [{"tool_calls": [{"name": "get_customer", "arguments": "not-json"}]}]}
    )
    result = await run_case(task, None, ExperimentConfig(max_tool_rounds=1), adapter, 0)
    assert result["evaluation"]["selected_first"] == "invalid_call"
    assert result["evaluation"]["schema_valid"] is False
    assert result["evaluation"]["comprehension_scored"]
