import pytest
from mcp.client import Client

from agent_tool_selection_tester.core import Fixture, make_variant
from agent_tool_selection_tester.mcp_runtime import build_server


@pytest.mark.asyncio
async def test_real_in_process_mcp_discovery_and_aliased_call() -> None:
    variant = make_variant("parameter_alias")
    fixture = Fixture.fresh()
    async with Client(build_server(variant.tools, fixture)) as client:
        discovered = await client.list_tools()
        lookup = next(tool for tool in discovered.tools if tool.name == "get_customer")
        assert "id" in lookup.input_schema["properties"]
        result = await client.call_tool("get_customer", {"id": "cus_1001"})
        assert not result.is_error
        assert "alex.morgan@example.com" in result.content[0].text


@pytest.mark.asyncio
async def test_mcp_invalid_schema_arguments_return_tool_error() -> None:
    async with Client(
        build_server(make_variant("description_paraphrase").tools, Fixture.fresh())
    ) as client:
        result = await client.call_tool("get_customer", {})
        assert result.is_error
        update = await client.call_tool(
            "update_customer", {"customer_id": "cus_1001", "email": "not-an-email"}
        )
        assert update.is_error


@pytest.mark.asyncio
async def test_order_alias_dispatch_and_rejected_cancel() -> None:
    tools = make_variant("parameter_alias").tools
    fixture = Fixture.fresh()
    async with Client(build_server(tools, fixture)) as client:
        result = await client.call_tool("get_order", {"id": "ord_2004"})
        assert not result.is_error
        assert '"status": "delivered"' in result.content[0].text
        rejected = await client.call_tool("cancel_order", {"id": "ord_2004"})
        assert rejected.is_error
        assert fixture.attempts[-1]["tool"] == "order.cancel"
        assert not fixture.changes
        assert fixture.orders["ord_2004"]["status"] == "delivered"
