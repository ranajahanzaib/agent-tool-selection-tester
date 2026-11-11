"""In-process MCP v2 server that exposes a selected public catalog."""

from __future__ import annotations

import json

from jsonschema import Draft202012Validator, FormatChecker
from mcp.server import Server, ServerRequestContext
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)

from .core import Fixture, ToolDef


def build_server(tools: list[ToolDef], fixture: Fixture) -> Server:
    """Create a real MCP server with dynamic names/schemas and isolated fixture dispatch."""
    public = {tool.name: tool for tool in tools}

    async def list_tools(
        _ctx: ServerRequestContext, _params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        return ListToolsResult(
            tools=[
                Tool(name=t.name, description=t.description, input_schema=t.input_schema)
                for t in tools
            ]
        )

    async def call_tool(
        _ctx: ServerRequestContext, params: CallToolRequestParams
    ) -> CallToolResult:
        tool = public.get(params.name)
        if tool is None:
            return CallToolResult(
                is_error=True,
                content=[TextContent(type="text", text=f"Unknown tool: {params.name}")],
            )
        args = params.arguments or {}
        errors = list(
            Draft202012Validator(tool.input_schema, format_checker=FormatChecker()).iter_errors(
                args
            )
        )
        if errors:
            message = "; ".join(error.message for error in errors)
            return CallToolResult(
                is_error=True,
                content=[TextContent(type="text", text=f"Invalid arguments: {message}")],
            )
        # Public aliases are a presentation-only change; normalize at the dispatch boundary.
        normalized = dict(args)
        if "id" in normalized and tool.id.startswith("customer."):
            normalized["customer_id"] = normalized.pop("id")
        elif "id" in normalized and tool.id.startswith("order."):
            normalized["order_id"] = normalized.pop("id")
        try:
            payload = fixture.call(tool.id, normalized)
        except (KeyError, TypeError, ValueError) as exc:
            return CallToolResult(
                is_error=True, content=[TextContent(type="text", text=f"Tool error: {exc}")]
            )
        if isinstance(payload, dict) and "error" in payload:
            return CallToolResult(
                is_error=True, content=[TextContent(type="text", text=str(payload["error"]))]
            )
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]
        )

    return Server(
        "agent_tool_selection_tester-customer-fixture",
        version="0.1.0",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )
