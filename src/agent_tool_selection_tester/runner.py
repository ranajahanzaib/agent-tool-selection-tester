"""Bounded MCP-to-model execution loop and scripted offline adapter."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Protocol

from mcp.client import Client

from .core import ExperimentConfig, Fixture, Task, Variant, canonical_json, catalog, evaluate
from .mcp_runtime import build_server

SYSTEM = "You are a careful customer-support assistant. Use available tools only when needed. Ask a concise clarification when required details are missing. Never claim a tool succeeded unless its result confirms it."


class ModelAdapter(Protocol):
    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], config: ExperimentConfig
    ) -> dict[str, Any]: ...


class OllamaAdapter:
    def __init__(self, model: str, host: str | None = None) -> None:
        try:
            from ollama import AsyncClient
        except ImportError as exc:
            raise RuntimeError("Ollama Python package is missing. Run `uv sync`.") from exc
        self.model = model
        self.client = AsyncClient(host=host) if host else AsyncClient()

    async def model_metadata(self) -> dict[str, Any]:
        """Return model metadata reported by Ollama, including the model digest."""
        response = await self.client.show(self.model)
        return response.model_dump(mode="json")

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], config: ExperimentConfig
    ) -> dict[str, Any]:
        response = await self.client.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            options={"temperature": config.temperature, "seed": config.seed},
            stream=False,
        )
        message = response.message
        calls = []
        for call in message.tool_calls or []:
            args = call.function.arguments
            calls.append(
                {"name": call.function.name, "arguments": args if isinstance(args, dict) else {}}
            )
        return {
            "content": message.content or "",
            "tool_calls": calls,
            "raw": response.model_dump(mode="json")
            if hasattr(response, "model_dump")
            else str(response),
        }


class FakeAdapter:
    """A deterministic scripted adapter useful for tests and offline pipeline checks."""

    def __init__(self, scripts: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.scripts = scripts or {}
        self.seen: list[dict[str, Any]] = []
        self._indices: dict[str, int] = {}

    def reset(self) -> None:
        """Reset script cursors so paired cases get the same scripted responses."""
        self._indices.clear()

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], config: ExperimentConfig
    ) -> dict[str, Any]:
        task_prompt = next((m["content"] for m in messages if m["role"] == "user"), "")
        names = [x["function"]["name"] for x in tools]
        choices = self.scripts.get(task_prompt, [])
        index = self._indices.get(task_prompt, 0)
        item = (
            choices[min(index, len(choices) - 1)]
            if choices
            else {"content": "I need more information to help with that."}
        )
        self._indices[task_prompt] = index + 1
        result: dict[str, Any] = {
            "content": item.get("content", ""),
            "tool_calls": item.get("tool_calls", []),
        }
        self.seen.append({"prompt": task_prompt, "tool_order": names, "response": result})
        return result


def _model_tools(discovered: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in discovered
    ]


async def run_case(
    task: Task,
    variant: Variant | None,
    config: ExperimentConfig,
    adapter: ModelAdapter,
    repetition: int,
) -> dict[str, Any]:
    tools = variant.tools if variant else catalog()
    reset_adapter = getattr(adapter, "reset", None)
    if reset_adapter is not None:
        reset_adapter()
    fixture = Fixture.fresh()
    server = build_server(tools, fixture)
    trace: list[dict[str, Any]] = [{"event": "task_submitted", "prompt": task.prompt}]
    calls: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task.prompt},
    ]
    started = time.perf_counter()
    error: str | None = None
    final = ""
    offered: list[Any] = []
    seen_calls: set[str] = set()
    try:
        async with Client(server) as client:
            offered = (await client.list_tools()).tools
            trace.append(
                {
                    "event": "tools_discovered",
                    "names": [t.name for t in offered],
                    "canonical_ids": [t.id for t in tools],
                }
            )
            llm_tools = _model_tools(offered)
            for turn in range(config.max_tool_rounds):
                trace.append(
                    {
                        "event": "model_request",
                        "turn": turn + 1,
                        "messages": messages,
                        "tools": llm_tools,
                        "settings": {
                            "model": config.model,
                            "temperature": config.temperature,
                            "seed": config.seed,
                            "timeout_seconds": config.timeout_seconds,
                        },
                    }
                )
                response = await asyncio.wait_for(
                    adapter.chat(messages, llm_tools, config), timeout=config.timeout_seconds
                )
                trace.append(
                    {
                        "event": "model_response",
                        "turn": turn + 1,
                        "raw": response.get("raw"),
                        "content": response.get("content", ""),
                    }
                )
                final = response.get("content", "")
                pending = response.get("tool_calls", [])
                if not pending:
                    break
                assistant_message = {
                    "role": "assistant",
                    "content": final or None,
                    "tool_calls": [
                        {
                            "function": {
                                "name": c.get("name"),
                                "arguments": c.get("arguments", {})
                                if isinstance(c.get("arguments", {}), dict)
                                else {},
                            },
                        }
                        for c in pending
                    ],
                }
                messages.append(assistant_message)
                for c in pending:
                    call = {"name": c.get("name", ""), "arguments": c.get("arguments", {})}
                    calls.append(call)
                    trace.append(
                        {
                            "event": "tool_call_requested",
                            "name": call["name"],
                            "arguments": call["arguments"],
                        }
                    )
                    signature = canonical_json(call)
                    if signature in seen_calls:
                        call["is_error"] = True
                        text = (
                            "Repeated identical tool call suppressed to prevent an execution loop."
                        )
                        is_error = True
                        trace.append(
                            {
                                "event": "repeated_call_suppressed",
                                "name": call["name"],
                                "arguments": call["arguments"],
                            }
                        )
                    else:
                        seen_calls.add(signature)
                        result = await client.call_tool(
                            call["name"],
                            call["arguments"] if isinstance(call["arguments"], dict) else {},
                        )
                        text = "\n".join(x.text for x in result.content if hasattr(x, "text"))
                        is_error = bool(result.is_error)
                        call["is_error"] = is_error
                    trace.append(
                        {
                            "event": "tool_call_result",
                            "name": call["name"],
                            "is_error": is_error,
                            "content": text,
                        }
                    )
                    messages.append({"role": "tool", "tool_name": call["name"], "content": text})
            else:
                trace.append({"event": "round_limit_reached", "limit": config.max_tool_rounds})
    except TimeoutError as exc:
        error = f"Model request timed out: {exc or f'{config.timeout_seconds:g}-second limit'}"
    except Exception as exc:  # noqa: BLE001 - provider and protocol errors are saved per case.
        error = f"{type(exc).__name__}: {exc}"
    evaluation = evaluate(task, calls, tools, error, final, fixture.changes)
    trace.append({"event": "evaluation", "result": evaluation})
    return {
        "task_id": task.id,
        "variant_id": variant.id if variant else "baseline",
        "mutation_type": variant.mutation_type if variant else "baseline",
        "condition_class": variant.condition_class if variant else "baseline",
        "repetition": repetition,
        "tool_order": [x.name for x in offered],
        "calls": calls,
        "final": final,
        "fixture_changes": fixture.changes,
        "fixture_attempts": fixture.attempts,
        "evaluation": evaluation,
        "trace": trace,
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
    }
