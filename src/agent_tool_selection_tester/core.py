"""Core domain, fixture tools, controlled variants, and evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from jsonschema import Draft202012Validator, FormatChecker
from pydantic import BaseModel, ConfigDict, Field


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


class ToolDef(BaseModel):
    model_config = ConfigDict()
    id: str
    name: str
    description: str
    input_schema: dict[str, Any]


class Mutation(BaseModel):
    type: Literal[
        "description_paraphrase",
        "description_shortened",
        "description_elaborated",
        "similar_tool_name",
        "parameter_alias",
        "parameter_description",
        "tool_order_shuffle",
        "schema_customer_id_optional",
        "schema_misleading_query",
    ]


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "customer-tool-comprehension"
    description: str = "Evaluate related customer-management tools."
    provider: Literal["ollama"] = "ollama"
    model: str = "qwen3:4b-instruct"
    temperature: float = Field(default=0, ge=0, le=2)
    seed: int = 42
    variant_seed: int = 42
    repetitions: int = Field(default=3, ge=1, le=100)
    max_tool_rounds: int = Field(default=4, ge=1, le=20)
    timeout_seconds: float = Field(default=120, gt=0, le=3600)
    tasks_file: str | None = None
    include_baseline: bool = True
    mutations: list[Mutation] = Field(
        default_factory=lambda: [
            Mutation(type=cast(Any, x))
            for x in [
                "description_paraphrase",
                "description_shortened",
                "description_elaborated",
                "similar_tool_name",
                "parameter_alias",
                "parameter_description",
                "tool_order_shuffle",
                "schema_customer_id_optional",
                "schema_misleading_query",
            ]
        ]
    )


@dataclass
class Fixture:
    customers: dict[str, dict[str, Any]]
    orders: dict[str, dict[str, Any]]
    changes: list[dict[str, Any]] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def fresh(cls) -> Fixture:
        data_path = Path(__file__).parent / "examples" / "fixture-data.json"
        data = json.loads(data_path.read_text())
        customers = {c["customer_id"]: c for c in data["customers"]}
        orders = {o["order_id"]: o for o in data["orders"]}
        return cls(customers, orders)

    def call(self, tool_id: str, args: dict[str, Any]) -> Any:
        cid, oid = str(args.get("customer_id", "")), str(args.get("order_id", ""))
        if tool_id == "customer.lookup_by_id":
            return self.customers.get(cid, {"error": f"Customer {cid} was not found."})
        if tool_id == "customer.search":
            q = str(args.get("query", "")).lower()
            return [
                c
                for c in self.customers.values()
                if q in c["name"].lower() or q in c["email"].lower()
            ]
        if tool_id == "customer.list_orders":
            return [o for o in self.orders.values() if o["customer_id"] == cid]
        if tool_id == "customer.update":
            self.attempts.append({"tool": tool_id, "arguments": dict(args)})
            customer = self.customers.get(cid)
            if not customer:
                return {"error": f"Customer {cid} was not found."}
            updates: dict[str, Any] = {k: args[k] for k in ("email", "status") if k in args}
            before = {k: customer[k] for k in updates}
            customer.update(updates)
            self.attempts[-1]["applied"] = True
            self.changes.append(
                {"tool": tool_id, "customer_id": cid, "before": before, "after": updates}
            )
            return customer
        if tool_id == "order.lookup":
            return self.orders.get(oid, {"error": f"Order {oid} was not found."})
        if tool_id == "order.cancel":
            self.attempts.append({"tool": tool_id, "arguments": dict(args)})
            order = self.orders.get(oid)
            if not order:
                return {"error": f"Order {oid} was not found."}
            if order["status"] in {"shipped", "delivered", "cancelled"}:
                return {"error": f"Order in state {order['status']} cannot be cancelled."}
            before = order["status"]
            order["status"] = "cancelled"
            self.attempts[-1]["applied"] = True
            self.changes.append(
                {"tool": tool_id, "order_id": oid, "before": before, "after": "cancelled"}
            )
            return order
        raise KeyError(f"Unknown canonical tool: {tool_id}")


_id_schema = {"type": "string", "minLength": 1}
_tools = [
    ToolDef(
        id="customer.lookup_by_id",
        name="get_customer",
        description="Retrieve a customer using their unique customer ID.",
        input_schema={
            "type": "object",
            "properties": {
                "customer_id": {**_id_schema, "description": "Unique customer identifier"}
            },
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    ),
    ToolDef(
        id="customer.search",
        name="search_customer",
        description="Search customer records using a name, email address, or other supported search term.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Name, email, or search term",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    ToolDef(
        id="customer.list_orders",
        name="get_customer_orders",
        description="Retrieve all orders belonging to a specific customer ID.",
        input_schema={
            "type": "object",
            "properties": {
                "customer_id": {**_id_schema, "description": "Unique customer identifier"}
            },
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    ),
    ToolDef(
        id="customer.update",
        name="update_customer",
        description="Update explicitly permitted customer fields for a customer ID.",
        input_schema={
            "type": "object",
            "properties": {
                "customer_id": {**_id_schema, "description": "Unique customer identifier"},
                "email": {"type": "string", "format": "email", "description": "New email address"},
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive"],
                    "description": "New customer status",
                },
            },
            "required": ["customer_id"],
            "additionalProperties": False,
            "minProperties": 2,
        },
    ),
    ToolDef(
        id="order.lookup",
        name="get_order",
        description="Retrieve details for a single order using its unique order ID.",
        input_schema={
            "type": "object",
            "properties": {"order_id": {**_id_schema, "description": "Unique order identifier"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    ),
    ToolDef(
        id="order.cancel",
        name="cancel_order",
        description="Cancel an order if its current state permits cancellation.",
        input_schema={
            "type": "object",
            "properties": {"order_id": {**_id_schema, "description": "Unique order identifier"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    ),
]


def catalog() -> list[ToolDef]:
    return copy.deepcopy(_tools)


_PARAPHRASES = {
    "customer.lookup_by_id": "Fetch the customer record associated with a specified identifier.",
    "customer.search": "Find customer records matching a name, email, or other search term.",
    "customer.list_orders": "List the orders associated with a given customer identifier.",
    "customer.update": "Change permitted fields on the customer record identified by ID.",
    "order.lookup": "Fetch one order record using its order identifier.",
    "order.cancel": "Request cancellation of an order when its state allows it.",
}
_ELABORATIONS = {
    "customer.lookup_by_id": "Use only when the unique customer ID is already known. Do not use this for a name or email search.",
    "customer.search": "Search by customer name, email address, or another supported term. Use direct lookup when a unique customer ID is known.",
    "customer.list_orders": "Use when the customer's ID is known and you need that customer's orders. Use order lookup for one known order ID.",
    "customer.update": "Use only when the user explicitly requests a permitted field change and provides the customer ID and new value.",
    "order.lookup": "Use to retrieve one order when its unique order ID is known. This does not retrieve a customer's full order list.",
    "order.cancel": "Use only when cancellation is explicitly requested and an order ID is known. Some order states cannot be cancelled.",
}
_NAMES = {
    "customer.lookup_by_id": "get_customer_details",
    "customer.search": "find_customer",
    "customer.list_orders": "get_customers",
    "customer.update": "update_customer_record",
    "order.lookup": "fetch_order",
    "order.cancel": "cancel_order_record",
}


@dataclass
class Variant:
    id: str
    mutation_type: str
    seed: int
    tools: list[ToolDef]
    order: list[str]
    before: list[dict[str, Any]]
    after: list[dict[str, Any]]
    affected: list[str]
    changed_fields: list[str]
    semantic_preserving: bool
    condition_class: str
    catalog_hash: str
    summary: str


def make_variant(mutation: str, seed: int = 42) -> Variant:
    supported = {
        "description_paraphrase",
        "description_shortened",
        "description_elaborated",
        "similar_tool_name",
        "parameter_alias",
        "parameter_description",
        "tool_order_shuffle",
        "schema_customer_id_optional",
        "schema_misleading_query",
    }
    if mutation not in supported:
        raise ValueError(
            f"Unsupported mutation type {mutation!r}; choose from {', '.join(sorted(supported))}."
        )
    before = catalog()
    tools = copy.deepcopy(before)
    affected: list[str] = []
    summary = mutation.replace("_", " ")
    for i, t in enumerate(tools):
        if mutation == "description_paraphrase":
            t.description = _PARAPHRASES[t.id]
        elif mutation == "description_shortened":
            t.description = {
                "customer.lookup_by_id": "Get customer.",
                "customer.search": "Find customer.",
                "customer.list_orders": "Get orders.",
                "customer.update": "Update customer.",
                "order.lookup": "Get order.",
                "order.cancel": "Cancel order.",
            }[t.id]
        elif mutation == "description_elaborated":
            t.description += " " + _ELABORATIONS[t.id]
        elif mutation == "similar_tool_name":
            t.name = _NAMES[t.id]
        elif mutation == "parameter_alias" and "customer_id" in t.input_schema.get(
            "properties", {}
        ):
            props = t.input_schema["properties"]
            props["id"] = props.pop("customer_id")
            t.input_schema["required"] = [
                "id" if x == "customer_id" else x for x in t.input_schema["required"]
            ]
            t.input_schema["properties"]["id"]["description"] = "Customer identifier (id)"
        elif mutation == "parameter_alias" and "order_id" in t.input_schema.get("properties", {}):
            props = t.input_schema["properties"]
            props["id"] = props.pop("order_id")
            t.input_schema["required"] = [
                "id" if x == "order_id" else x for x in t.input_schema["required"]
            ]
            t.input_schema["properties"]["id"]["description"] = "Order identifier (id)"
        elif mutation == "parameter_description" and t.input_schema.get("properties"):
            first = next(iter(t.input_schema["properties"].values()))
            first["description"] = "ID" if seed % 2 else "Identifier of the record requested"
        elif mutation == "schema_customer_id_optional" and t.id == "customer.lookup_by_id":
            t.input_schema["required"] = []
        elif mutation == "schema_misleading_query" and t.id == "customer.lookup_by_id":
            t.input_schema["properties"]["query"] = {
                "type": "string",
                "description": "Optional customer name or email query",
            }
        if t.model_dump() != before[i].model_dump():
            affected.append(t.id)
    order = [x.name for x in tools]
    if mutation == "tool_order_shuffle":
        random.Random(seed).shuffle(tools)
        order = [x.name for x in tools]
        affected = [x.id for x in tools]
    after = [x.model_dump(mode="json") for x in tools]
    changed_fields = sorted(
        {
            key
            for original, changed in zip(
                [x.model_dump(mode="json") for x in before], after, strict=True
            )
            for key in original
            if original[key] != changed[key]
        }
    )
    if mutation == "tool_order_shuffle":
        changed_fields = ["tool_order"]
    summary = f"{mutation.replace('_', ' ')} changed {', '.join(changed_fields)} on {len(affected)} tools."
    h = digest(after)
    return Variant(
        id=f"{mutation}-{seed}-{h[7:15]}",
        mutation_type=mutation,
        seed=seed,
        tools=tools,
        order=order,
        before=[x.model_dump(mode="json") for x in before],
        after=after,
        affected=affected,
        changed_fields=changed_fields,
        semantic_preserving=mutation
        not in {"schema_customer_id_optional", "schema_misleading_query"},
        condition_class="contract-changing"
        if mutation.startswith("schema_")
        else "semantics-preserving",
        catalog_hash=h,
        summary=summary,
    )


class Task(BaseModel):
    id: str
    prompt: str
    plans: list[list[tuple[str, dict[str, Any]]]] = Field(default_factory=list)
    should_mutate: bool = False
    category: str = "selection"


TASKS = [
    Task(
        id="task_001",
        prompt="Retrieve the customer record for ID cus_1001.",
        plans=[[("customer.lookup_by_id", {"customer_id": "cus_1001"})]],
    ),
    Task(
        id="task_002",
        prompt="Find all orders belonging to customer cus_1001.",
        plans=[[("customer.list_orders", {"customer_id": "cus_1001"})]],
    ),
    Task(
        id="task_003",
        prompt="Search for the customer with email alex.morgan@example.com.",
        plans=[[("customer.search", {"query": "alex.morgan@example.com"})]],
    ),
    Task(
        id="task_004",
        prompt="Look up order ord_2004 and tell me its status.",
        plans=[[("order.lookup", {"order_id": "ord_2004"})]],
    ),
    Task(
        id="task_005",
        prompt="Change cus_1002's email to alex.new@example.com.",
        plans=[[("customer.update", {"customer_id": "cus_1002", "email": "alex.new@example.com"})]],
        should_mutate=True,
    ),
    Task(
        id="task_006",
        prompt="Cancel order ord_2006.",
        plans=[[("order.cancel", {"order_id": "ord_2006"})]],
        should_mutate=True,
    ),
    Task(
        id="task_007",
        prompt="What is the current email on customer cus_1003?",
        plans=[[("customer.lookup_by_id", {"customer_id": "cus_1003"})]],
    ),
    Task(
        id="task_008",
        prompt="Find customer Jamie Chen.",
        plans=[[("customer.search", {"query": "Jamie Chen"})]],
    ),
    Task(
        id="task_009",
        prompt="Get the customer profile for cus_1010, then list that customer's orders.",
        plans=[
            [
                ("customer.lookup_by_id", {"customer_id": "cus_1010"}),
                ("customer.list_orders", {"customer_id": "cus_1010"}),
            ]
        ],
    ),
    Task(
        id="task_010",
        prompt="Cancel order ord_2002.",
        plans=[[("order.cancel", {"order_id": "ord_2002"})]],
        should_mutate=True,
    ),
    Task(
        id="task_011",
        prompt="Tell me what customer ID cus_1001 means. Do not look anything up; the ID is just an example.",
        plans=[[]],
        category="no_tool",
    ),
    Task(id="task_012", prompt="Cancel the order for Alex.", plans=[[]], category="clarification"),
    Task(
        id="task_013",
        prompt="Find all customers named Alex Morgan.",
        plans=[[("customer.search", {"query": "Alex Morgan"})]],
    ),
    Task(
        id="task_014",
        prompt="Show me the customer record associated with email alex.morgan@example.com.",
        plans=[[("customer.search", {"query": "alex.morgan@example.com"})]],
    ),
    Task(
        id="task_015",
        prompt="Retrieve order ord_2012.",
        plans=[[("order.lookup", {"order_id": "ord_2012"})]],
    ),
    Task(
        id="task_016",
        prompt="Set customer cus_1004 status to inactive.",
        plans=[[("customer.update", {"customer_id": "cus_1004", "status": "inactive"})]],
        should_mutate=True,
    ),
    Task(
        id="task_017",
        prompt="Cancel delivered order ord_2004.",
        plans=[[("order.cancel", {"order_id": "ord_2004"})]],
        should_mutate=True,
        category="expected_rejection",
    ),
]


def evaluate(
    task: Task,
    calls: list[dict[str, Any]],
    tools: list[ToolDef],
    infrastructure_error: str | None = None,
    final: str = "",
    changes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if infrastructure_error:
        return {"infrastructure_error": infrastructure_error, "comprehension_scored": False}
    by_name = {t.name: t for t in tools}
    observed = [
        (
            by_name[c["name"]].id if c.get("name", "") in by_name else "unknown",
            c.get("arguments", {}),
        )
        for c in calls
    ]
    valid_first = {p[0][0] for p in task.plans if p}
    expected_no_tool = any(not p for p in task.plans)
    if not calls:
        selected = "no_tool"
    elif calls[0].get("name", "") not in by_name:
        selected = "unknown_tool"
    elif not isinstance(calls[0].get("arguments"), dict):
        selected = "invalid_call"
    else:
        selected = observed[0][0]
    selection = selected == "no_tool" if expected_no_tool else selected in valid_first
    schemas = {t.id: t.input_schema for t in tools}
    schema_ok = [
        bool(
            c.get("name", "") in by_name
            and isinstance(c.get("arguments"), dict)
            and Draft202012Validator(
                schemas[by_name[c["name"]].id], format_checker=FormatChecker()
            ).is_valid(c["arguments"])
        )
        for c in calls
    ]

    def normalized_args(call: dict[str, Any]) -> dict[str, Any]:
        raw_args = call.get("arguments", {})
        if not isinstance(raw_args, dict):
            return {}
        args = dict(raw_args)
        tool = by_name.get(call.get("name", ""))
        if tool and "id" in args:
            if tool.id.startswith("customer."):
                args["customer_id"] = args.pop("id")
            elif tool.id.startswith("order."):
                args["order_id"] = args.pop("id")
        return args

    def matched_indices(plan: list[tuple[str, dict[str, Any]]]) -> list[int]:
        indices: list[int] = []
        cursor = 0
        for expected_tool, expected_args in plan:
            found = next(
                (
                    index
                    for index in range(cursor, len(calls))
                    if observed[index][0] == expected_tool
                    and normalized_args(calls[index]) == expected_args
                    and (task.category == "expected_rejection" or not calls[index].get("is_error"))
                ),
                None,
            )
            if found is None:
                return []
            indices.append(found)
            cursor = found + 1
        return indices

    correct_args = [
        any(
            (
                by_name.get(c.get("name", ""))
                and by_name[c["name"]].id == tid
                and normalized_args(c) == args
            )
            for plan in task.plans
            for tid, args in plan
        )
        for c in calls
    ]
    complete_plans = [
        indices for plan in task.plans if plan for indices in [matched_indices(plan)] if indices
    ]
    matched = max(complete_plans, key=len, default=[])
    requires_calls = any(task.plans)
    success = bool(matched) if requires_calls else not calls
    clarification = "?" in final or any(
        word in final.lower() for word in ("which", "what", "could you", "please provide")
    )
    if task.category == "clarification":
        success = not calls and clarification
    if task.category == "expected_rejection":
        refusal = any(
            word in final.lower() for word in ("cannot", "can't", "unable", "not be cancelled")
        )
        success = (
            bool(matched) and any(calls[index].get("is_error") for index in matched) and refusal
        )
    changes = changes or []
    mutation_calls = [
        c
        for c in calls
        if c.get("name", "") in by_name
        and by_name[c["name"]].id in {"customer.update", "order.cancel"}
    ]
    unauthorized = bool(mutation_calls) and not task.should_mutate

    def change_matches_plan(change: dict[str, Any]) -> bool:
        for plan in task.plans:
            for tool_id, args in plan:
                if change["tool"] != tool_id:
                    continue
                if (
                    tool_id == "customer.update"
                    and change.get("customer_id") == args.get("customer_id")
                    and change.get("after")
                    == {k: v for k, v in args.items() if k in {"email", "status"}}
                ):
                    return True
                if (
                    tool_id == "order.cancel"
                    and change.get("order_id") == args.get("order_id")
                    and change.get("after") == "cancelled"
                ):
                    return True
        return False

    def mutation_call_matches_plan(call: dict[str, Any]) -> bool:
        canonical = by_name[call["name"]].id
        args = normalized_args(call)
        return any(
            canonical == tool_id and args == expected_args
            for plan in task.plans
            for tool_id, expected_args in plan
            if tool_id in {"customer.update", "order.cancel"}
        )

    incorrect_mutation = any(not change_matches_plan(change) for change in changes) or any(
        not mutation_call_matches_plan(call) for call in mutation_calls
    )
    first_bad = bool(calls) and (
        selected not in valid_first or not schema_ok[0] or not correct_args[0]
    )
    recovered = first_bad and success
    return {
        "selection_correct": selection,
        "expected_first": sorted(valid_first) if valid_first else "no_tool",
        "selected_first": selected,
        "schema_valid": all(schema_ok) if schema_ok else None,
        "semantic_arguments_correct": all(correct_args) if correct_args else not calls,
        "task_success": success,
        "call_count": len(calls),
        "unnecessary_calls": max(0, len(calls) - len(matched)),
        "recovered": recovered,
        "unauthorized_mutation": unauthorized,
        "incorrect_mutation": incorrect_mutation,
        "comprehension_scored": True,
    }


def confusion_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for row in results:
        ev = row["evaluation"]
        if not ev.get("comprehension_scored"):
            continue
        exp = ev.get("expected_first")
        if isinstance(exp, list):
            if len(exp) != 1:
                continue
            exp = exp[0]
        key = (str(exp), str(ev.get("selected_first", "no_tool")))
        counts[key] = counts.get(key, 0) + 1
    return [{"expected": k[0], "selected": k[1], "count": v} for k, v in sorted(counts.items())]
