from agent_tool_selection_tester.core import (
    TASKS,
    Fixture,
    catalog,
    confusion_rows,
    evaluate,
    make_variant,
)


def test_mutations_are_repeatable_and_preserve_identity() -> None:
    original = [tool.model_dump() for tool in catalog()]
    one = make_variant("similar_tool_name", 19)
    two = make_variant("similar_tool_name", 19)
    assert one.catalog_hash == two.catalog_hash
    assert one.changed_fields == ["name"]
    assert [x.id for x in one.tools] == [x["id"] for x in original]
    assert len({x.name for x in one.tools}) == len(one.tools)
    assert [tool.model_dump() for tool in catalog()] == original


def test_parameter_alias_and_order_variant() -> None:
    alias = make_variant("parameter_alias")
    customer = next(t for t in alias.tools if t.id == "customer.lookup_by_id")
    assert "id" in customer.input_schema["properties"]
    assert "customer_id" not in customer.input_schema["properties"]
    order = make_variant("tool_order_shuffle", 12)
    assert order.order != [t.name for t in catalog()]


def test_fixture_resets_and_records_mutation() -> None:
    first, second = Fixture.fresh(), Fixture.fresh()
    first.call("customer.update", {"customer_id": "cus_1001", "email": "new@example.com"})
    assert first.changes
    assert second.customers["cus_1001"]["email"] == "alex.morgan@example.com"


def test_evaluation_and_matrix() -> None:
    task = TASKS[0]
    tools = catalog()
    correct = evaluate(
        task, [{"name": "get_customer", "arguments": {"customer_id": "cus_1001"}}], tools
    )
    wrong = evaluate(task, [{"name": "search_customer", "arguments": {"query": "Alex"}}], tools)
    assert correct["selection_correct"] and correct["task_success"]
    assert not wrong["selection_correct"]
    assert confusion_rows([{"evaluation": correct}, {"evaluation": wrong}])[0]["count"] >= 1


def test_infrastructure_error_is_not_scored() -> None:
    result = evaluate(TASKS[0], [], catalog(), infrastructure_error="timeout")
    assert result == {"infrastructure_error": "timeout", "comprehension_scored": False}


def test_schema_conditions_are_marked_contract_changing() -> None:
    variant = make_variant("schema_customer_id_optional")
    assert not variant.semantic_preserving
    assert variant.condition_class == "contract-changing"
    assert (
        next(t for t in variant.tools if t.id == "customer.lookup_by_id").input_schema["required"]
        == []
    )


def test_recovery_and_clarification_are_evaluated_separately() -> None:
    tools = catalog()
    task = TASKS[0]
    recovered = evaluate(
        task,
        [
            {"name": "search_customer", "arguments": {"query": "Alex"}},
            {"name": "get_customer", "arguments": {"customer_id": "cus_1001"}},
        ],
        tools,
    )
    assert recovered["task_success"] and recovered["recovered"]
    clarify = evaluate(TASKS[11], [], tools, final="Which order ID should I cancel?")
    assert clarify["task_success"]
    assert not evaluate(TASKS[11], [], tools, final="I cannot do that.")["task_success"]


def test_unsupported_variant_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unsupported mutation"):
        make_variant("schema_anything_else")


def test_expected_rejection_is_successful_without_a_state_change() -> None:
    task = TASKS[-1]
    call = {"name": "cancel_order", "arguments": {"order_id": "ord_2004"}, "is_error": True}
    result = evaluate(
        task, [call], catalog(), final="That order was delivered and cannot be cancelled."
    )
    assert result["task_success"]
    assert not result["unauthorized_mutation"]
    assert not result["incorrect_mutation"]


def test_confusion_matrix_has_unknown_and_invalid_call_columns() -> None:
    task = TASKS[0]
    unknown = evaluate(task, [{"name": "missing_tool", "arguments": {}}], catalog())
    invalid = evaluate(task, [{"name": "get_customer", "arguments": "not-json"}], catalog())
    rows = confusion_rows([{"evaluation": unknown}, {"evaluation": invalid}])
    assert {row["selected"] for row in rows} == {"unknown_tool", "invalid_call"}
