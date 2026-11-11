"""Streamlit analysis dashboard for stored Agent Tool Selection Tester runs."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from agent_tool_selection_tester.core import confusion_rows
from agent_tool_selection_tester.experiment import aggregate
from agent_tool_selection_tester.storage import Store

st.set_page_config(page_title="Agent Tool Selection Tester · Experiment analysis", layout="wide")
st.markdown(
    "<style>.block-container{max-width:1500px;padding-top:1.5rem}.stMetric{border:1px solid #283242;padding:.6rem;border-radius:8px}</style>",
    unsafe_allow_html=True,
)
st.title("Agent Tool Selection Tester")
st.caption(
    "Local tool comprehension experiments · measurements are observations from this run, not universal model claims"
)

try:
    store = Store()
    runs = store.runs()
    if not runs:
        st.info("No experiment runs found. Use `agent-tool-selection-tester run` to create a local Ollama experiment.")
        st.stop()
    labels = {f"{r['name']} · {r['id'][:8]} · {r['started_at']}": r["id"] for r in runs}
    choice = st.sidebar.selectbox("Experiment run", list(labels))
    info, all_cases = store.get_run(labels[choice])
    store.close()
except Exception as exc:  # noqa: BLE001 - report database and serialization errors in the UI.
    st.error(f"Could not read Agent Tool Selection Tester's local result store: {exc}")
    st.stop()

manifest = json.loads(info["manifest"])
run_config = json.loads(info["config"])
if manifest.get("synthetic"):
    st.warning(manifest.get("disclosure", "Synthetic demonstration data; not model measurements."))
variants = sorted({r["variant_id"] for r in all_cases})
names = {r["variant_id"]: r["mutation_type"] for r in all_cases}
selected = st.sidebar.multiselect(
    "Conditions", variants, default=variants, format_func=lambda x: names.get(x, x)
)
cases = [r for r in all_cases if r["variant_id"] in selected]
scored = [r for r in cases if r["evaluation"].get("comprehension_scored")]
summary = aggregate(cases)
st.subheader(info["name"])
planned_cases = (
    len(all_cases)
    if manifest.get("synthetic")
    else (len(run_config.get("mutations", [])) + int(run_config.get("include_baseline", True)))
    * len(manifest.get("tasks", []))
    * int(run_config.get("repetitions", manifest.get("repetitions", 1)))
)
run_status = "complete" if len(all_cases) >= planned_cases else "partial"
st.caption(
    f"Model: {manifest.get('model')} · Run: `{info['id']}` · Status: {run_status} ({len(all_cases)}/{planned_cases} cases) · Tasks: {len(manifest.get('tasks', []))} · Conditions: {len(variants)} · Repetitions: {run_config.get('repetitions', manifest.get('repetitions', 1))}"
)
metric_specs = [
    ("First selection", "selection_correct", scored, False),
    (
        "Schema validity",
        "schema_valid",
        [r for r in scored if r["evaluation"].get("schema_valid") is not None],
        False,
    ),
    (
        "Argument accuracy",
        "semantic_arguments_correct",
        [r for r in scored if r["evaluation"].get("call_count", 0) > 0],
        False,
    ),
    ("Task success", "task_success", scored, False),
    ("Unauthorized writes", "unauthorized_mutation", scored, True),
    ("Incorrect writes", "incorrect_mutation", scored, True),
]
cols = st.columns(len(metric_specs))
for col, (label, key, metric_rows, count_metric) in zip(cols, metric_specs):
    denom = len(metric_rows)
    count = sum(bool(r["evaluation"].get(key)) for r in metric_rows)
    value = (
        f"{count}/{denom}"
        if count_metric
        else (f"{count / denom:.0%} · {count}/{denom}" if denom else "n/a")
    )
    col.metric(label, value, help=f"{count} of {denom} cases")

left, right = st.columns([1.35, 1])
pair_filter: tuple[str, str] | None = None
with left:
    st.subheader("First tool confusion")
    matrix = confusion_rows(cases)
    if matrix:
        df = pd.DataFrame(matrix)
        pivot = df.pivot_table(
            index="expected", columns="selected", values="count", aggfunc="sum", fill_value=0
        )
        pivot = pivot.reindex(sorted(set(pivot.index) | set(pivot.columns)), axis=0, fill_value=0)
        pivot = pivot.reindex(sorted(set(pivot.index) | set(pivot.columns)), axis=1, fill_value=0)
        row_percentages = st.checkbox("Normalize each expected-tool row", value=False)
        plot_values = (
            pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0) if row_percentages else pivot
        )
        fig = px.imshow(
            plot_values,
            text_auto=".0%" if row_percentages else True,
            color_continuous_scale="Blues",
            labels={
                "x": "Selected canonical tool",
                "y": "Expected canonical tool",
                "color": "Cases",
            },
            aspect="auto",
        )
        if row_percentages:
            fig.update_coloraxes(colorbar_tickformat=".0%")
        fig.update_layout(
            margin={"l": 10, "r": 10, "t": 25, "b": 10}, height=max(360, 34 * len(pivot))
        )
        if row_percentages:
            fig.update_traces(
                customdata=pivot.to_numpy(),
                hovertemplate="Expected: %{y}<br>Selected: %{x}<br>Row share: %{z:.1%}<br>Cases: %{customdata}<extra></extra>",
            )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Diagonal cells are correct selections; off-diagonal cells are confusions. Rows with multiple accepted first tools are omitted."
        )
        pairs = {
            f"{row['expected']} → {row['selected']} (n={row['count']})": (
                row["expected"],
                row["selected"],
            )
            for row in matrix
        }
        selected_pair = st.selectbox("Inspect a confusion cell", ["All cases", *pairs])
        pair_filter = pairs.get(selected_pair)
    else:
        st.info("No scored cases with a single expected first tool in this selection.")
with right:
    st.subheader("Condition comparison")
    if summary:
        comp = pd.DataFrame(summary)
        sort_mode = st.selectbox(
            "Sort conditions",
            ["Experiment order", "Largest observed degradation", "Largest observed improvement"],
        )
        if sort_mode == "Largest observed degradation":
            comp = comp.sort_values("selection_delta_pp", ascending=True, na_position="last")
        elif sort_mode == "Largest observed improvement":
            comp = comp.sort_values("selection_delta_pp", ascending=False, na_position="last")
        base = comp[comp.mutation_type == "baseline"]
        if not base.empty:
            b = float(base.iloc[0].selection_accuracy or 0)
            comp["Selection Δ vs baseline (pp)"] = ((comp.selection_accuracy - b) * 100).round(1)
        st.dataframe(
            comp[
                [
                    "mutation_type",
                    "condition_class",
                    "sample_count",
                    "schema_sample_count",
                    "selection_accuracy",
                    "selection_delta_pp",
                    "schema_validity",
                    "argument_sample_count",
                    "argument_accuracy",
                    "task_success",
                    "unauthorized_mutations",
                    "incorrect_mutations",
                    "recovered_cases",
                    "infrastructure_errors",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        chart = px.bar(
            comp,
            x="mutation_type",
            y="selection_accuracy",
            color="mutation_type",
            hover_data=["sample_count", "task_success"],
            labels={"mutation_type": "Condition", "selection_accuracy": "First selection accuracy"},
        )
        chart.update_layout(
            showlegend=False, yaxis_tickformat=".0%", margin={"l": 10, "r": 10, "t": 10, "b": 10}
        )
        st.plotly_chart(chart, width="stretch")

st.subheader("Execution traces")
trace_cases = cases
if pair_filter:
    expected, chosen = pair_filter
    trace_cases = [
        r
        for r in cases
        if r["evaluation"].get("selected_first") == chosen
        and (
            r["evaluation"].get("expected_first") == expected
            or r["evaluation"].get("expected_first") == [expected]
        )
    ]
if trace_cases:
    trace_choices = {
        f"{r['task_id']} · {names.get(r['variant_id'], r['variant_id'])} · rep {r['repetition'] + 1} · {'success' if r['evaluation'].get('task_success') else 'review'}": i
        for i, r in enumerate(trace_cases)
    }
    trace_label = st.selectbox("Select a case", list(trace_choices))
    row = trace_cases[trace_choices[trace_label]]
    st.write(
        f"**Task:** {row['task_id']} · **Duration:** {row['duration_ms']} ms · **First selected:** {row['evaluation'].get('selected_first')} · **Expected:** {row['evaluation'].get('expected_first')}"
    )
    first_tool_event = next(
        (e for e in row["trace"] if e.get("event") == "tool_call_requested"), None
    )
    if first_tool_event:
        definition_set = manifest.get("tool_catalog", [])
        if row["variant_id"] != "baseline":
            variant_info = next(
                (v for v in manifest.get("variants", []) if v.get("id") == row["variant_id"]), None
            )
            if variant_info:
                definition_set = variant_info.get("after", definition_set)
        first_tool_id = next(
            (t["id"] for t in definition_set if t["name"] == first_tool_event["name"]), "unknown"
        )
        expected = row["evaluation"].get("expected_first")
        if isinstance(expected, list):
            if first_tool_id not in expected:
                st.error(
                    f"First incorrect selection: `{first_tool_event['name']}` resolves to `{first_tool_id}`."
                )
        elif first_tool_id != expected:
            st.error(
                f"First incorrect selection: `{first_tool_event['name']}` resolves to `{first_tool_id}`."
            )
        if row["evaluation"].get("recovered"):
            st.warning("The run recovered after an incorrect first decision.")
    for event in row["trace"]:
        st.markdown(f"`{event.get('event')}`")
        if event.get("event") in {"tool_call_requested", "tool_call_result"}:
            if event.get("event") == "tool_call_result" and event.get("is_error"):
                st.error(f"Tool returned an error: {event.get('content')}")
            st.json(event)
        elif event.get("event") in {"model_request", "model_response"}:
            with st.expander("Model request/response payload"):
                st.json(event)
        elif event.get("event") == "repeated_call_suppressed":
            st.warning("Repeated tool call was suppressed to stop a loop.")
    with st.expander("Raw evaluation and fixture mutations"):
        st.json(
            {
                "evaluation": row["evaluation"],
                "fixture_changes": row["fixture_changes"],
                "fixture_attempts": row.get("fixture_attempts", []),
                "offered_tool_order": row["tool_order"],
            }
        )

st.subheader("Tool inspector")
baseline_tools = {tool["id"]: tool for tool in manifest.get("tool_catalog", [])}
tool_ids = sorted(baseline_tools)
if tool_ids:
    inspected = st.selectbox("Canonical tool", tool_ids)
    condition_ids = [x for x in selected if x != "baseline"]
    chosen_variant = st.selectbox(
        "Interface condition", ["baseline", *condition_ids], format_func=lambda x: names.get(x, x)
    )
    original = baseline_tools[inspected]
    updated = original
    if chosen_variant != "baseline":
        detail = next((v for v in manifest.get("variants", []) if v["id"] == chosen_variant), None)
        if detail:
            updated = next((t for t in detail["after"] if t["id"] == inspected), original)
    affected_tasks = [
        task["id"]
        for task in manifest.get("tasks", [])
        if any(step[0] == inspected for plan in task.get("plans", []) for step in plan)
    ]
    error_cases = [
        r
        for r in cases
        if r["task_id"] in affected_tasks
        and (
            r["evaluation"].get("schema_valid") is False
            or not r["evaluation"].get("semantic_arguments_correct", True)
        )
    ]
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Baseline · `{original['name']}`**")
        st.write(original["description"])
        st.json(original["input_schema"])
    with col_b:
        st.markdown(f"**Selected condition · `{updated['name']}`**")
        st.write(updated["description"])
        st.json(updated["input_schema"])
    st.caption(
        f"Canonical ID: `{inspected}` · Affected tasks: {', '.join(affected_tasks) or 'none'} · Related argument/schema errors in current selection: {len(error_cases)}"
    )

st.subheader("Measured findings")
if summary and any(s["mutation_type"] == "baseline" for s in summary):
    base = next(s for s in summary if s["mutation_type"] == "baseline")
    findings = []
    for s in summary:
        if s["mutation_type"] == "baseline" or s["selection_accuracy"] is None:
            continue
        delta = (s["selection_accuracy"] - base["selection_accuracy"]) * 100
        if abs(delta) >= 0.1:
            findings.append(
                f"{s['mutation_type'].replace('_', ' ').title()} was associated with {abs(delta):.1f} percentage point {'more' if delta > 0 else 'fewer'} correct first selections (variant n={s['sample_count']}; baseline n={base['sample_count']})."
            )
    if findings:
        for finding in findings:
            st.write("• " + finding)
    else:
        st.write(
            f"No first-selection change from baseline was observed for the selected conditions (baseline n={base['sample_count']})."
        )
else:
    st.caption("Select the baseline condition to generate evidence-linked comparisons.")

with st.expander("Reproducibility manifest"):
    st.json(manifest)
