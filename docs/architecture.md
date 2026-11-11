# Architecture

Agent Tool Selection Tester keeps the tool presentation layer separate from fixture semantics. `ToolDef.id` is canonical; `ToolDef.name`, description, and JSON Schema are exposed to MCP and Ollama. `mcp_runtime` creates an MCP SDK v2 low-level `Server` with explicit `on_list_tools` and `on_call_tool` handlers. The runner connects using the SDK's in-process `Client(server)` transport, so discovery and invocation use real MCP operations without a per-case process or port.

The experiment runner starts a new fixture snapshot for every task, repetition, and condition. The runner converts the discovered schema into Ollama's function-calling format. Calls go back through `Client.call_tool`; a variant-name map routes public names to the correct fixture function. Parameter aliases are normalized only at this boundary.

The evaluator is a pure Python rules module. It does not depend on Streamlit or call a model. SQLite contains an append-only run table and case rows with full trace/evaluation JSON. The dashboard and exporters read persisted records; neither invokes the model.

## Data flow

```text
YAML config + ground-truth tasks
  -> controlled catalog condition
  -> in-process MCP tools/list
  -> Ollama structured tool call
  -> in-process MCP tools/call
  -> rule evaluator + trace
  -> append-only SQLite
  -> CLI / JSON + CSV / Streamlit + Plotly
```
