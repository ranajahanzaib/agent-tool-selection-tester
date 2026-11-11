# Experiment format

The bundled YAML config selects the local model, generation settings, repetitions, variant seed, round limit, and single-dimension mutation list. Semantic-preserving mutations are `description_paraphrase`, `description_shortened`, `description_elaborated`, `similar_tool_name`, `parameter_alias`, `parameter_description`, and `tool_order_shuffle`. The separately labeled contract-changing conditions are `schema_customer_id_optional` and `schema_misleading_query`.

```yaml
name: customer-tool-comprehension
model:
  provider: ollama
  name: qwen3:4b-instruct
  temperature: 0
  seed: 42
  max_tool_rounds: 4
experiment:
  repetitions: 3
  variant_seed: 42
mutations:
  - type: similar_tool_name
  - type: tool_order_shuffle
```

Ground truth is bundled separately in `customer-tasks.yaml`; each task has one or more ordered `plans`, where each step includes a canonical tool ID and expected arguments. An empty plan represents a task that requires no tool call (including the clarification case). `should_mutate` identifies tasks that explicitly authorize fixture writes. The CLI uses this YAML task file for the built-in experiment.
