# Methodology

Agent Tool Selection Tester uses paired tasks across a baseline and one-variable-at-a-time interface variants. Each run repeats each condition and task, resets fixture state, and records the actual public tool order passed into inference. The baseline tool definitions remain unchanged; variants preserve the canonical ID and handler.

The evaluator checks observable tool calls against explicit accepted plans. It distinguishes initial tool selection, exposed-schema validity, expected argument values, task plan completion, and unauthorized write attempts. It does not use a second model as a judge. Provider/runtime errors are retained but excluded from comprehension rates.

Small sample percentages are descriptive only. Temperature zero and a seed do not guarantee bit-for-bit inference across model digests, Ollama releases, and hardware. Compare raw counts and sample sizes and inspect traces before drawing conclusions. The two bundled schema conditions intentionally change the input contract and are labeled separately from semantics-preserving variants. They retain the same task expectations, so their effect on schema validity and task completion is visible in the results.
