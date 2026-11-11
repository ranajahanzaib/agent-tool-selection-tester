# Contributing

Contributions that improve Agent Tool Selection Tester's correctness, usability, and documentation are welcome.

## Development setup

Use Python 3.12 or newer and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/ranajahanzaib/agent-tool-selection-tester.git
cd agent-tool-selection-tester
uv sync --extra dev
```

Ollama is optional for development and is not needed for the automated test suite. Tests use the deterministic fixture and scripted model adapter.

## Before opening a pull request

Run the same checks used in CI:

```bash
uv run ruff check src tests
uv run mypy src/agent_tool_selection_tester
uv run pytest
```

For behavior changes, include or update tests and user-facing documentation where needed. Keep generated experiment examples fictional, and label any synthetic results clearly. Do not commit credentials, local databases, model files, or other generated artifacts.

Open an issue for bugs or feature proposals and a pull request for a focused change. Describe observable behavior, relevant test coverage, and any compatibility impact.
