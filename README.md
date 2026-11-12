# Harness

An agent harness for coding tasks, with its own evaluation harness built in.

The agent works inside a Docker sandbox and fixes code with a small set of tools.
The eval side runs it on a task suite and measures which design choices actually
help, with confidence intervals instead of single numbers.

> Work in progress. Results and full docs land at the end of the build.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add your GEMINI_API_KEY
harness check          # one tiny call to confirm the key works
pytest
```

## Layout

| path | what it does |
|---|---|
| `harness/llm/` | Gemini client: retries, rate limits, token usage, record/replay cache |
| `harness/tools/` | tool registry: schemas from type hints, safe dispatch |
| `tests/` | unit tests, no network needed |
