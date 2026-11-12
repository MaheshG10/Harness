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

## Run the agent on a task

```bash
harness build-sandbox                         # once: builds the Docker image
harness run examples/tasks/mean_off_by_one    # watch it work, step by step
```

A task is a folder with a `task.yaml` (the prompt and the test command) and a
`repo/` the agent works on. The repo is copied into a fresh sandbox, so the
original is never touched. After the agent finishes, the test command decides
pass or fail.

No Docker? Add `--sandbox local`. It runs on your machine with CPU and memory
caps, but it is not isolated, so only use it on tasks you trust.

## The sandbox

Each task gets its own container: no network, capped CPU, memory and process
count, a read-only root filesystem, and all Linux capabilities dropped. The
workspace is mounted at `/work`. Command timeouts are enforced inside the
container, because killing `docker exec` on the host leaves the process
running.

## Layout

| path | what it does |
|---|---|
| `harness/llm/` | Gemini client: retries, rate limits, token usage, record/replay cache |
| `harness/tools/` | tool registry: schemas from type hints, safe dispatch |
| `harness/sandbox/` | Docker sandbox and a local fallback |
| `harness/agent/` | the agent loop, its tools and its prompt |
| `examples/tasks/` | three small demo tasks |
| `tests/` | unit and end-to-end tests, no API calls needed |
