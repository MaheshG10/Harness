from __future__ import annotations

import argparse
import sys
import time

from .config import Settings
from .llm import GenerateConfig, LLMError, Message


def cmd_check(args: argparse.Namespace) -> int:
    s = Settings.from_env()
    if not s.api_key:
        print("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1
    client = s.client(args.model, cache_mode="off")
    t0 = time.perf_counter()
    try:
        resp = client.generate(
            [Message.user("Reply with exactly one word: ok")],
            config=GenerateConfig(temperature=0, max_output_tokens=256),
        )
    except LLMError as e:
        print(f"failed: {e}")
        return 1
    ms = (time.perf_counter() - t0) * 1000
    print(f"model    {client.model} -> {resp.model_version}")
    print(f"reply    {(resp.message.text or '').strip()!r}")
    print(f"tokens   in={resp.usage.input_tokens} out={resp.usage.output_tokens} "
          f"thinking={resp.usage.thinking_tokens}")
    print(f"latency  {ms:.0f} ms")
    print(f"today    {client.limiter.used_today} requests used" +
          (f" of {s.rpd}" if s.rpd else ""))
    return 0


def _short(value, n: int = 60) -> str:
    s = value if isinstance(value, str) else repr(value)
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[: n - 3] + "..."


def print_step(step) -> None:
    from .agent import Step

    assert isinstance(step, Step)
    tag = " (cached)" if step.from_cache else ""
    print(f"step {step.index + 1:>2}  {step.usage.total:>6} tok  {step.latency:5.1f}s{tag}")
    if step.text:
        print(f"         says: {_short(step.text, 100)}")
    for c, r in zip(step.tool_calls, step.results):
        args = ", ".join(f"{k}={_short(v, 40)}" for k, v in c.args.items())
        mark = "x" if r.is_error else ">"
        first = r.content.strip().splitlines()[0] if r.content.strip() else ""
        print(f"         {c.name}({args})")
        print(f"           {mark} {_short(first, 90)}")


def cmd_run(args: argparse.Namespace) -> int:
    from .agent import Agent, AgentConfig, ToolContext, build_registry
    from .llm import QuotaExhausted
    from .sandbox import DockerError, make_sandbox
    from .task import load_task

    s = Settings.from_env()
    task = load_task(args.task)
    client = s.client(args.model)
    agent = Agent(
        client,
        build_registry(),
        AgentConfig(max_steps=args.max_steps, seed=args.seed, temperature=args.temperature),
        on_step=print_step,
    )
    print(f"task {task.id} | model {client.model} | sandbox {args.sandbox}\n")
    try:
        with make_sandbox(args.sandbox, task.repo, image=args.image) as sb:
            result = agent.run(task.prompt, ToolContext(sb))
            check = sb.exec(task.test_command, timeout=task.test_timeout)
    except DockerError as e:
        print(f"sandbox error: {e}")
        return 1
    except QuotaExhausted as e:
        print(f"stopped: {e}")
        return 1
    passed = check.exit_code == 0
    print()
    print(f"status   {result.status}" + (f" ({result.error})" if result.error else ""))
    print(f"summary  {result.summary or '-'}")
    print(f"tests    {'PASS' if passed else 'FAIL'}")
    print(f"steps    {len(result.steps)}, tool calls {result.tool_calls}, tool errors {result.tool_errors}")
    print(f"tokens   {result.usage.total} (in {result.usage.input_tokens}, out {result.usage.output_tokens}, "
          f"thinking {result.usage.thinking_tokens})")
    print(f"time     {result.duration:.1f}s | model {result.model_version or '-'}")
    return 0 if passed else 2


def cmd_build(args: argparse.Namespace) -> int:
    from .sandbox import DockerError, build_image

    try:
        build_image(args.image)
    except DockerError as e:
        print(e)
        return 1
    print(f"built {args.image}")
    return 0


def main(argv: list[str] | None = None) -> int:
    from .sandbox import DEFAULT_IMAGE

    p = argparse.ArgumentParser(prog="harness")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="make one tiny call to confirm the API key works")
    c.add_argument("--model", default=None)
    c.set_defaults(fn=cmd_check)

    r = sub.add_parser("run", help="run the agent on one task folder")
    r.add_argument("task", help="folder with task.yaml and repo/")
    r.add_argument("--model", default=None)
    r.add_argument("--sandbox", choices=["docker", "local"], default="docker")
    r.add_argument("--image", default=DEFAULT_IMAGE)
    r.add_argument("--max-steps", type=int, default=30)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--temperature", type=float, default=0.7)
    r.set_defaults(fn=cmd_run)

    b = sub.add_parser("build-sandbox", help="build the docker image the sandbox uses")
    b.add_argument("--image", default=DEFAULT_IMAGE)
    b.set_defaults(fn=cmd_build)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
