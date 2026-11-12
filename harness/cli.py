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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="harness")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="make one tiny call to confirm the API key works")
    c.add_argument("--model", default=None)
    c.set_defaults(fn=cmd_check)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
