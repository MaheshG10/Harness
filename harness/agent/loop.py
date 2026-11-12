"""The agent loop.

  ask the model -> run the tool calls it made -> feed results back -> repeat

until it calls `finish` or hits a budget. Every turn is recorded as a Step so a
run can be inspected, replayed and scored later.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol

from ..llm.errors import CacheMiss, LLMError, QuotaExhausted
from ..llm.types import GenerateConfig, LLMResponse, Message, ToolCall, ToolResult, ToolSpec, Usage
from ..tools import ToolRegistry
from .context import ContextStrategy, KeepAll
from .prompts import NUDGE_MALFORMED, NUDGE_NO_TOOL, SYSTEM_PROMPT
from .tools import ToolContext

Status = Literal["finished", "max_steps", "token_budget", "no_action", "error"]


class LLM(Protocol):
    model: str

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        system: str | None = None,
        config: GenerateConfig | None = None,
    ) -> LLMResponse: ...


@dataclass
class AgentConfig:
    max_steps: int = 30
    max_total_tokens: int = 600_000
    max_idle_turns: int = 2  # consecutive replies with no tool call
    temperature: float = 0.7
    seed: int | None = None
    max_output_tokens: int = 8192


@dataclass
class Step:
    index: int
    text: str | None
    tool_calls: list[ToolCall]
    results: list[ToolResult]
    usage: Usage
    latency: float
    from_cache: bool
    finish_reason: str | None
    context_messages: int  # how many messages the model was shown


@dataclass
class RunResult:
    status: Status
    steps: list[Step] = field(default_factory=list)
    summary: str | None = None
    error: str | None = None
    usage: Usage = field(default_factory=Usage)
    duration: float = 0.0
    files_changed: list[str] = field(default_factory=list)
    model_version: str | None = None
    messages: list[Message] = field(default_factory=list)

    @property
    def tool_calls(self) -> int:
        return sum(len(s.tool_calls) for s in self.steps)

    @property
    def tool_errors(self) -> int:
        return sum(r.is_error for s in self.steps for r in s.results)


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        context: ContextStrategy | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        on_step: Callable[[Step], None] | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.config = config or AgentConfig()
        self.context = context or KeepAll()
        self.system_prompt = system_prompt
        self.on_step = on_step

    def run(self, task: str, ctx: ToolContext) -> RunResult:
        cfg = self.config
        gen = GenerateConfig(cfg.temperature, cfg.max_output_tokens, cfg.seed)
        system = self.system_prompt.format(workdir=ctx.sandbox.workdir)
        specs = self.tools.specs()
        messages: list[Message] = [Message.user(task)]
        result = RunResult(status="max_steps", messages=messages)
        t_start = time.perf_counter()
        idle = 0

        def done(status: Status, error: str | None = None) -> RunResult:
            result.status = status
            result.error = error
            result.summary = ctx.summary
            result.duration = time.perf_counter() - t_start
            result.files_changed = sorted(ctx.files_changed)
            return result

        for i in range(cfg.max_steps):
            view = self.context.prepare(messages)
            t0 = time.perf_counter()
            try:
                resp = self.llm.generate(view, specs, system, gen)
            except QuotaExhausted:
                raise  # the runner pauses the whole experiment, not just this task
            except (CacheMiss, LLMError) as e:
                return done("error", f"{type(e).__name__}: {e}")
            latency = time.perf_counter() - t0

            result.usage = result.usage + resp.usage
            result.model_version = resp.model_version or result.model_version
            msg = resp.message
            results: list[ToolResult] = []

            if msg.tool_calls:
                messages.append(msg)
                idle = 0
                for call in msg.tool_calls:
                    results.append(self.tools.call(call, ctx))
                    if ctx.finished:
                        break
                # every call needs a response, even ones skipped after finish
                for call in msg.tool_calls[len(results):]:
                    results.append(ToolResult(call.name, "skipped: run already finished", True, call.id))
                messages.append(Message.results(results))
            else:
                if msg.text:
                    messages.append(msg)
                idle += 1
                nudge = NUDGE_MALFORMED if resp.finish_reason == "MALFORMED_FUNCTION_CALL" else NUDGE_NO_TOOL
                last = messages[-1]
                if last.role == "user" and last.text is None:
                    last.text = nudge  # keep user/model turns alternating
                else:
                    messages.append(Message.user(nudge))

            step = Step(
                index=i,
                text=msg.text,
                tool_calls=list(msg.tool_calls),
                results=results,
                usage=resp.usage,
                latency=latency,
                from_cache=resp.from_cache,
                finish_reason=resp.finish_reason,
                context_messages=len(view),
            )
            result.steps.append(step)
            if self.on_step:
                self.on_step(step)

            if ctx.finished:
                return done("finished")
            if idle >= cfg.max_idle_turns:
                return done("no_action")
            if result.usage.total >= cfg.max_total_tokens:
                return done("token_budget")

        return done("max_steps")
