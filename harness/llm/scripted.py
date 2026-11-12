"""A stand-in model that plays back a fixed script.

Used to test the agent loop end to end (real tools, real sandbox) without
spending API calls, and to demo the harness offline.
"""

from __future__ import annotations

from typing import Callable

from .types import GenerateConfig, LLMResponse, Message, ToolCall, ToolSpec, Usage

Turn = Message | Callable[[list[Message]], Message]


def call(name: str, **args) -> Message:
    return Message(role="model", tool_calls=[ToolCall(name, args)])


def say(text: str) -> Message:
    return Message(role="model", text=text)


class ScriptedLLM:
    model = "scripted"

    def __init__(self, turns: list[Turn], tokens_per_turn: tuple[int, int] = (100, 20)):
        self.turns = list(turns)
        self.tokens = tokens_per_turn
        self.seen: list[list[Message]] = []

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        system: str | None = None,
        config: GenerateConfig | None = None,
    ) -> LLMResponse:
        self.seen.append(list(messages))
        if not self.turns:
            msg = say("(script ended)")
        else:
            turn = self.turns.pop(0)
            msg = turn(messages) if callable(turn) else turn
        return LLMResponse(msg, Usage(*self.tokens), "STOP", "scripted-1")
