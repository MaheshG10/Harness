"""Provider-neutral message types.

The agent loop only ever sees these. The Gemini client converts them to and
from the wire format, so swapping providers later touches one file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "model"]


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str | None = None
    # Gemini 3+ attaches an opaque thought signature to function calls.
    # It must be sent back unchanged on the next turn or the API rejects it.
    signature: str | None = None


@dataclass
class ToolResult:
    name: str
    content: str
    is_error: bool = False
    id: str | None = None


@dataclass
class Message:
    role: Role
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    text_signature: str | None = None

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role="user", text=text)

    @classmethod
    def results(cls, results: list[ToolResult]) -> "Message":
        return cls(role="user", tool_results=results)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.thinking_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.thinking_tokens + other.thinking_tokens,
            self.cached_tokens + other.cached_tokens,
        )


@dataclass
class LLMResponse:
    message: Message
    usage: Usage
    finish_reason: str | None = None
    model_version: str | None = None
    from_cache: bool = False


@dataclass
class GenerateConfig:
    temperature: float = 0.7
    max_output_tokens: int = 8192
    seed: int | None = None
