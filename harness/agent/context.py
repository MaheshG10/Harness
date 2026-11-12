"""Context strategies decide what part of the history the model sees each turn.

The full history is always kept for tracing. A strategy only shapes the view
sent to the model. More strategies arrive with the context work.
"""

from __future__ import annotations

from ..llm.types import Message


class ContextStrategy:
    name = "base"

    def prepare(self, messages: list[Message]) -> list[Message]:
        raise NotImplementedError


class KeepAll(ContextStrategy):
    """Send everything. The naive baseline."""

    name = "keep_all"

    def prepare(self, messages: list[Message]) -> list[Message]:
        return messages
