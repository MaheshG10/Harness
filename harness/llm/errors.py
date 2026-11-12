from __future__ import annotations


class LLMError(Exception):
    """Base error for model calls."""


class RequestError(LLMError):
    """The request itself is wrong (bad schema, bad key). Retrying won't help."""

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class TransientError(LLMError):
    """Rate limit or server hiccup. Safe to retry after a delay."""

    def __init__(self, status: int, message: str, retry_after: float | None = None):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.retry_after = retry_after


class QuotaExhausted(LLMError):
    """A daily quota is used up. Retrying today is pointless, so the runner
    should save progress and stop."""


class CacheMiss(LLMError):
    """Replay mode found no recorded response for this request."""
