from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .types import Usage


@dataclass
class UsageMeter:
    """Thread-safe running totals for one run."""

    calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    usage: Usage = field(default_factory=Usage)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_call(self, usage: Usage, from_cache: bool) -> None:
        with self._lock:
            self.calls += 1
            self.cache_hits += int(from_cache)
            self.usage = self.usage + usage

    def add_retry(self) -> None:
        with self._lock:
            self.retries += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "cache_hits": self.cache_hits,
                "retries": self.retries,
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "thinking_tokens": self.usage.thinking_tokens,
                "cached_tokens": self.usage.cached_tokens,
                "total_tokens": self.usage.total,
            }
