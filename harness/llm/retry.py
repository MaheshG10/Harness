from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from .errors import TransientError

T = TypeVar("T")


@dataclass
class RetryPolicy:
    max_attempts: int = 6
    base_delay: float = 2.0
    max_delay: float = 90.0

    def delay(self, attempt: int, hint: float | None, rng: random.Random) -> float:
        # The server's own hint wins; Gemini sends one on most 429s.
        if hint is not None:
            return min(hint + rng.uniform(0, 1.0), self.max_delay)
        # Full jitter keeps parallel workers from retrying in lockstep.
        cap = min(self.base_delay * (2 ** attempt), self.max_delay)
        return rng.uniform(0, cap)


def call_with_retry(
    fn: Callable[[], T],
    policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, TransientError, float], None] | None = None,
    rng: random.Random | None = None,
) -> T:
    rng = rng or random.Random()
    for attempt in range(policy.max_attempts):
        try:
            return fn()
        except TransientError as e:
            if attempt == policy.max_attempts - 1:
                raise
            wait = policy.delay(attempt, e.retry_after, rng)
            if on_retry:
                on_retry(attempt + 1, e, wait)
            sleep(wait)
    raise AssertionError("unreachable")
