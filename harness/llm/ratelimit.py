"""Client-side rate limiting for the Gemini free tier.

Free tier quotas are small (requests per minute, tokens per minute, requests
per day). Hitting them costs a 429 plus a long backoff, so it is cheaper to
pace ourselves. Minute limits use a sliding window. The daily count is saved
to disk so it survives restarts, and it resets at midnight Pacific time,
which is when Google resets daily quotas.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from .errors import QuotaExhausted

WINDOW = 60.0
PACIFIC = ZoneInfo("America/Los_Angeles")


class RateLimiter:
    def __init__(
        self,
        rpm: int | None = None,
        tpm: int | None = None,
        rpd: int | None = None,
        state_path: str | Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        today: Callable[[], str] | None = None,
    ):
        self.rpm, self.tpm, self.rpd = rpm, tpm, rpd
        self.state_path = Path(state_path) if state_path else None
        self.clock, self.sleep = clock, sleep
        self.today = today or (lambda: datetime.now(PACIFIC).date().isoformat())
        self._lock = threading.Lock()
        # (timestamp, tokens) for requests inside the current minute window
        self._window: deque[list[float]] = deque()
        self._day, self._day_count = self._load_day()

    def acquire(self, est_tokens: int = 0) -> list[float]:
        """Block until a request fits the limits. Returns a ticket that
        `settle` uses to swap the estimate for the real token count."""
        while True:
            with self._lock:
                self._check_day()
                now = self.clock()
                self._prune(now)
                wait = self._wait_needed(now, est_tokens)
                if wait <= 0:
                    ticket = [now, float(est_tokens)]
                    self._window.append(ticket)
                    self._day_count += 1
                    self._save_day()
                    return ticket
            self.sleep(wait)

    def settle(self, ticket: list[float], actual_tokens: int) -> None:
        with self._lock:
            ticket[1] = float(actual_tokens)

    @property
    def used_today(self) -> int:
        return self._day_count

    def _wait_needed(self, now: float, est_tokens: int) -> float:
        waits = [0.0]
        if self.rpm and len(self._window) >= self.rpm:
            waits.append(self._window[0][0] + WINDOW - now)
        if self.tpm and self._window:
            used = sum(t for _, t in self._window)
            if used + est_tokens > self.tpm:
                # wait until enough old requests age out of the window
                need = used + est_tokens - self.tpm
                freed = 0.0
                for ts, tok in self._window:
                    freed += tok
                    if freed >= need:
                        waits.append(ts + WINDOW - now)
                        break
        return max(waits)

    def _prune(self, now: float) -> None:
        while self._window and now - self._window[0][0] >= WINDOW:
            self._window.popleft()

    def _check_day(self) -> None:
        day = self.today()
        if day != self._day:
            self._day, self._day_count = day, 0
        if self.rpd and self._day_count >= self.rpd:
            raise QuotaExhausted(
                f"daily request budget used ({self._day_count}/{self.rpd}); "
                "resumes after midnight Pacific"
            )

    def _load_day(self) -> tuple[str, int]:
        day = self.today()
        if self.state_path and self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                if data.get("day") == day:
                    return day, int(data.get("count", 0))
            except (ValueError, OSError):
                pass
        return day, 0

    def _save_day(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"day": self._day, "count": self._day_count}))
        os.replace(tmp, self.state_path)
