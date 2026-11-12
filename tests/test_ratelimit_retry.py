import random

import pytest

from harness.llm.errors import QuotaExhausted, TransientError
from harness.llm.ratelimit import RateLimiter
from harness.llm.retry import RetryPolicy, call_with_retry


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def test_rpm_waits_for_window_to_slide():
    clock = FakeClock()
    lim = RateLimiter(rpm=2, clock=clock, sleep=clock.sleep)
    lim.acquire()
    lim.acquire()
    lim.acquire()  # third call must wait out the 60s window
    assert clock.t == pytest.approx(60.0)


def test_tpm_waits_until_enough_tokens_free():
    clock = FakeClock()
    lim = RateLimiter(tpm=1000, clock=clock, sleep=clock.sleep)
    t1 = lim.acquire(600)
    clock.t = 10
    lim.acquire(300)
    clock.t = 20
    lim.acquire(300)  # 1200 > 1000, so wait for the first request to age out
    assert clock.t == pytest.approx(60.0)
    lim.settle(t1, 50)
    assert t1[1] == 50


def test_daily_budget_persists_and_stops(tmp_path):
    state = tmp_path / "day.json"
    lim = RateLimiter(rpd=2, state_path=state, today=lambda: "2026-11-12")
    lim.acquire()
    lim.acquire()
    again = RateLimiter(rpd=2, state_path=state, today=lambda: "2026-11-12")
    assert again.used_today == 2
    with pytest.raises(QuotaExhausted):
        again.acquire()


def test_daily_budget_resets_next_day(tmp_path):
    state = tmp_path / "day.json"
    day = ["2026-11-12"]
    lim = RateLimiter(rpd=1, state_path=state, today=lambda: day[0])
    lim.acquire()
    day[0] = "2026-11-13"
    lim.acquire()
    assert lim.used_today == 1


def test_retry_uses_server_hint_then_succeeds():
    clock = FakeClock()
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise TransientError(429, "slow down", retry_after=5.0)
        return "ok"

    out = call_with_retry(flaky, RetryPolicy(), sleep=clock.sleep, rng=random.Random(0))
    assert out == "ok"
    assert len(clock.slept) == 2
    assert all(5.0 <= s <= 6.0 for s in clock.slept)


def test_retry_gives_up_after_max_attempts():
    clock = FakeClock()

    def always_fails():
        raise TransientError(503, "down")

    with pytest.raises(TransientError):
        call_with_retry(always_fails, RetryPolicy(max_attempts=3), sleep=clock.sleep)
    assert len(clock.slept) == 2


def test_backoff_is_capped():
    p = RetryPolicy(base_delay=2, max_delay=10)
    rng = random.Random(1)
    assert all(p.delay(a, None, rng) <= 10 for a in range(20))
