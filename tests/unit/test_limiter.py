"""TokenBucket and backoff. PRD §11.3, NFR-3.

These run with a simulated clock: no real sleeping, no network.
"""

import pytest

from app.llm.limiter import TokenBucket, backoff_delay


class FakeClock:
    """Monotonic clock that only moves when the bucket sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


async def test_acquire_within_capacity_does_not_sleep():
    clock = FakeClock()
    bucket = TokenBucket(rate_per_sec=1.0, capacity=5.0, clock=clock, sleep=clock.sleep)

    for _ in range(5):
        await bucket.acquire()

    assert clock.slept == []


async def test_acquire_beyond_capacity_waits_for_refill():
    clock = FakeClock()
    bucket = TokenBucket(rate_per_sec=2.0, capacity=1.0, clock=clock, sleep=clock.sleep)

    await bucket.acquire()  # drains the bucket
    await bucket.acquire()  # must wait 1 token / 2 per sec = 0.5s

    assert clock.slept == pytest.approx([0.5])


async def test_refill_is_proportional_to_elapsed_time():
    clock = FakeClock()
    bucket = TokenBucket(rate_per_sec=10.0, capacity=10.0, clock=clock, sleep=clock.sleep)

    await bucket.acquire(cost=10.0)
    clock.now += 0.3  # 3 tokens refilled

    await bucket.acquire(cost=3.0)

    assert clock.slept == []


async def test_refill_never_exceeds_capacity():
    clock = FakeClock()
    bucket = TokenBucket(rate_per_sec=1.0, capacity=2.0, clock=clock, sleep=clock.sleep)

    clock.now += 1000.0  # would be 1000 tokens if uncapped
    await bucket.acquire(cost=2.0)
    await bucket.acquire(cost=1.0)  # capacity was 2, so this one must wait

    assert clock.slept == pytest.approx([1.0])


async def test_cost_larger_than_capacity_is_rejected():
    clock = FakeClock()
    bucket = TokenBucket(rate_per_sec=1.0, capacity=5.0, clock=clock, sleep=clock.sleep)

    with pytest.raises(ValueError, match="exceeds bucket capacity"):
        await bucket.acquire(cost=6.0)


def test_rejects_nonpositive_rate_and_capacity():
    with pytest.raises(ValueError):
        TokenBucket(rate_per_sec=0.0, capacity=1.0)
    with pytest.raises(ValueError):
        TokenBucket(rate_per_sec=1.0, capacity=0.0)


def test_backoff_delay_is_within_full_jitter_bounds():
    for attempt in range(8):
        for _ in range(50):
            d = backoff_delay(attempt, base=1.0, cap=60.0)
            assert 0.0 <= d <= min(60.0, 1.0 * 2**attempt)


def test_backoff_delay_is_capped():
    assert all(backoff_delay(20, base=1.0, cap=60.0) <= 60.0 for _ in range(50))


def test_backoff_delay_varies():
    """Full jitter, not deterministic exponential."""
    samples = {backoff_delay(5, base=1.0, cap=60.0) for _ in range(50)}
    assert len(samples) > 1
