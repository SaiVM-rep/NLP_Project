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


# ------------------------------------------------------- sliding window ---
# A token bucket (capacity C, refill C/60 per s) admits up to 2C in a 60 s window:
# a full burst plus a minute of refill. Providers count per rolling minute, so RPM/TPM
# need an exact window (NFR-3, FR-3.3).


class WClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


async def test_token_bucket_admits_double_the_limit_in_a_minute() -> None:
    """Documents why SlidingWindow exists."""
    from app.llm.limiter import TokenBucket

    clock = WClock()
    b = TokenBucket(10 / 60, 10, clock=clock, sleep=clock.sleep)
    times = []
    for _ in range(20):
        await b.acquire()
        times.append(clock.now)
    assert sum(1 for t in times if t < 60) > 10


async def test_window_never_exceeds_limit_in_any_60s() -> None:
    from app.llm.limiter import SlidingWindow

    clock = WClock()
    w = SlidingWindow(10, clock=clock, sleep=clock.sleep)
    times = []
    for _ in range(35):
        await w.acquire()
        times.append(clock.now)
    for t0 in times:
        assert sum(1 for t in times if t0 <= t < t0 + 60) <= 10
    assert times[9] == 0.0 and times[10] == 60.0  # full burst, then wait for expiry


async def test_window_weighted_cost_for_tpm() -> None:
    from app.llm.limiter import SlidingWindow

    clock = WClock()
    w = SlidingWindow(100, clock=clock, sleep=clock.sleep)
    await w.acquire(60)
    await w.acquire(30)
    await w.acquire(30)  # 120 > 100: waits until the first 60 expires
    assert clock.now == 60.0


async def test_window_cost_over_limit_raises() -> None:
    from app.llm.limiter import SlidingWindow

    w = SlidingWindow(100)
    with pytest.raises(ValueError):
        await w.acquire(101)


async def test_window_exposes_capacity_like_the_bucket() -> None:
    from app.llm.limiter import SlidingWindow

    assert SlidingWindow(250_000).capacity == 250_000
