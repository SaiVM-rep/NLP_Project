"""Token bucket and retry backoff. PRD §11.3, NFR-3.

Limits come from configs/models.yaml, never hard-coded here.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Protocol


class Limiter(Protocol):
    capacity: float

    async def acquire(self, cost: float = 1.0) -> None: ...


class TokenBucket:
    """Classic token bucket with an injectable clock, so tests never sleep."""

    def __init__(
        self,
        rate_per_sec: float,
        capacity: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.rate_per_sec = rate_per_sec
        self.capacity = capacity
        self._clock = clock
        self._sleep = sleep
        self._tokens = capacity
        self._updated = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = now - self._updated
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_sec)
            self._updated = now

    async def acquire(self, cost: float = 1.0) -> None:
        """Wait until `cost` tokens are available, then consume them."""
        if cost > self.capacity:
            raise ValueError(f"cost {cost} exceeds bucket capacity {self.capacity}")
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= cost:
                    self._tokens -= cost
                    return
                await self._sleep((cost - self._tokens) / self.rate_per_sec)


class SlidingWindow:
    """At most `limit` units of cost in any `window` seconds, exactly.

    Use for RPM (cost 1) and TPM (cost = tokens). A token bucket of capacity C refilling
    C per window admits up to 2C in one window (burst + refill); providers count per
    rolling minute, so that overshoot would come back as 429s (NFR-3, FR-3.3).
    """

    def __init__(
        self,
        limit: float,
        window: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if limit <= 0 or window <= 0:
            raise ValueError("limit and window must be positive")
        self.capacity = limit
        self.window = window
        self._clock = clock
        self._sleep = sleep
        self._events: deque[tuple[float, float]] = deque()  # (time, cost)
        self._lock = asyncio.Lock()

    async def acquire(self, cost: float = 1.0) -> None:
        if cost > self.capacity:
            raise ValueError(f"cost {cost} exceeds window limit {self.capacity}")
        async with self._lock:
            while True:
                now = self._clock()
                while self._events and self._events[0][0] + self.window <= now:
                    self._events.popleft()
                used = sum(c for _, c in self._events)
                if used + cost <= self.capacity:
                    self._events.append((now, cost))
                    return
                # sleep until enough of the oldest events have left the window
                need, freed = used + cost - self.capacity, 0.0
                for t, c in self._events:
                    freed += c
                    if freed >= need:
                        await self._sleep(t + self.window - now)
                        break


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    """Exponential backoff with full jitter."""
    return random.uniform(0, min(cap, base * 2**attempt))
