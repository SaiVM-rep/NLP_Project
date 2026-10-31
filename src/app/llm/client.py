"""One LLM interface for every provider. PRD §11.3.

Responsibilities, in order of execution:
  cache lookup -> budget guard -> rate limit -> call -> retry/backoff
  -> fallback policy -> cache store -> llm_calls log (FR-7.4)

A cache hit short-circuits everything that costs quota: it does not touch the
limiter, the budget, or the provider. That is what makes an experiment rerun
free (NFR-1) and deterministic (NFR-6).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Literal

from app.core.config import ModelProfile
from app.llm.base import (
    RETRYABLE,
    BudgetExhausted,
    LLMResult,
    Provider,
    RateLimited,
)
from app.llm.cache import DiskCache, cache_key
from app.llm.limiter import Limiter, backoff_delay
from app.route.tokens import estimate_tokens


class LLMClient:
    def __init__(
        self,
        provider: Provider,
        profile: ModelProfile,
        cache: DiskCache,
        *,
        rpm_bucket: Limiter | None = None,
        tpm_bucket: Limiter | None = None,
        max_retries: int = 3,
        fallback: Literal["fail", "ollama"] = "fail",
        fallback_provider: Provider | None = None,
        daily_cap: int | None = None,
        call_log: Callable[[dict[str, Any]], None] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if fallback == "ollama" and fallback_provider is None:
            raise ValueError("fallback='ollama' requires a fallback_provider")

        self.provider = provider
        self.profile = profile
        self.cache = cache
        self.rpm_bucket = rpm_bucket
        self.tpm_bucket = tpm_bucket
        self.max_retries = max_retries
        self.fallback = fallback
        self.fallback_provider = fallback_provider
        self.daily_cap = daily_cap if daily_cap is not None else profile.limits.rpd
        self.call_log = call_log
        self._sleep = sleep
        self._calls_today = 0

    # ----------------------------------------------------------- public ---

    async def generate(
        self,
        prompt: str,
        *,
        purpose: str,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        seed: int | None = None,
        think: bool | None = None,
        accept: Callable[[str], bool] | None = None,
        num_ctx: int | None = None,
    ) -> LLMResult:
        """`accept`: the caller's check on the reply text. Rejected replies are never
        cached, and a cached reply it rejects is re-requested, so one malformed answer
        cannot be replayed by every retry and rerun. `num_ctx`: context window for this
        call (Ollama); an execution setting like concurrency, so not part of the cache key."""
        params: dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "json_schema": json_schema,
        }
        # Only when set, so cache keys of calls that never chose stay valid.
        if think is not None:
            params["think"] = think
        key = cache_key(self.provider.name, self.profile.model, prompt, params, seed)

        cached = self.cache.get(key)
        if cached is not None and (accept is None or accept(cached["text"])):
            result = LLMResult(**cached)
            result.cache_hit = True
            self._log(purpose, key, result, latency_ms=0, error=None)
            return result

        if self._calls_today >= self.daily_cap:
            raise BudgetExhausted(
                f"daily cap reached: {self._calls_today} of {self.daily_cap} calls used; "
                "pausing so the run can resume rather than fail mid-experiment"
            )

        started = time.monotonic()
        try:
            sent = params if num_ctx is None else {**params, "num_ctx": num_ctx}
            result = await self._call_with_retry(prompt, sent, seed)
        except Exception as exc:
            self._log(
                purpose,
                key,
                None,
                latency_ms=int((time.monotonic() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        latency_ms = int((time.monotonic() - started) * 1000)
        # Only successful, accepted responses are cached; a failure must not poison reruns.
        if accept is None or accept(result.text):
            self.cache.set(key, result.model_dump())
        self._log(purpose, key, result, latency_ms=latency_ms, error=None)
        return result

    # ---------------------------------------------------------- internal ---

    async def _call_with_retry(
        self, prompt: str, params: dict[str, Any], seed: int | None
    ) -> LLMResult:
        last: Exception | None = None
        budget_stop = False

        for attempt in range(self.max_retries + 1):
            # Every retry is a real request against the quota, so the cap has to
            # be re-checked here and not only on the way in (NFR-3).
            if self._calls_today >= self.daily_cap:
                budget_stop = True
                break
            if self.rpm_bucket is not None:
                await self.rpm_bucket.acquire()
            if self.tpm_bucket is not None:
                # Input tokens (what Gemini's TPM counts). Clamped: a prompt over one
                # minute's budget waits for a full bucket; the provider judges its length.
                cost = min(estimate_tokens(prompt, 0.0), self.tpm_bucket.capacity)
                await self.tpm_bucket.acquire(cost)
            self._calls_today += 1
            try:
                result = await self.provider.generate(
                    prompt, model=self.profile.model, params=params, seed=seed
                )
                result.retries = attempt
                return result
            except RETRYABLE as exc:
                last = exc
                if attempt == self.max_retries:
                    break
                delay = backoff_delay(attempt)
                if isinstance(exc, RateLimited) and exc.retry_after is not None:
                    delay = max(delay, exc.retry_after)
                await self._sleep(delay)

        # The local fallback costs no provider quota, so it is still valid even
        # when the stop was the budget guard rather than an error.
        if self.fallback == "ollama" and self.fallback_provider is not None:
            return await self._call_fallback(prompt, params, seed)
        if budget_stop:
            raise BudgetExhausted(
                f"daily cap reached mid-retry: {self._calls_today} of "
                f"{self.daily_cap} calls used; pausing instead of overshooting"
            )
        assert last is not None
        raise last

    async def _call_fallback(
        self, prompt: str, params: dict[str, Any], seed: int | None
    ) -> LLMResult:
        assert self.fallback_provider is not None
        return await self.fallback_provider.generate(
            prompt, model=self.profile.model, params=params, seed=seed
        )

    def _log(
        self,
        purpose: str,
        prompt_hash: str,
        result: LLMResult | None,
        *,
        latency_ms: int,
        error: str | None,
    ) -> None:
        if self.call_log is None:
            return
        # prompt_hash, never the prompt: logs must not carry document text (NFR-7).
        self.call_log(
            {
                "ts": datetime.now(UTC).isoformat(),
                "provider": result.provider if result else self.provider.name,
                "model": result.model if result else self.profile.model,
                "purpose": purpose,
                "prompt_hash": prompt_hash,
                "tokens_in": result.tokens_in if result else 0,
                "tokens_out": result.tokens_out if result else 0,
                "latency_ms": latency_ms,
                "retries": result.retries if result else 0,
                "cache_hit": result.cache_hit if result else False,
                "error": error,
            }
        )
