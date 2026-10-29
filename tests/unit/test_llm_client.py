"""LLM client: cache, limiter, retry, fallback, budget guard, call log.

PRD §11.3, FR-7.4, NFR-1, NFR-2, NFR-3, NFR-6.

No network. Providers are fakes that record what they were asked for.
"""

import pytest

from app.core.config import Limits, ModelProfile
from app.llm.base import BudgetExhausted, LLMResult, ProviderError, RateLimited
from app.llm.cache import DiskCache
from app.llm.client import LLMClient
from app.llm.limiter import TokenBucket


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class FakeProvider:
    """Records calls; can be scripted to fail a given number of times first."""

    def __init__(self, name="gemini", text="a summary", fail_times=0, exc=None):
        self.name = name
        self.text = text
        self.fail_times = fail_times
        self.exc = exc or RateLimited("429")
        self.calls: list[dict] = []

    async def generate(self, prompt, *, model, params, seed):
        self.calls.append({"prompt": prompt, "model": model, "params": params, "seed": seed})
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.exc
        return LLMResult(
            text=self.text,
            tokens_in=10,
            tokens_out=5,
            provider=self.name,
            model=model,
            cache_hit=False,
        )


PROFILE = ModelProfile(
    provider="gemini",
    model="gemini-2.5-flash",
    effective_context=1_000_000,
    limits=Limits(rpm=10, tpm=250_000, rpd=250),
)


def _client(tmp_path, provider, **kw):
    clock = FakeClock()
    kw.setdefault("cache", DiskCache(tmp_path))
    kw.setdefault("rpm_bucket", TokenBucket(10 / 60, 10, clock=clock, sleep=clock.sleep))
    kw.setdefault("sleep", clock.sleep)
    client = LLMClient(provider=provider, profile=PROFILE, **kw)
    client._test_clock = clock  # test introspection only
    return client


# ---------------------------------------------------------------- cache ---


async def test_first_call_hits_the_provider(tmp_path):
    p = FakeProvider()
    r = await _client(tmp_path, p).generate("doc", purpose="summarize")
    assert r.text == "a summary"
    assert r.cache_hit is False
    assert len(p.calls) == 1


async def test_identical_second_call_is_served_from_cache(tmp_path):
    """NFR-1/NFR-2: a rerun must cost zero quota."""
    p = FakeProvider()
    c = _client(tmp_path, p)
    await c.generate("doc", purpose="summarize")
    r = await c.generate("doc", purpose="summarize")

    assert r.cache_hit is True
    assert r.text == "a summary"
    assert len(p.calls) == 1, "provider was called twice for an identical request"


async def test_cache_persists_across_client_instances(tmp_path):
    p1 = FakeProvider()
    await _client(tmp_path, p1).generate("doc", purpose="summarize")

    p2 = FakeProvider(text="DIFFERENT")
    r = await _client(tmp_path, p2).generate("doc", purpose="summarize")

    assert r.text == "a summary", "second client ignored the cache on disk"
    assert p2.calls == []


async def test_different_seed_is_a_different_cache_entry(tmp_path):
    p = FakeProvider()
    c = _client(tmp_path, p)
    await c.generate("doc", purpose="summarize", seed=1)
    await c.generate("doc", purpose="summarize", seed=2)
    assert len(p.calls) == 2


async def test_disabled_cache_always_calls_the_provider(tmp_path):
    p = FakeProvider()
    c = _client(tmp_path, p, cache=DiskCache(tmp_path, enabled=False))
    await c.generate("doc", purpose="summarize")
    await c.generate("doc", purpose="summarize")
    assert len(p.calls) == 2


# -------------------------------------------------------------- limiter ---


async def test_rate_limiter_is_acquired_before_a_real_call(tmp_path):
    """FR-3.3 / NFR-3: concurrency must never exceed the configured RPM."""
    clock = FakeClock()
    bucket = TokenBucket(1.0, 1.0, clock=clock, sleep=clock.sleep)
    c = _client(tmp_path, FakeProvider(), rpm_bucket=bucket, sleep=clock.sleep)

    await c.generate("doc1", purpose="p")
    await c.generate("doc2", purpose="p")

    assert clock.slept, "second call did not wait on the rate limiter"


async def test_cache_hit_does_not_consume_rate_limit(tmp_path):
    """A cached answer uses no quota, so it must not drain the bucket."""
    clock = FakeClock()
    bucket = TokenBucket(1.0, 1.0, clock=clock, sleep=clock.sleep)
    c = _client(tmp_path, FakeProvider(), rpm_bucket=bucket, sleep=clock.sleep)

    await c.generate("doc", purpose="p")  # drains the single token
    await c.generate("doc", purpose="p")  # cache hit

    assert clock.slept == [], "a cache hit waited on the rate limiter"


# ---------------------------------------------------------------- retry ---


async def test_retries_on_429_then_succeeds(tmp_path):
    p = FakeProvider(fail_times=2)
    c = _client(tmp_path, p, max_retries=3)

    r = await c.generate("doc", purpose="p")

    assert r.text == "a summary"
    assert len(p.calls) == 3


async def test_retry_count_is_reported_on_the_result(tmp_path):
    p = FakeProvider(fail_times=2)
    r = await _client(tmp_path, p, max_retries=3).generate("doc", purpose="p")
    assert r.retries == 2


async def test_backoff_sleeps_between_retries(tmp_path):
    p = FakeProvider(fail_times=2)
    c = _client(tmp_path, p, max_retries=3)
    await c.generate("doc", purpose="p")
    assert len(c._test_clock.slept) >= 2


async def test_gives_up_after_max_retries(tmp_path):
    p = FakeProvider(fail_times=99)
    with pytest.raises(RateLimited):
        await _client(tmp_path, p, max_retries=2).generate("doc", purpose="p")
    assert len(p.calls) == 3  # initial + 2 retries


async def test_failed_call_is_not_cached(tmp_path):
    p = FakeProvider(fail_times=99)
    c = _client(tmp_path, p, max_retries=0)
    with pytest.raises(RateLimited):
        await c.generate("doc", purpose="p")

    p2 = FakeProvider()
    r = await _client(tmp_path, p2, max_retries=0).generate("doc", purpose="p")
    assert r.cache_hit is False and len(p2.calls) == 1


async def test_non_transient_error_is_not_retried(tmp_path):
    p = FakeProvider(fail_times=99, exc=ProviderError("bad request"))
    with pytest.raises(ProviderError):
        await _client(tmp_path, p, max_retries=5).generate("doc", purpose="p")
    assert len(p.calls) == 1, "a non-transient error was retried"


# ------------------------------------------------------------- fallback ---


async def test_fallback_fail_is_the_default(tmp_path):
    """PRD §11.3: experiments must never silently mix models."""
    p = FakeProvider(fail_times=99)
    with pytest.raises(RateLimited):
        await _client(tmp_path, p, max_retries=0).generate("doc", purpose="p")


async def test_fallback_ollama_switches_provider_after_exhaustion(tmp_path):
    primary = FakeProvider(name="gemini", fail_times=99)
    secondary = FakeProvider(name="ollama", text="local summary")
    c = _client(tmp_path, primary, max_retries=1, fallback="ollama", fallback_provider=secondary)

    r = await c.generate("doc", purpose="p")

    assert r.provider == "ollama"
    assert r.text == "local summary"


async def test_fallback_ollama_without_a_secondary_provider_raises(tmp_path):
    p = FakeProvider(fail_times=99)
    with pytest.raises(ValueError, match="fallback_provider"):
        _client(tmp_path, p, fallback="ollama")


# --------------------------------------------------------- budget guard ---


async def test_daily_budget_guard_stops_the_run(tmp_path):
    """NFR-3: stop cleanly at the cap, never fail mid-experiment."""
    c = _client(tmp_path, FakeProvider(), daily_cap=2)
    await c.generate("d1", purpose="p")
    await c.generate("d2", purpose="p")

    with pytest.raises(BudgetExhausted):
        await c.generate("d3", purpose="p")


async def test_cache_hits_do_not_count_against_the_daily_budget(tmp_path):
    c = _client(tmp_path, FakeProvider(), daily_cap=1)
    await c.generate("d1", purpose="p")
    r = await c.generate("d1", purpose="p")  # cached, must not raise
    assert r.cache_hit is True


async def test_retries_cannot_overshoot_the_daily_cap(tmp_path):
    """NFR-3 says *never* exceed the provider limit.

    Each retry is a real request against the quota, so a single generate() with
    retries left must still stop at the cap instead of burning through it.
    """
    p = FakeProvider(fail_times=99)
    c = _client(tmp_path, p, max_retries=5, daily_cap=2)

    with pytest.raises((RateLimited, BudgetExhausted)):
        await c.generate("doc", purpose="p")

    assert len(p.calls) <= 2, f"made {len(p.calls)} requests against a cap of 2"


async def test_budget_exhausted_reports_the_counts(tmp_path):
    c = _client(tmp_path, FakeProvider(), daily_cap=1)
    await c.generate("d1", purpose="p")
    with pytest.raises(BudgetExhausted) as exc:
        await c.generate("d2", purpose="p")
    assert "1" in str(exc.value)


# -------------------------------------------------------------- logging ---


async def test_every_call_is_logged(tmp_path):
    """FR-7.4: timestamp, provider, model, tokens, latency, retries, cache hit."""
    log: list[dict] = []
    c = _client(tmp_path, FakeProvider(), call_log=log.append)

    await c.generate("doc", purpose="summarize")

    assert len(log) == 1
    entry = log[0]
    for field in (
        "ts",
        "provider",
        "model",
        "purpose",
        "prompt_hash",
        "tokens_in",
        "tokens_out",
        "latency_ms",
        "retries",
        "cache_hit",
    ):
        assert field in entry, f"missing {field} in llm_calls entry"
    assert entry["purpose"] == "summarize"
    assert entry["cache_hit"] is False


async def test_cache_hits_are_logged_too(tmp_path):
    log: list[dict] = []
    c = _client(tmp_path, FakeProvider(), call_log=log.append)
    await c.generate("doc", purpose="p")
    await c.generate("doc", purpose="p")

    assert len(log) == 2
    assert [e["cache_hit"] for e in log] == [False, True]


async def test_failed_call_is_logged_with_its_error(tmp_path):
    log: list[dict] = []
    p = FakeProvider(fail_times=99)
    c = _client(tmp_path, p, max_retries=0, call_log=log.append)
    with pytest.raises(RateLimited):
        await c.generate("doc", purpose="p")

    assert len(log) == 1
    assert log[0]["error"]


async def test_prompt_hash_is_stable_and_not_the_raw_prompt(tmp_path):
    """Logs must not leak uploaded document text (NFR-7)."""
    log: list[dict] = []
    c = _client(tmp_path, FakeProvider(), call_log=log.append)
    await c.generate("secret document text", purpose="p")
    assert "secret" not in log[0]["prompt_hash"]
    assert len(log[0]["prompt_hash"]) == 64


# --------------------------------------------------------- param wiring ---


async def test_decoding_params_reach_the_provider(tmp_path):
    p = FakeProvider()
    c = _client(tmp_path, p)
    await c.generate("doc", purpose="p", temperature=0.7, top_p=0.5, seed=99)
    call = p.calls[0]
    assert call["params"]["temperature"] == 0.7
    assert call["params"]["top_p"] == 0.5
    assert call["seed"] == 99


async def test_model_comes_from_the_profile(tmp_path):
    p = FakeProvider()
    await _client(tmp_path, p).generate("doc", purpose="p")
    assert p.calls[0]["model"] == "gemini-2.5-flash"


async def test_think_flag_reaches_provider_and_splits_the_cache(tmp_path):
    p = FakeProvider()
    c = _client(tmp_path, p)
    await c.generate("doc", purpose="p")
    await c.generate("doc", purpose="p", think=False)
    assert len(p.calls) == 2  # different params -> different cache key, no false hit
    assert "think" not in p.calls[0]["params"]  # unset keeps pre-existing cache keys valid
    assert p.calls[1]["params"]["think"] is False
