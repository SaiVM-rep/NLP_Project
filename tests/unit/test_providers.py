"""Gemini and Ollama providers. PRD §11.3, §9.

Both are driven over httpx, so both are testable with a MockTransport and no
network. What matters here is the mapping from HTTP reality to our error types:
the retry policy in client.py is only correct if 429/5xx/4xx land in the right
buckets.
"""

import httpx
import pytest

from app.llm.base import ProviderError, RateLimited, TransientError
from app.llm.providers.gemini import GeminiProvider
from app.llm.providers.ollama import OllamaProvider

PARAMS = {"temperature": 0.2, "top_p": 0.9, "json_schema": None}


def _transport(handler):
    return httpx.MockTransport(handler)


def _gemini(handler, **kw):
    return GeminiProvider(
        api_key="test-key", client=httpx.AsyncClient(transport=_transport(handler)), **kw
    )


def _ollama(handler, **kw):
    return OllamaProvider(
        base_url="http://localhost:11434",
        client=httpx.AsyncClient(transport=_transport(handler)),
        **kw,
    )


GEMINI_OK = {
    "candidates": [{"content": {"parts": [{"text": "a summary"}]}}],
    "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 30},
}

OLLAMA_OK = {
    "response": "a summary",
    "prompt_eval_count": 120,
    "eval_count": 30,
}


# --------------------------------------------------------------- gemini ---


async def test_gemini_returns_text_and_token_counts():
    p = _gemini(lambda r: httpx.Response(200, json=GEMINI_OK))
    r = await p.generate("doc", model="gemini-2.5-flash", params=PARAMS, seed=1)

    assert r.text == "a summary"
    assert (r.tokens_in, r.tokens_out) == (120, 30)
    assert r.provider == "gemini"
    assert r.model == "gemini-2.5-flash"


async def test_gemini_sends_the_api_key_and_model_in_the_request():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json=GEMINI_OK)

    await _gemini(handler).generate("doc", model="gemini-2.5-flash", params=PARAMS, seed=None)
    assert "gemini-2.5-flash:generateContent" in seen["url"]
    assert "key=test-key" in seen["url"]


async def test_gemini_passes_decoding_params():
    seen = {}

    def handler(request):
        import json

        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=GEMINI_OK)

    await _gemini(handler).generate(
        "doc", model="m", params={"temperature": 0.7, "top_p": 0.5, "json_schema": None}, seed=42
    )
    cfg = seen["body"]["generationConfig"]
    assert cfg["temperature"] == 0.7
    assert cfg["topP"] == 0.5


async def test_gemini_requests_json_when_a_schema_is_given():
    """FR-3.1: structured output where the provider supports a response schema."""
    seen = {}

    def handler(request):
        import json

        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=GEMINI_OK)

    schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
    await _gemini(handler).generate(
        "doc",
        model="m",
        params={"temperature": 0.2, "top_p": 0.9, "json_schema": schema},
        seed=None,
    )
    cfg = seen["body"]["generationConfig"]
    assert cfg["responseMimeType"] == "application/json"
    assert cfg["responseSchema"] == schema


async def test_gemini_429_raises_rate_limited():
    p = _gemini(lambda r: httpx.Response(429, json={"error": {"message": "quota"}}))
    with pytest.raises(RateLimited):
        await p.generate("doc", model="m", params=PARAMS, seed=None)


async def test_gemini_429_honours_retry_after():
    p = _gemini(lambda r: httpx.Response(429, headers={"Retry-After": "7"}, json={}))
    with pytest.raises(RateLimited) as exc:
        await p.generate("doc", model="m", params=PARAMS, seed=None)
    assert exc.value.retry_after == 7.0


async def test_gemini_5xx_raises_transient():
    p = _gemini(lambda r: httpx.Response(503, json={}))
    with pytest.raises(TransientError):
        await p.generate("doc", model="m", params=PARAMS, seed=None)


async def test_gemini_4xx_raises_permanent_provider_error():
    """A bad key or bad model must NOT be retried."""
    p = _gemini(lambda r: httpx.Response(400, json={"error": {"message": "bad model"}}))
    with pytest.raises(ProviderError):
        await p.generate("doc", model="m", params=PARAMS, seed=None)


async def test_gemini_timeout_is_transient():
    def handler(request):
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(TransientError):
        await _gemini(handler).generate("doc", model="m", params=PARAMS, seed=None)


async def test_gemini_empty_candidates_is_a_provider_error():
    """A safety block returns 200 with no candidates; that is not a summary."""
    p = _gemini(lambda r: httpx.Response(200, json={"candidates": []}))
    with pytest.raises(ProviderError, match="no candidates"):
        await p.generate("doc", model="m", params=PARAMS, seed=None)


async def test_gemini_counts_tokens():
    """FR-2.1: provider token count for documents near tau."""
    p = _gemini(lambda r: httpx.Response(200, json={"totalTokens": 4321}))
    assert await p.count_tokens("doc", model="gemini-2.5-flash") == 4321


# --------------------------------------------------------------- ollama ---


async def test_ollama_returns_text_and_token_counts():
    p = _ollama(lambda r: httpx.Response(200, json=OLLAMA_OK))
    r = await p.generate("doc", model="qwen3:8b", params=PARAMS, seed=1)

    assert r.text == "a summary"
    assert (r.tokens_in, r.tokens_out) == (120, 30)
    assert r.provider == "ollama"
    assert r.model == "qwen3:8b"


async def test_ollama_disables_streaming():
    seen = {}

    def handler(request):
        import json

        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OLLAMA_OK)

    await _ollama(handler).generate("doc", model="qwen3:8b", params=PARAMS, seed=None)
    assert seen["body"]["stream"] is False


async def test_ollama_passes_seed_and_decoding_options():
    """NFR-6: determinism needs the seed to actually reach the runtime."""
    seen = {}

    def handler(request):
        import json

        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OLLAMA_OK)

    await _ollama(handler).generate(
        "doc",
        model="qwen3:8b",
        params={"temperature": 0.7, "top_p": 0.5, "json_schema": None},
        seed=1234,
    )
    opts = seen["body"]["options"]
    assert opts["seed"] == 1234
    assert opts["temperature"] == 0.7
    assert opts["top_p"] == 0.5


async def test_ollama_requests_json_format_when_schema_given():
    seen = {}

    def handler(request):
        import json

        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OLLAMA_OK)

    await _ollama(handler).generate(
        "doc",
        model="m",
        params={"temperature": 0.2, "top_p": 0.9, "json_schema": {"type": "object"}},
        seed=None,
    )
    assert seen["body"]["format"] == "json"


async def test_ollama_unloads_the_model_when_asked():
    """CLAUDE.md DECISION-3: 8 GB GPU cannot hold the LLM and the verifiers."""
    seen = {}

    def handler(request):
        import json

        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OLLAMA_OK)

    await _ollama(handler, keep_alive=0).generate("doc", model="qwen3:8b", params=PARAMS, seed=None)
    assert seen["body"]["keep_alive"] == 0


async def test_ollama_connection_refused_is_transient():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(TransientError):
        await _ollama(handler).generate("doc", model="m", params=PARAMS, seed=None)


async def test_ollama_404_model_not_pulled_is_permanent():
    p = _ollama(lambda r: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(ProviderError, match="model"):
        await p.generate("doc", model="nope:1b", params=PARAMS, seed=None)


async def test_ollama_5xx_is_transient():
    p = _ollama(lambda r: httpx.Response(500, json={"error": "boom"}))
    with pytest.raises(TransientError):
        await p.generate("doc", model="m", params=PARAMS, seed=None)


# ----------------------------------------------------- protocol conformance ---


async def test_both_providers_satisfy_the_same_protocol():
    """PRD §11.3: prompts never branch on provider."""
    from app.llm.base import Provider

    assert isinstance(_gemini(lambda r: httpx.Response(200, json=GEMINI_OK)), Provider)
    assert isinstance(_ollama(lambda r: httpx.Response(200, json=OLLAMA_OK)), Provider)


async def test_providers_report_distinct_names():
    g = _gemini(lambda r: httpx.Response(200, json=GEMINI_OK))
    o = _ollama(lambda r: httpx.Response(200, json=OLLAMA_OK))
    assert {g.name, o.name} == {"gemini", "ollama"}


async def test_ollama_sends_think_flag_only_when_set():
    """Qwen3 thinks by default: 533 tokens / 72 s vs 21 / 10 s on a claim (2026-10-29)."""
    import json

    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=OLLAMA_OK)

    await _ollama(handler).generate("doc", model="m", params={**PARAMS, "think": False}, seed=None)
    await _ollama(handler).generate("doc", model="m", params=PARAMS, seed=None)
    assert bodies[0]["think"] is False
    assert "think" not in bodies[1]  # server default untouched
