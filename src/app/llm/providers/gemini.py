"""Gemini provider over the REST API.

DEVIATION from PRD §9, recorded as DECISION-6 in CLAUDE.md:
the PRD names the `google-genai` SDK but tags it **[VERIFY]**. This uses the
documented REST endpoint directly instead, because:
  - it is injectable with httpx.MockTransport, so the retry/error mapping that
    client.py depends on is unit-testable with no network (NFR-8);
  - it adds no dependency whose surface the PRD has not confirmed;
  - `Provider` isolates the choice, so swapping to the SDK later touches only
    this file.

Error mapping is the point of this module: 429 -> RateLimited, 5xx/timeout ->
TransientError (both retried), 4xx -> ProviderError (never retried).
"""

from __future__ import annotations

from typing import Any

import httpx

from app.llm.base import LLMResult, ProviderError, RateLimited, TransientError

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 429:
        raise RateLimited(f"gemini 429: {response.text[:200]}", _retry_after(response))
    if response.status_code >= 500:
        raise TransientError(f"gemini {response.status_code}: {response.text[:200]}")
    if response.status_code >= 400:
        raise ProviderError(f"gemini {response.status_code}: {response.text[:200]}")


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient | None = None,
        base_url: str = BASE_URL,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        prompt: str,
        *,
        model: str,
        params: dict[str, Any],
        seed: int | None,
    ) -> LLMResult:
        generation_config: dict[str, Any] = {
            "temperature": params.get("temperature", 0.2),
            "topP": params.get("top_p", 0.9),
        }
        if seed is not None:
            generation_config["seed"] = seed
        if params.get("json_schema"):
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = params["json_schema"]

        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }

        try:
            response = await self._client.post(url, json=body)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TransientError(f"gemini transport: {exc}") from exc

        _raise_for_status(response)
        payload = response.json()

        candidates = payload.get("candidates") or []
        if not candidates:
            # 200 with no candidates = safety block or empty generation.
            raise ProviderError(f"gemini returned no candidates: {str(payload)[:200]}")

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts)
        usage = payload.get("usageMetadata", {})

        return LLMResult(
            text=text,
            tokens_in=usage.get("promptTokenCount", 0),
            tokens_out=usage.get("candidatesTokenCount", 0),
            provider=self.name,
            model=model,
            cache_hit=False,
        )

    async def count_tokens(self, text: str, *, model: str) -> int:
        """Provider-side token count, for documents near tau (FR-2.1)."""
        url = f"{self.base_url}/models/{model}:countTokens?key={self.api_key}"
        try:
            response = await self._client.post(
                url, json={"contents": [{"parts": [{"text": text}]}]}
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TransientError(f"gemini transport: {exc}") from exc
        _raise_for_status(response)
        total: int = response.json()["totalTokens"]
        return total
