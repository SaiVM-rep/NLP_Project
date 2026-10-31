"""Ollama provider over the local REST API.

The GPU is 8 GB, not the 12 GB the PRD assumed (CLAUDE.md DEVIATION-1), so the
generator must be evictable before the NLI verifiers load (DECISION-3): `unload`.
`keep_alive` per request controls how long Ollama keeps the model loaded between calls;
0 would reload it before every call (DECISION-35).
"""

from __future__ import annotations

from typing import Any

import httpx

from app.llm.base import LLMResult, ProviderError, RateLimited, TransientError


def _json_schema(node: Any) -> Any:
    """Gemini-style schema (type "OBJECT"/"STRING") -> standard JSON Schema for Ollama."""
    if isinstance(node, dict):
        return {
            k: (v.lower() if k == "type" and isinstance(v, str) else _json_schema(v))
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_json_schema(v) for v in node]
    return node


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 429:
        raise RateLimited(f"ollama 429: {response.text[:200]}")
    if response.status_code >= 500:
        raise TransientError(f"ollama {response.status_code}: {response.text[:200]}")
    if response.status_code >= 400:
        # 404 here means the model was never pulled — retrying cannot fix it.
        raise ProviderError(f"ollama {response.status_code}: {response.text[:200]}")


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        client: httpx.AsyncClient | None = None,
        timeout: float = 600.0,
        keep_alive: float | str | None = None,
        num_ctx: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx

    async def generate(
        self,
        prompt: str,
        *,
        model: str,
        params: dict[str, Any],
        seed: int | None,
    ) -> LLMResult:
        options: dict[str, Any] = {
            "temperature": params.get("temperature", 0.2),
            "top_p": params.get("top_p", 0.9),
        }
        if seed is not None:
            options["seed"] = seed
        num_ctx = params.get("num_ctx") or self.num_ctx  # per call wins (DECISION-35)
        if num_ctx is not None:
            options["num_ctx"] = num_ctx

        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if params.get("json_schema"):
            # Constrain decoding to the schema itself. Plain "json" mode let qwen3 return
            # valid JSON of the wrong shape ({"<summary text>": "#summary"}).
            body["format"] = _json_schema(params["json_schema"])
        if self.keep_alive is not None:
            body["keep_alive"] = self.keep_alive
        if params.get("think") is not None:
            body["think"] = params["think"]

        try:
            response = await self._client.post(f"{self.base_url}/api/generate", json=body)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TransientError(f"ollama transport: {exc}") from exc

        _raise_for_status(response)
        payload = response.json()

        return LLMResult(
            text=payload.get("response", ""),
            tokens_in=payload.get("prompt_eval_count", 0),
            tokens_out=payload.get("eval_count", 0),
            provider=self.name,
            model=model,
            cache_hit=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def unload(self, model: str) -> None:
        """Evict the model from VRAM. Required before loading the verifiers."""
        try:
            await self._client.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0},
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TransientError(f"ollama transport: {exc}") from exc
