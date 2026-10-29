"""Provider-agnostic LLM types. PRD §11.3.

Prompts never branch on provider — everything above this layer sees only
`Provider` and `LLMResult`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class LLMError(Exception):
    """Base for everything this layer raises."""


class ProviderError(LLMError):
    """Permanent failure (bad request, auth, unsupported model). Not retried."""


class RateLimited(LLMError):
    """HTTP 429. Retried with backoff, honouring Retry-After when given."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TransientError(LLMError):
    """5xx / timeout / connection reset. Retried with backoff."""


class BudgetExhausted(LLMError):
    """Daily cap reached. The runner pauses and checkpoints (NFR-3)."""


RETRYABLE = (RateLimited, TransientError)


class LLMResult(BaseModel):
    text: str
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    provider: str
    model: str
    cache_hit: bool = False
    retries: int = Field(default=0, ge=0)


@runtime_checkable
class Provider(Protocol):
    """What the client needs from a backend. Implemented in llm/providers/."""

    name: str

    async def generate(
        self,
        prompt: str,
        *,
        model: str,
        params: dict[str, Any],
        seed: int | None,
    ) -> LLMResult: ...
