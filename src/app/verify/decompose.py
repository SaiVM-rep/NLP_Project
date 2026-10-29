"""FR-5.1: split a summary into sentences, then each sentence into atomic claims.

PRD §11.8 steps 1-2. Runs on the local model by default (`verify.decomposer`) so
claim decomposition never spends API quota.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from app.ingest.spans import sentence_spans
from app.llm.client import LLMClient

PROMPT = "claim_decompose.v2.txt"  # DECISION-15; v1 kept frozen
SCHEMA = {
    "type": "OBJECT",
    "properties": {"claims": {"type": "ARRAY", "items": {"type": "STRING"}}},
    "required": ["claims"],
}
REPAIR = '\n\nYour previous reply was not valid JSON. Reply with only: {"claims": ["...", "..."]}'


class DecomposeError(RuntimeError):
    pass


class Claim(BaseModel):
    text: str
    sentence_idx: int


def _parse(text: str) -> list[str] | None:
    body = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        out = json.loads(body)
    except json.JSONDecodeError:
        return None
    claims = out.get("claims") if isinstance(out, dict) else None
    if not isinstance(claims, list) or not all(isinstance(c, str) for c in claims):
        return None
    return [c.strip() for c in claims if c.strip()]


async def decompose(
    summary: str, client: LLMClient, prompt_dir: str | Path = "prompts"
) -> list[Claim]:
    template = (Path(prompt_dir) / PROMPT).read_text(encoding="utf-8")
    sentences = [summary[s.start : s.end] for s in sentence_spans(summary, 0)]
    out: list[Claim] = []
    for i, sent in enumerate(sentences):
        prompt = template.replace("{sentence}", sent).replace(
            "{context}", sentences[i - 1] if i else "(none)"
        )
        claims = None
        for attempt in (prompt, prompt + REPAIR):
            # Greedy and no hidden reasoning: decomposition is measurement, not generation.
            # Qwen3 thinking cost 533 tokens / 72 s vs 21 / 10 s per sentence (2026-10-29).
            res = await client.generate(
                attempt,
                purpose="claim_decompose",
                json_schema=SCHEMA,
                temperature=0.0,
                think=False,
            )
            if (claims := _parse(res.text)) is not None:
                break
        if claims is None:
            raise DecomposeError(f"invalid JSON twice decomposing sentence {i}: {sent[:80]!r}")
        out.extend(Claim(text=c, sentence_idx=i) for c in claims)
    return out
