"""Route A: whole document in one prompt. PRD FR-3.1, FR-2.4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

from app.core.config import AppConfig
from app.core.schemas import Document, Route
from app.llm.client import LLMClient
from app.route.router import route, tau_for
from app.route.tokens import estimate_tokens

PROMPT = "single_pass.v1.txt"
SCHEMA = {"type": "OBJECT", "properties": {"summary": {"type": "STRING"}}, "required": ["summary"]}
REPAIR = '\n\nYour previous reply was not valid JSON. Reply with only: {"summary": "<text>"}'


class SummaryError(RuntimeError):
    pass


class SummaryResult(BaseModel):
    doc_id: str
    summary: str
    route: Route
    tokens: int
    tau: int
    prompt: str
    prompt_hash: str
    model: str


def _parse(text: str) -> str | None:
    body = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        out = json.loads(body)
    except json.JSONDecodeError:
        return None
    summary = out.get("summary") if isinstance(out, dict) else None
    return summary.strip() if isinstance(summary, str) and summary.strip() else None


async def summarize(
    doc: Document,
    client: LLMClient,
    cfg: AppConfig,
    prompt_dir: str | Path = "prompts",
    tau_override: int | None = None,
) -> SummaryResult:
    tokens = estimate_tokens(doc.full_text, cfg.route.token_margin)
    tau = tau_for(cfg.active_profile, cfg.route.tau_fraction, tau_override)
    if route(tokens, tau) is Route.B:
        raise SummaryError(
            f"{tokens} tokens > tau {tau}: needs Route B (map-reduce), not built until P4"
        )

    raw = (Path(prompt_dir) / PROMPT).read_bytes()
    template = raw.decode("utf-8")
    pct = cfg.summary.compression_pct
    # str.replace, not .format: the template contains literal JSON braces.
    prompt = (
        template.replace("{target_words}", str(round(len(doc.full_text.split()) * pct / 100)))
        .replace("{compression_pct}", f"{pct:g}")
        .replace("{document}", doc.full_text)
    )
    dec = cfg.decoding
    summary = None
    for attempt_prompt in (prompt, prompt + REPAIR):
        res = await client.generate(
            attempt_prompt,
            purpose="single_pass",
            json_schema=SCHEMA,
            temperature=dec.temperature,
            top_p=dec.top_p,
            seed=dec.seed,
        )
        if (summary := _parse(res.text)) is not None:
            break
    if summary is None:
        raise SummaryError(f"model returned invalid JSON twice for {doc.doc_id}")

    return SummaryResult(
        doc_id=doc.doc_id,
        summary=summary,
        route=Route.A,
        tokens=tokens,
        tau=tau,
        prompt=PROMPT,
        prompt_hash=hashlib.sha256(raw).hexdigest(),  # same hash as manifest.prompt_hashes
        model=res.model,
    )
