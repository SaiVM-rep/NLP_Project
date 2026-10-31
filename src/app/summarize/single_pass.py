"""Route A: whole document in one prompt. PRD FR-3.1, FR-2.4.

Documents over tau go to Route B (`map_reduce.summarize_b`).
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import AppConfig
from app.core.schemas import Document, Route
from app.llm.client import LLMClient
from app.route.router import route, tau_for
from app.route.tokens import estimate_tokens
from app.summarize.common import (
    COMPRESS_PROMPT,
    SummaryError,
    SummaryResult,
    ask_summary,
    capped,
    compress_if_long,
    fill,
    load_prompt,
)
from app.summarize.map_reduce import summarize_b

PROMPT = "single_pass.v1.txt"

__all__ = ["PROMPT", "SummaryError", "SummaryResult", "summarize"]


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
        return await summarize_b(doc, client, cfg, prompt_dir, tokens=tokens, tau=tau)

    template, digest = load_prompt(prompt_dir, PROMPT)
    pct = cfg.summary.compression_pct
    words = len(doc.full_text.split())
    target = capped(round(words * pct / 100), cfg)
    if target < round(words * pct / 100):  # capped: state the share actually asked for
        pct = round(100 * target / max(1, words), 1)
    prompt = fill(
        template,
        target_words=str(target),
        compression_pct=f"{pct:g}",
        document=doc.full_text,
    )
    summary, model = await ask_summary(
        client, prompt, purpose="single_pass", cfg=cfg, what=doc.doc_id
    )
    hashes = {PROMPT: digest}
    if comp := await compress_if_long(
        summary, client, cfg, prompt_dir=prompt_dir, what=doc.doc_id
    ):  # DECISION-38, capped uploads only
        summary, hashes[COMPRESS_PROMPT] = comp
    return SummaryResult(
        doc_id=doc.doc_id,
        summary=summary,
        route=Route.A,
        tokens=tokens,
        tau=tau,
        prompt_hashes=hashes,
        model=model,
    )
