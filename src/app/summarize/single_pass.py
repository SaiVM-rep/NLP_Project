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
from app.summarize.common import SummaryError, SummaryResult, ask_summary, fill, load_prompt
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
    prompt = fill(
        template,
        target_words=str(round(len(doc.full_text.split()) * pct / 100)),
        compression_pct=f"{pct:g}",
        document=doc.full_text,
    )
    summary, model = await ask_summary(
        client, prompt, purpose="single_pass", cfg=cfg, what=doc.doc_id
    )
    return SummaryResult(
        doc_id=doc.doc_id,
        summary=summary,
        route=Route.A,
        tokens=tokens,
        tau=tau,
        prompt_hashes={PROMPT: digest},
        model=model,
    )
