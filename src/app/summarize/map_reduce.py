"""Route B: map-reduce over chunks. PRD FR-3.3, FR-3.4, FR-3.5, §11.5.

Map: every chunk is summarized concurrently. `max_concurrency` caps requests in flight;
the client's RPM/TPM windows cap the rate (FR-3.3).

Reduce: nodes are merged in groups of `fan_in_k` until one remains. A group of one is
carried up unchanged (no call, no new node). Every node is kept with `level`,
`parent_id` and `source_chunk_ids`, so E3 can verify claims at each level (FR-3.4, G5).

Target length: a leaf asks for `compression_pct` of its chunk's words (floor
`MIN_TARGET_WORDS`); a merge asks for the sum of its children's targets, so the root
lands near the same overall compression as Route A.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar

from app.core.config import AppConfig
from app.core.schemas import Document, Route, SummaryNode
from app.llm.client import LLMClient
from app.route.chunker import Chunk, Count, chunk_sections
from app.route.router import tau_for
from app.route.tokens import estimate_tokens
from app.summarize.common import (
    SummaryError,
    SummaryResult,
    ask_summary,
    capped,
    fill,
    load_prompt,
)

CHUNK_PROMPT = "chunk_summary.v1.txt"
MERGE_PROMPT = "merge.v1.txt"
MIN_TARGET_WORDS = 10  # a heading-sized chunk would otherwise ask for 0-1 words

T = TypeVar("T")


async def _all(coros: list[Coroutine[Any, Any, T]]) -> list[T]:
    """Like gather (results in input order), but the first failure cancels the rest:
    gather left sibling calls running, spending quota after the run had failed."""
    try:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(c) for c in coros]
    except BaseExceptionGroup as eg:
        raise eg.exceptions[0] from None  # callers catch SummaryError / LLMError
    return [t.result() for t in tasks]


def _heading(doc: Document, chunk: Chunk) -> str:
    return " / ".join(s.heading for s in doc.sections if s.id in chunk.section_ids and s.heading)


async def summarize_b(
    doc: Document,
    client: LLMClient,
    cfg: AppConfig,
    prompt_dir: str | Path = "prompts",
    *,
    count: Count | None = None,
    chunker: Callable[..., list[Chunk]] = chunk_sections,
    max_concurrency: int | None = None,
    tokens: int | None = None,
    tau: int | None = None,
) -> SummaryResult:
    count = count or functools.partial(estimate_tokens, margin=cfg.route.token_margin)
    chunks = chunker(doc, cfg.chunk, count)
    if not chunks:
        raise SummaryError(f"{doc.doc_id}: no text to summarize")
    chunk_tpl, chunk_hash = load_prompt(prompt_dir, CHUNK_PROMPT)
    merge_tpl, merge_hash = load_prompt(prompt_dir, MERGE_PROMPT)
    pct = cfg.summary.compression_pct
    sem = asyncio.Semaphore(max_concurrency or cfg.active_profile.max_concurrency)
    models: list[str] = []

    async def limited(make: Callable[[], Awaitable[T]]) -> T:
        async with sem:
            return await make()

    async def leaf(c: Chunk) -> tuple[str, int]:
        text = doc.text_of(c.span)
        target = capped(max(MIN_TARGET_WORDS, round(len(text.split()) * pct / 100)), cfg)
        prompt = fill(chunk_tpl, heading=_heading(doc, c), target_words=str(target), chunk=text)
        summary, model = await ask_summary(
            client, prompt, purpose="chunk_summary", cfg=cfg, what=f"{doc.doc_id}/{c.id}"
        )
        models.append(model)
        return summary, target

    async def merge(group: list[tuple[SummaryNode, int]], node_id: str) -> tuple[str, int]:
        target = capped(sum(t for _, t in group), cfg)  # DECISION-37
        prompt = fill(
            merge_tpl,
            target_words=str(target),
            summaries="\n\n".join(n.text for n, _ in group),
        )
        summary, model = await ask_summary(
            client, prompt, purpose="merge", cfg=cfg, what=f"{doc.doc_id}/{node_id}"
        )
        models.append(model)
        return summary, target

    # results come back in input order: the tree follows document order, not completion order
    leaves = await _all([limited(functools.partial(leaf, c)) for c in chunks])
    nodes = [
        SummaryNode(id=f"n0-{i}", level=0, text=text, source_chunk_ids=[c.id])
        for i, (c, (text, _)) in enumerate(zip(chunks, leaves, strict=True))
    ]
    current = list(zip(nodes, [t for _, t in leaves], strict=True))

    level, k = 0, cfg.chunk.fan_in_k
    while len(current) > 1:
        level += 1
        groups = [current[i : i + k] for i in range(0, len(current), k)]
        ids = [f"n{level}-{j}" for j in range(len(groups))]
        merged = await _all(
            [
                limited(functools.partial(merge, g, nid))
                for g, nid in zip(groups, ids, strict=True)
                if len(g) > 1
            ]
        )
        it = iter(merged)
        nxt: list[tuple[SummaryNode, int]] = []
        for g, nid in zip(groups, ids, strict=True):
            if len(g) == 1:
                nxt.append(g[0])  # carried up: no call, no new node
                continue
            text, target = next(it)
            parent = SummaryNode(
                id=nid,
                level=level,
                text=text,
                source_chunk_ids=[cid for n, _ in g for cid in n.source_chunk_ids],
            )
            for n, _ in g:
                n.parent_id = parent.id
            nodes.append(parent)
            nxt.append((parent, target))
        current = nxt

    root = current[0][0]
    return SummaryResult(
        doc_id=doc.doc_id,
        summary=root.text,
        route=Route.B,
        tokens=(
            tokens if tokens is not None else estimate_tokens(doc.full_text, cfg.route.token_margin)
        ),
        tau=tau if tau is not None else tau_for(cfg.active_profile, cfg.route.tau_fraction),
        prompt_hashes={CHUNK_PROMPT: chunk_hash, MERGE_PROMPT: merge_hash},
        model=models[-1],
        nodes=nodes,
        chunks=chunks,
        root_id=root.id,
    )
