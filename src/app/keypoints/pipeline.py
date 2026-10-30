"""PRD §11.7: candidates -> salience -> near-duplicate clusters -> top K -> attribution.

The LLM is released before the embedder loads, and the embedder is loaded once for both
candidates and source sentences (DECISION-3, one model resident on the 8 GB GPU).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
from pathlib import Path

from app.core.config import AppConfig
from app.core.schemas import Document, KeyPoint
from app.keypoints.attribute import attribute
from app.keypoints.candidates import generate_candidates
from app.keypoints.cluster import cluster, select
from app.keypoints.rank import salience
from app.llm.client import LLMClient
from app.verify.evidence import Embed


async def extract_keypoints(
    doc: Document,
    client: LLMClient,
    open_embedder: Callable[[], AbstractContextManager[Embed]],
    cfg: AppConfig,
    *,
    release_llm: Callable[[], Awaitable[None]] | None = None,
    prompt_dir: str | Path = "prompts",
) -> list[KeyPoint]:
    cands = await generate_candidates(doc, client, cfg, prompt_dir)
    if release_llm is not None:
        await release_llm()
    if not cands:
        return []
    kp = cfg.keypoints
    with open_embedder() as embed:
        vecs = embed([c.text for c in cands])
        sent_vecs = embed([doc.text_of(s) for s in doc.sentences])

    sal = salience(vecs, sent_vecs, kp.lambda_graph)
    top = select(
        cluster(vecs, sal, dup_distance=kp.dup_distance), [c.position for c in cands], kp.top_k
    )
    reps = [c.rep for c in top]
    attributed = attribute(vecs[reps], sent_vecs, doc.sentences, theta_attr=kp.theta_attr)
    return [
        KeyPoint(
            text=cands[c.rep].text,
            rank=rank,
            cluster_size=len(c.members),
            evidence=span,
            similarity=sim,
        )
        for rank, (c, (span, sim)) in enumerate(zip(top, attributed, strict=True), start=1)
    ]
