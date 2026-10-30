"""Verify stage: summary nodes -> claims -> evidence -> both verifiers -> labels.

PRD §11.8 steps 1-6. Stages run phase by phase over *all* claims so only one model
is resident at a time on the 8 GB GPU (DECISION-3): the decomposer LLM is released,
then embedder, DeBERTa and MiniCheck each open, score everything, and close.

Models arrive as openers (context-manager factories) so ordering is unit-testable.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from app.core.config import VerifyCfg
from app.core.schemas import ClaimResult, Document, SummaryNode
from app.llm.client import LLMClient
from app.summarize.common import SummaryResult
from app.verify.decompose import Claim, decompose
from app.verify.evidence import Embed, retrieve_evidence
from app.verify.labels import label
from app.verify.minicheck import SupportFn, chunk_spans, support_score
from app.verify.nli import LabelIdx, score_claim


class Nli(Protocol):
    label_idx: LabelIdx

    def __call__(self, premises: list[str], hyps: list[str]) -> NDArray[np.float32]: ...


@dataclass(frozen=True)
class Openers:
    embedder: Callable[[], AbstractContextManager[Embed]]
    nli: Callable[[], AbstractContextManager[Nli]]
    minicheck: Callable[[], AbstractContextManager[SupportFn]]


def openers_from_config(
    cfg: VerifyCfg, *, device: str = "cuda", minicheck_local: Path | None = None
) -> Openers:
    """Real models, pinned by `configs/default.yaml` `verify.*`. Each loads only when opened.

    Local cache only (FR-5.6): HF loaders otherwise call huggingface.co even for a pinned
    sha. Missing weights fail here; fetch them first with `make models`. MiniCheck loads
    from the bf16 safetensors copy `make models` writes when it exists (DECISION-22),
    else from the pinned fp32 `.bin`.
    """
    from app.verify.embedder import SentenceEmbedder
    from app.verify.minicheck import MiniCheckT5
    from app.verify.models import bf16_dir, default_hf_home
    from app.verify.nli import DebertaNLI

    e, n, m = cfg.embedder, cfg.nli, cfg.minicheck
    local = minicheck_local or bf16_dir(m, default_hf_home())

    def open_minicheck() -> MiniCheckT5:
        ckpt, rev = (str(local), None) if local.is_dir() else (m.checkpoint, m.revision)
        return MiniCheckT5.from_pretrained(
            ckpt, revision=rev, device=device, batch_size=m.batch_size, local_files_only=True
        )

    return Openers(
        embedder=lambda: SentenceEmbedder.from_pretrained(
            e.checkpoint, revision=e.revision, device=device, local_files_only=True
        ),
        nli=lambda: DebertaNLI.from_pretrained(
            n.checkpoint,
            revision=n.revision,
            device=device,
            batch_size=n.batch_size,
            local_files_only=True,
        ),
        minicheck=open_minicheck,
    )


def verify_claims(
    doc: Document, claims: list[tuple[str, Claim]], openers: Openers, cfg: VerifyCfg
) -> list[ClaimResult]:
    """`claims` are (summary_node_id, claim) pairs; one ClaimResult each, same order."""
    if not claims:
        return []
    texts = [c.text for _, c in claims]

    with openers.embedder() as embed:
        evidence = retrieve_evidence(texts, doc, embed, cfg.top_n_evidence)

    with openers.nli() as nli:
        nli_scores = [
            score_claim(t, [doc.text_of(s) for s in ev], nli, nli.label_idx)
            for t, ev in zip(texts, evidence, strict=True)
        ]

    chunks = [doc.text_of(s) for s in chunk_spans(doc, cfg.minicheck.chunk_words)]
    with openers.minicheck() as mc:
        support = [support_score(t, chunks, mc) for t in texts]

    n, m = cfg.nli, cfg.minicheck
    out = []
    for (node_id, c), ev, (entail, contra), sup in zip(
        claims, evidence, nli_scores, support, strict=True
    ):
        out.append(
            ClaimResult(
                claim=c.text,
                summary_sentence_idx=c.sentence_idx,
                summary_node_id=node_id,
                evidence=ev,
                deberta_entail=entail,
                deberta_contra=contra,
                minicheck_support=sup,
                label_deberta=label(entail, contra, theta_e=n.theta_e, theta_c=n.theta_c),
                # theta_c unused: MiniCheck has no contradiction score
                label_minicheck=label(sup, None, theta_e=m.theta_e, theta_c=1.0),
            )
        )
    return out


async def verify_nodes(
    doc: Document,
    nodes: list[SummaryNode],
    client: LLMClient,
    openers: Openers,
    cfg: VerifyCfg,
    *,
    release_llm: Callable[[], Awaitable[None]] | None = None,
    prompt_dir: str | Path = "prompts",
) -> list[ClaimResult]:
    """Decompose every node first, free the LLM's VRAM, then verify all claims."""
    claims: list[tuple[str, Claim]] = []
    for node in nodes:
        claims.extend((node.id, c) for c in await decompose(node.text, client, prompt_dir))
    if release_llm is not None:
        await release_llm()
    return verify_claims(doc, claims, openers, cfg)


ROUTE_A_NODE_ID = "root"


def summary_nodes(result: SummaryResult) -> list[SummaryNode]:
    """What to verify: every merge-tree node for Route B (G5, E3); Route A is one level-0 node."""
    if result.nodes:
        return list(result.nodes)
    return [SummaryNode(id=ROUTE_A_NODE_ID, level=0, text=result.summary)]


@dataclass(frozen=True)
class VerifiedSummary:
    """Claims point at `nodes` by id, so `eval.metrics.faithfulness_by_level` applies directly."""

    nodes: list[SummaryNode]
    claims: list[ClaimResult]


async def verify_summary(
    doc: Document,
    result: SummaryResult,
    client: LLMClient,
    openers: Openers,
    cfg: VerifyCfg,
    *,
    release_llm: Callable[[], Awaitable[None]] | None = None,
    prompt_dir: str | Path = "prompts",
) -> VerifiedSummary:
    nodes = summary_nodes(result)
    claims = await verify_nodes(
        doc, nodes, client, openers, cfg, release_llm=release_llm, prompt_dir=prompt_dir
    )
    return VerifiedSummary(nodes=nodes, claims=claims)
