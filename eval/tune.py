"""Threshold tuning on AggreFact dev (CLAUDE.md M7, SC4).

AggreFact labels are binary (supported or not), so they tune `theta_e` only. `theta_c`
splits "unsupported" from "contradicted", which needs 3-way labels (human annotation).
Scoring is done once on the GPU and stored; the sweep below runs on the stored scores.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from typing import NamedTuple

from app.core.config import VerifyCfg
from app.core.schemas import Document, Span
from app.verify.evidence import retrieve_evidence
from app.verify.minicheck import chunk_spans, support_score
from app.verify.nli import score_claim
from app.verify.pipeline import Openers
from eval.data import AggreFactItem, LongDoc, to_document


class ThetaFit(NamedTuple):
    theta: float
    balanced_accuracy: float


def balanced_accuracy(truth: Sequence[int], pred: Sequence[bool]) -> float:
    """Mean of per-class recall (LLM-AggreFact's metric). NaN if a class is absent."""
    pos = [p for t, p in zip(truth, pred, strict=True) if t == 1]
    neg = [p for t, p in zip(truth, pred, strict=True) if t == 0]
    if not pos or not neg:
        return math.nan
    return (sum(pos) / len(pos) + sum(not p for p in neg) / len(neg)) / 2


def best_theta(
    scores: Sequence[float], truth: Sequence[int], grid: Sequence[float] | None = None
) -> ThetaFit:
    """Theta maximising balanced accuracy; `score >= theta` means supported.

    Several thetas often tie (any value between two adjacent scores): take the middle one,
    an edge would sit right on a score and move with a few rows.
    """
    grid = grid if grid is not None else [i / 100 for i in range(1, 100)]
    fits = [ThetaFit(g, balanced_accuracy(truth, [s >= g for s in scores])) for g in grid]
    top = max(f.balanced_accuracy for f in fits)
    tied = [f for f in fits if f.balanced_accuracy >= top - 1e-12]
    return tied[len(tied) // 2]


def evaluate(scores: Sequence[float], truth: Sequence[int], *, theta: float) -> dict[str, float]:
    """SC4 on the test split at a frozen theta: P/R/F1 for "supported", plus balanced accuracy."""
    pred = [s >= theta for s in scores]
    tp = sum(p and t == 1 for p, t in zip(pred, truth, strict=True))
    fp = sum(p and t == 0 for p, t in zip(pred, truth, strict=True))
    fn = sum(not p and t == 1 for p, t in zip(pred, truth, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy(truth, pred),
    }


def stratified_sample(
    items: Sequence[AggreFactItem], *, per_dataset: int, seed: int
) -> list[AggreFactItem]:
    """At most `per_dataset` rows from each sub-benchmark, deterministic for a seed.

    RAGTruth alone is half of dev; an unstratified sample would tune on one benchmark.
    """
    by_ds: dict[str, list[AggreFactItem]] = defaultdict(list)
    for it in items:
        by_ds[it.dataset].append(it)
    rng = random.Random(seed)
    out: list[AggreFactItem] = []
    for ds in sorted(by_ds):
        rows = by_ds[ds]
        out.extend(rng.sample(rows, per_dataset) if len(rows) > per_dataset else rows)
    return out


class RowScore(NamedTuple):
    dataset: str
    label: int
    entail: float  # DeBERTa max entailment over the top-n evidence sentences
    contra: float
    minicheck: float  # MiniCheck max support over whole-document chunks


def score_rows(items: Sequence[AggreFactItem], openers: Openers, cfg: VerifyCfg) -> list[RowScore]:
    """Raw verifier scores per row, same order. One model resident at a time (DECISION-3).

    Same inputs as `verify_claims` (evidence for DeBERTa, whole-document chunks for
    MiniCheck), so thresholds tuned here transfer to the pipeline. Rows sharing a document
    reuse one parsed `Document`.
    """
    docs: dict[str, Document] = {}
    for it in items:
        if it.doc not in docs:
            docs[it.doc] = to_document(
                LongDoc(id=f"aggrefact-{len(docs):05d}", article=it.doc, abstract=""), "aggrefact"
            )
    by_doc: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_doc[it.doc].append(i)

    evidence: dict[int, list[Span]] = {}
    with openers.embedder() as embed:
        for text, idxs in by_doc.items():
            found = retrieve_evidence(
                [items[i].claim for i in idxs], docs[text], embed, cfg.top_n_evidence
            )
            evidence.update(zip(idxs, found, strict=True))

    nli_scores: list[tuple[float, float]] = []
    with openers.nli() as nli:
        for i, it in enumerate(items):
            premises = [docs[it.doc].text_of(sp) for sp in evidence[i]]
            nli_scores.append(score_claim(it.claim, premises, nli, nli.label_idx))

    chunks = {
        t: [d.text_of(sp) for sp in chunk_spans(d, cfg.minicheck.chunk_words)]
        for t, d in docs.items()
    }
    with openers.minicheck() as mc:
        support = [support_score(it.claim, chunks[it.doc], mc) for it in items]

    return [
        RowScore(it.dataset, it.label, e, c, m)
        for it, (e, c), m in zip(items, nli_scores, support, strict=True)
    ]
