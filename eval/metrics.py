"""FR-5.5 metrics, computed only from stored claim records so they are reproducible.

Coverage is `faithfulness` applied the other way round: the labels of reference-abstract
facts checked against the generated summary (PRD §11.8, definition pending §19 q2).

ROUGE is for comparability only, never evidence of faithfulness (PRD §12.5).
Correlations validate metrics against SummEval human ratings (SC4).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any, Literal, NamedTuple

from app.core.schemas import ClaimLabel, ClaimResult, SummaryNode
from app.ingest.spans import sentence_spans


def faithfulness(labels: Sequence[ClaimLabel]) -> float:
    """Supported / total. NaN for no claims, so aggregates skip it instead of biasing."""
    if not labels:
        return float("nan")
    return sum(lab is ClaimLabel.SUPPORTED for lab in labels) / len(labels)


def faithfulness_by_level(
    claims: Sequence[ClaimResult],
    nodes: Sequence[SummaryNode],
    verifier: Literal["deberta", "minicheck"],
) -> dict[int, float]:
    """E3: faithfulness per merge level. A claim on an unknown node raises KeyError."""
    level = {n.id: n.level for n in nodes}
    grouped: dict[int, list[ClaimLabel]] = defaultdict(list)
    for c in claims:
        lab = c.label_deberta if verifier == "deberta" else c.label_minicheck
        grouped[level[c.summary_node_id]].append(lab)
    return {lvl: faithfulness(labs) for lvl, labs in sorted(grouped.items())}


def compression_ratio(*, out_tokens: int, in_tokens: int) -> float:
    if in_tokens <= 0:
        raise ValueError(f"in_tokens must be positive, got {in_tokens}")
    return out_tokens / in_tokens


# ----------------------------------------------------------------- rouge ---

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL", "rougeLsum")


@cache
def _scorer() -> Any:
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=True)


def _lines(text: str) -> str:
    # rougeLsum splits on newlines; arXiv/PubMed papers report it on sentence splits
    return "\n".join(text[s.start : s.end] for s in sentence_spans(text, 0))


def rouge(summary: str, reference: str | Sequence[str]) -> dict[str, float]:
    """F1 for ROUGE-1/2/L/Lsum, Porter stemming, case and punctuation ignored.

    Several references: best F1 per metric (rouge-score `score_multi`), as for SummEval.
    """
    refs = [reference] if isinstance(reference, str) else list(reference)
    if not refs:
        raise ValueError("rouge needs at least one reference")
    # (targets, prediction)
    scores = _scorer().score_multi([_lines(r) for r in refs], _lines(summary))
    return {t: float(scores[t].fmeasure) for t in ROUGE_TYPES}


# ----------------------------------------------------------- correlation ---


class Correlation(NamedTuple):
    spearman: float
    kendall: float
    n: int  # pairs (or documents, for per_doc_correlation) that contributed


NAN = float("nan")


def correlation(x: Sequence[float], y: Sequence[float]) -> Correlation:
    """Spearman rho and Kendall tau-b. NaN pairs dropped; <3 points or constant -> NaN."""
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} vs {len(y)}")
    pairs = [(a, b) for a, b in zip(x, y, strict=True) if not (math.isnan(a) or math.isnan(b))]
    n = len(pairs)
    xs, ys = [a for a, _ in pairs], [b for _, b in pairs]
    # Checked here rather than left to scipy, whose ConstantInputWarning is an error in tests.
    if n < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return Correlation(NAN, NAN, n)
    from scipy.stats import kendalltau, spearmanr

    return Correlation(float(spearmanr(xs, ys).statistic), float(kendalltau(xs, ys).statistic), n)


def per_doc_correlation(
    groups: Mapping[str, tuple[Sequence[float], Sequence[float]]],
) -> Correlation:
    """SummEval summary-level: correlate within each document, average over documents."""
    per = [correlation(x, y) for x, y in groups.values()]
    ok = [c for c in per if not math.isnan(c.spearman)]
    if not ok:
        return Correlation(NAN, NAN, 0)
    return Correlation(
        sum(c.spearman for c in ok) / len(ok), sum(c.kendall for c in ok) / len(ok), len(ok)
    )
