"""FR-5.5 metrics, computed only from stored claim records so they are reproducible.

Coverage is `faithfulness` applied the other way round: the labels of reference-abstract
facts checked against the generated summary (PRD §11.8, definition pending §19 q2).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from app.core.schemas import ClaimLabel, ClaimResult, SummaryNode


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
