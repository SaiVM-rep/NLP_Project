"""Key-point attribution vs a labelled reference (FR-4.5).

A key point is attributed to one sentence of the paper (or none, "unattributed"). The
reference marks every sentence that on its own supports the point, or "unsupported" if
nothing does. Correct = the pick is any reference sentence, or the system abstained on an
unsupported point.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def sentence_index(sentences: Sequence[dict[str, int]], span: dict[str, int] | None) -> int | None:
    """Index of the sentence `span` is; None for no evidence. Attribution evidence is always
    a whole sentence, so anything else is a bug upstream, not something to round."""
    if span is None:
        return None
    for i, s in enumerate(sentences):
        if s["start"] == span["start"] and s["end"] == span["end"]:
            return i
    raise ValueError(f"evidence {span} is not a sentence of the document")


def score_attribution(items: Sequence[dict[str, Any]]) -> dict[str, float]:
    """items: {"system": idx | None, "gold": [idx...], "unsupported": bool}."""
    attributed = [it for it in items if it["system"] is not None]
    right_pick = [it for it in attributed if it["system"] in set(it["gold"])]
    right_abstain = [it for it in items if it["system"] is None and it["unsupported"]]
    return {
        "n": len(items),
        "n_attributed": len(attributed),
        "n_unsupported": sum(bool(it["unsupported"]) for it in items),
        "precision_at_1": len(right_pick) / len(attributed) if attributed else math.nan,
        "accuracy": (len(right_pick) + len(right_abstain)) / len(items) if items else math.nan,
    }
