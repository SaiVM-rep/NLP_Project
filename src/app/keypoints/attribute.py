"""FR-4.5: attribute each key point to its most similar source sentence. PRD §11.7 step 8."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from app.core.schemas import Span
from app.keypoints.rank import unit


def attribute(
    vecs: NDArray[np.float32],
    sent_vecs: NDArray[np.float32],
    sentences: Sequence[Span],
    *,
    theta_attr: float,
) -> list[tuple[Span | None, float]]:
    """(best sentence or None if below `theta_attr`, its cosine) per point."""
    if not sentences:
        return [(None, 0.0) for _ in range(len(vecs))]
    sim = unit(vecs) @ unit(sent_vecs).T
    best = sim.argmax(axis=1)  # first maximum: ties go to the earlier sentence
    out: list[tuple[Span | None, float]] = []
    for i, j in enumerate(best):
        s = float(sim[i, j])
        out.append((sentences[j] if s >= theta_attr else None, s))
    return out
