"""FR-4.3 near-duplicate clustering and FR-4.4 ranking. PRD §11.7 steps 6-7.

Single linkage with a cosine-distance cut: two candidates closer than `dup_distance` always
end up in one cluster, so no two representatives are duplicates (the FR-4.3 acceptance).
The number of clusters is not fixed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.cluster.hierarchy import fcluster, linkage

from app.keypoints.rank import unit


class Cluster(BaseModel):
    members: list[int]  # candidate indices, ascending
    rep: int  # highest-salience member


def cluster(
    vecs: NDArray[np.float32], sal: NDArray[np.float64], *, dup_distance: float
) -> list[Cluster]:
    n = len(vecs)
    if n == 0:
        return []
    if n == 1:
        return [Cluster(members=[0], rep=0)]
    # cosine distance on unit vectors, clipped: float error can make it slightly negative
    dist = np.clip(1.0 - unit(vecs) @ unit(vecs).T, 0.0, 2.0)
    condensed = dist[np.triu_indices(n, k=1)]
    labels = fcluster(linkage(condensed, method="single"), t=dup_distance, criterion="distance")
    out = []
    for lab in sorted(set(labels.tolist())):
        members = [i for i in range(n) if labels[i] == lab]
        # ties go to the earlier candidate, so the choice is deterministic
        rep = max(members, key=lambda i: (sal[i], -i))
        out.append(Cluster(members=members, rep=rep))
    return out


def select(clusters: Sequence[Cluster], positions: Sequence[int], k: int) -> list[Cluster]:
    """Top k: bigger clusters first (more sections agree), then earlier in the document."""
    return sorted(clusters, key=lambda c: (-len(c.members), positions[c.rep], c.rep))[:k]
