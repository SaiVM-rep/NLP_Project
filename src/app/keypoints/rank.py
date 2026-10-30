"""FR-4.2: candidate salience. PRD §11.7 steps 3-5.

`salience = lambda * graph_score + (1 - lambda) * doc_score`. PageRank sums to 1 over the
candidates, so its scale shrinks as candidates grow; it is divided by its max to sit on the
same [0, 1]-ish scale as the cosine `doc_score` before mixing.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import NDArray


def unit(m: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(norms == 0, 1, norms)).astype(np.float32)


def graph_scores(vecs: NDArray[np.float32]) -> NDArray[np.float64]:
    """Weighted PageRank over the candidate cosine-similarity graph, scaled to max 1."""
    n = len(vecs)
    sim = np.clip(unit(vecs) @ unit(vecs).T, 0.0, None)  # PageRank needs non-negative weights
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_weighted_edges_from(
        (i, j, float(sim[i, j])) for i in range(n) for j in range(i + 1, n) if sim[i, j] > 0
    )
    pr = nx.pagerank(g, weight="weight")
    scores = np.array([pr[i] for i in range(n)], dtype=np.float64)
    out: NDArray[np.float64] = scores / scores.max()
    return out


def doc_scores(vecs: NDArray[np.float32], sent_vecs: NDArray[np.float32]) -> NDArray[np.float64]:
    """Cosine of each candidate to the centroid of the document's sentence embeddings."""
    centroid = unit(sent_vecs).mean(axis=0, keepdims=True)
    return (unit(vecs) @ unit(centroid).T)[:, 0].astype(np.float64)


def salience(
    vecs: NDArray[np.float32], sent_vecs: NDArray[np.float32], lambda_graph: float
) -> NDArray[np.float64]:
    return lambda_graph * graph_scores(vecs) + (1 - lambda_graph) * doc_scores(vecs, sent_vecs)
