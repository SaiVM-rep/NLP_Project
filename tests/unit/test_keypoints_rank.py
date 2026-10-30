"""FR-4.2..4.5: salience, near-duplicate clustering, ranking, attribution. PRD §11.7 steps 3-8.

Hand-made vectors stand in for the sentence-transformer, so every expected value is exact.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

from app.core.schemas import Span
from app.keypoints.attribute import attribute
from app.keypoints.cluster import Cluster, cluster, select
from app.keypoints.rank import doc_scores, graph_scores, salience


def v(*rows: list[float]) -> NDArray[np.float32]:
    return np.array(rows, dtype=np.float32)


# three candidates about A, one about B: the A ones support each other in the graph
HUB = v([1, 0], [0.9, 0.1], [0.95, 0.05], [0, 1])


# ------------------------------------------------------------------ rank ---


def test_graph_score_favours_candidates_others_agree_with() -> None:
    g = graph_scores(HUB)
    assert g.argmin() == 3
    assert g.max() == pytest.approx(1.0)  # scaled to [0, 1] so it mixes with doc_score


def test_graph_score_of_a_single_candidate_is_one() -> None:
    assert graph_scores(v([0.3, 0.4])).tolist() == [1.0]


def test_unrelated_candidates_score_equally() -> None:
    assert graph_scores(v([1, 0], [0, 1])).tolist() == pytest.approx([1.0, 1.0])


def test_doc_score_is_cosine_to_the_sentence_centroid() -> None:
    sents = v([1, 0], [1, 0], [0, 1])  # centroid of unit vectors leans to A
    d = doc_scores(v([1, 0], [0, 1]), sents)
    c = np.array([2, 1]) / np.linalg.norm([2, 1])
    assert d.tolist() == pytest.approx([c[0], c[1]])


def test_salience_mixes_graph_and_doc_score_by_lambda() -> None:
    sents = v([0, 1])  # document is about B; the graph prefers A
    g, d = graph_scores(HUB), doc_scores(HUB, sents)
    assert salience(HUB, sents, 1.0) == pytest.approx(g)
    assert salience(HUB, sents, 0.0) == pytest.approx(d)
    assert salience(HUB, sents, 0.5) == pytest.approx(0.5 * g + 0.5 * d)


# --------------------------------------------------------------- cluster ---


def test_near_duplicates_form_one_cluster_with_the_most_salient_as_representative() -> None:
    out = cluster(HUB, np.array([0.2, 0.9, 0.5, 0.4]), dup_distance=0.15)
    assert sorted(out, key=lambda c: c.members) == [
        Cluster(members=[0, 1, 2], rep=1),
        Cluster(members=[3], rep=3),
    ]


def test_single_candidate_is_its_own_cluster() -> None:
    assert cluster(v([1, 0]), np.array([0.5]), dup_distance=0.15) == [Cluster(members=[0], rep=0)]


def test_no_candidates_no_clusters() -> None:
    assert cluster(np.zeros((0, 2), dtype=np.float32), np.zeros(0), dup_distance=0.15) == []


@settings(max_examples=200, deadline=None)
@given(
    arrays(
        np.float32, st.tuples(st.integers(2, 30), st.just(4)), elements=st.floats(-1, 1, width=32)
    ),
    st.floats(0.01, 0.5),
)
def test_no_two_representatives_are_duplicates(vecs: NDArray[np.float32], dup: float) -> None:
    """FR-4.3 acceptance: no two output points within the duplicate threshold."""
    vecs = vecs + np.float32(1e-3)  # keep away from the zero vector, whose cosine is undefined
    sal = np.arange(len(vecs), dtype=np.float64)
    reps = [c.rep for c in cluster(vecs, sal, dup_distance=dup)]
    u = vecs[reps] / np.linalg.norm(vecs[reps], axis=1, keepdims=True)
    dist = 1 - u @ u.T
    np.fill_diagonal(dist, np.inf)
    assert (dist > dup - 1e-5).all()
    assert sorted(m for c in cluster(vecs, sal, dup_distance=dup) for m in c.members) == list(
        range(len(vecs))
    )


def test_select_ranks_by_cluster_size_then_document_position() -> None:
    clusters = [
        Cluster(members=[5], rep=5),
        Cluster(members=[3, 4], rep=4),
        Cluster(members=[0], rep=0),
        Cluster(members=[1, 2], rep=2),
    ]
    positions = [0, 1, 2, 3, 4, 5]
    assert select(clusters, positions, k=3) == [
        Cluster(members=[1, 2], rep=2),  # size 2, earliest
        Cluster(members=[3, 4], rep=4),
        Cluster(members=[0], rep=0),  # size 1, earliest
    ]


def test_select_is_independent_of_input_order() -> None:
    clusters = [Cluster(members=[i], rep=i) for i in range(6)]
    pos = [5, 4, 3, 2, 1, 0]
    assert select(clusters, pos, k=6) == select(clusters[::-1], pos, k=6)


# ------------------------------------------------------------- attribute ---


SENTS = [Span(start=0, end=10), Span(start=11, end=20)]


def test_attributes_to_the_most_similar_source_sentence() -> None:
    out = attribute(v([1, 0.1], [0.1, 1]), v([1, 0], [0, 1]), SENTS, theta_attr=0.5)
    assert [span for span, _ in out] == SENTS
    assert out[0][1] == pytest.approx(1 / np.sqrt(1.01))


def test_below_theta_attr_is_unattributed_but_keeps_its_similarity() -> None:
    ((span, sim),) = attribute(v([1, 1]), v([1, 0], [0, 1]), SENTS, theta_attr=0.8)
    assert span is None
    assert sim == pytest.approx(1 / np.sqrt(2))


def test_document_without_sentences_attributes_nothing() -> None:
    assert attribute(v([1, 0]), np.zeros((0, 2), dtype=np.float32), [], theta_attr=0.5) == [
        (None, 0.0)
    ]
