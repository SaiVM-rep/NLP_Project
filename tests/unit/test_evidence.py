"""FR-5.2: top-n source sentences per claim by embedding cosine; evidence stored as offsets."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from app.core.schemas import Document, Span
from app.verify.evidence import retrieve_evidence, top_n

TEXT = "Cats purr. Dogs bark. Birds sing."
DOC = Document(
    doc_id="d",
    source="pdf",
    title="t",
    full_text=TEXT,
    sentences=[Span(start=0, end=10), Span(start=11, end=21), Span(start=22, end=33)],
)

# One axis per animal; deliberately unnormalised so the test proves cosine, not raw dot.
VECS = {
    "Cats purr.": [10.0, 0.0, 0.0],
    "Dogs bark.": [0.0, 1.0, 0.0],
    "Birds sing.": [0.0, 0.0, 1.0],
    "a cat": [1.0, 0.0, 0.1],
    "a dog": [0.1, 1.0, 0.0],
}


def fake_embed(texts: list[str]) -> NDArray[np.float32]:
    return np.array([VECS[t] for t in texts], dtype=np.float32)


def test_top_n_orders_by_cosine_not_dot_product() -> None:
    claims = fake_embed(["a dog"])
    sents = fake_embed(["Cats purr.", "Dogs bark.", "Birds sing."])
    # Raw dot would rank "Cats purr." (10 * 0.1 = 1.0) level with "Dogs bark." (1.0).
    assert top_n(claims, sents, n=2) == [[1, 0]]


def test_top_n_larger_than_corpus_returns_everything() -> None:
    sents = fake_embed(["Cats purr.", "Dogs bark."])
    assert sorted(top_n(fake_embed(["a cat"]), sents, n=5)[0]) == [0, 1]


def test_ties_break_by_document_order_so_runs_are_reproducible() -> None:
    sents = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert top_n(np.array([[1.0, 0.0]], dtype=np.float32), sents, n=3) == [[0, 1, 2]]


def test_zero_vector_does_not_produce_nan_ranking() -> None:
    sents = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert top_n(np.array([[1.0, 0.0]], dtype=np.float32), sents, n=1) == [[1]]


def test_retrieve_evidence_returns_document_spans() -> None:
    ev = retrieve_evidence(["a cat", "a dog"], DOC, fake_embed, n=1)
    assert ev == [[Span(start=0, end=10)], [Span(start=11, end=21)]]
    assert TEXT[ev[1][0].start : ev[1][0].end] == "Dogs bark."


def test_no_claims_or_no_sentences_is_empty() -> None:
    assert retrieve_evidence([], DOC, fake_embed, n=3) == []
    empty = DOC.model_copy(update={"sentences": []})
    assert retrieve_evidence(["a cat"], empty, fake_embed, n=3) == [[]]
