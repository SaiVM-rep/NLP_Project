"""FR-5.3, PRD §11.8 step 5: MiniCheck chunking + aggregation (pure, no torch).

Mirrors upstream Liyan06/MiniCheck@b58b9fa `Inferencer.inference_per_example`:
sentences packed greedily into chunks of <= chunk_size whitespace words, max over chunks.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from app.core.schemas import Document, Span
from app.verify.minicheck import chunk_spans, pick_chunks, support_score


def doc_of(sentences: list[str]) -> Document:
    text = " ".join(sentences)
    spans, pos = [], 0
    for s in sentences:
        spans.append(Span(start=pos, end=pos + len(s)))
        pos += len(s) + 1
    return Document(doc_id="d", source="pdf", title="t", full_text=text, sentences=spans)


def texts(doc: Document, spans: list[Span]) -> list[str]:
    return [doc.full_text[s.start : s.end] for s in spans]


def test_packs_sentences_up_to_word_budget() -> None:
    d = doc_of(["a b c.", "d e.", "f g h i.", "j."])  # 3, 2, 4, 1 words
    assert texts(d, chunk_spans(d, max_words=5)) == ["a b c. d e.", "f g h i. j."]


def test_budget_is_inclusive() -> None:
    d = doc_of(["a b.", "c d e."])  # 2 + 3 == 5 fits; upstream breaks only on > n
    assert texts(d, chunk_spans(d, max_words=5)) == ["a b. c d e."]


def test_oversized_sentence_is_its_own_chunk_not_dropped() -> None:
    d = doc_of(["a.", "w w w w w w w w.", "b."])
    assert texts(d, chunk_spans(d, max_words=3)) == ["a.", "w w w w w w w w.", "b."]


def test_chunks_are_spans_over_full_text() -> None:
    # Offsets are the backbone (SC2): chunk text is a slice, never a re-join.
    text = "One two.\n\nThree four."
    d = Document(
        doc_id="d",
        source="pdf",
        title="t",
        full_text=text,
        sentences=[Span(start=0, end=8), Span(start=10, end=21)],
    )
    assert chunk_spans(d, max_words=10) == [Span(start=0, end=21)]


def test_no_sentences_no_chunks() -> None:
    d = Document(doc_id="d", source="pdf", title="t", full_text="", sentences=[])
    assert chunk_spans(d, max_words=500) == []


def test_bad_budget_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_spans(doc_of(["a."]), max_words=0)


def test_support_score_is_max_over_chunks() -> None:
    seen: list[tuple[list[str], list[str]]] = []

    def probs(chunks: list[str], claims: list[str]) -> NDArray[np.float32]:
        seen.append((chunks, claims))
        return np.array([0.2, 0.9, 0.4], dtype=np.float32)

    assert support_score("c", ["x", "y", "z"], probs) == pytest.approx(0.9)
    assert seen == [(["x", "y", "z"], ["c", "c", "c"])]  # chunk is the premise


def test_support_score_no_chunks_is_zero_without_model_call() -> None:
    def probs(chunks: list[str], claims: list[str]) -> NDArray[np.float32]:
        raise AssertionError("must not be called")

    assert support_score("c", [], probs) == 0.0


def test_support_score_python_float_clipped() -> None:
    def probs(chunks: list[str], claims: list[str]) -> NDArray[np.float32]:
        return np.array([1.0000001], dtype=np.float32)

    s = support_score("c", ["x"], probs)
    assert type(s) is float and s == 1.0


# ---------------------------------------------------- top-k chunk selection ---
# Uploads score each claim against its k best chunks only (2026-11-25 experiment:
# k=5 kept 253/254 labels on 7 papers at 43% of the MiniCheck pairs).

FOUR = ["s0 a b.", "s1 c d.", "s2 e f.", "s3 g h."]


def test_pick_chunks_ranks_chunks_by_their_best_sentence() -> None:
    d = doc_of(FOUR)
    chunks = chunk_spans(d, 6)  # two sentences per chunk -> chunks [s0 s1], [s2 s3]
    sim = np.array([[0.1, 0.2, 0.9, 0.0], [0.8, 0.0, 0.3, 0.3]], dtype=np.float32)
    assert pick_chunks(sim, d, chunks, 1) == [[1], [0]]


def test_pick_chunks_ties_keep_document_order() -> None:
    d = doc_of(FOUR)
    chunks = chunk_spans(d, 3)  # one sentence per chunk
    sim = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    assert pick_chunks(sim, d, chunks, 2) == [[0, 1]]


def test_pick_chunks_k_at_least_chunk_count_keeps_all() -> None:
    d = doc_of(FOUR)
    chunks = chunk_spans(d, 3)
    sim = np.array([[0.1, 0.4, 0.3, 0.2]], dtype=np.float32)
    assert pick_chunks(sim, d, chunks, 9) == [[1, 2, 3, 0]]
