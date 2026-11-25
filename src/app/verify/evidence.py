"""FR-5.2: evidence retrieval. PRD §11.8 step 3.

The embedder is injected (`texts -> (len, dim) array`) so this is testable without
torch; the real one is a sentence-transformer, loaded and unloaded by the caller
(DECISION-3, sequential model residency).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from app.core.schemas import Document, Span

Embed = Callable[[list[str]], NDArray[np.float32]]


def _unit(m: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    # Zero vector scores 0, never NaN.
    return (m / np.where(norms == 0, 1, norms)).astype(np.float32)


def _top(scores: NDArray[np.float32], n: int) -> list[list[int]]:
    # Stable sort on the negated score: equal scores keep document order.
    order = np.argsort(-scores, axis=1, kind="stable")[:, :n]
    return [[int(i) for i in row] for row in order]


def top_n(
    claim_vecs: NDArray[np.float32], sent_vecs: NDArray[np.float32], n: int
) -> list[list[int]]:
    """Indices of the n most cosine-similar sentences per claim, best first."""
    return _top(_unit(claim_vecs) @ _unit(sent_vecs).T, n)


def sentence_similarity(claims: list[str], doc: Document, embed: Embed) -> NDArray[np.float32]:
    """claims x doc.sentences cosine similarity."""
    sents = [doc.full_text[s.start : s.end] for s in doc.sentences]
    return _unit(embed(claims)) @ _unit(embed(sents)).T


def evidence_from(sim: NDArray[np.float32], doc: Document, n: int) -> list[list[Span]]:
    return [[doc.sentences[i] for i in row] for row in _top(sim, n)]


def retrieve_evidence(claims: list[str], doc: Document, embed: Embed, n: int) -> list[list[Span]]:
    if not claims:
        return []
    if not doc.sentences:
        return [[] for _ in claims]
    return evidence_from(sentence_similarity(claims, doc, embed), doc, n)
