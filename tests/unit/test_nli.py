"""FR-5.3, PRD §11.8 step 4: DeBERTa-v3 NLI over (premise sentence, claim) pairs."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from app.verify.nli import LabelIdx, label_indices, score_claim

# MoritzLaurer checkpoints vs cross-encoder: same labels, different order. Measured 2026-10-29.
MORITZ = {"0": "entailment", "1": "neutral", "2": "contradiction"}
CROSS_ENCODER = {0: "contradiction", 1: "entailment", 2: "neutral"}


def test_label_indices_read_from_config_not_assumed() -> None:
    assert label_indices(MORITZ) == LabelIdx(entail=0, contra=2)
    assert label_indices(CROSS_ENCODER) == LabelIdx(entail=1, contra=0)


def test_label_indices_case_insensitive() -> None:
    assert label_indices({0: "ENTAILMENT", 1: "Neutral", 2: "Contradiction"}) == LabelIdx(0, 2)


def test_label_indices_missing_label_raises() -> None:
    # A 2-way head (entail / not_entail) cannot give a contradiction score; fail loudly.
    with pytest.raises(ValueError, match="contradiction"):
        label_indices({0: "entailment", 1: "not_entailment"})


def _table(rows: dict[str, list[float]]):  # type: ignore[no-untyped-def]
    calls: list[tuple[list[str], list[str]]] = []

    def probs(premises: list[str], hyps: list[str]) -> NDArray[np.float32]:
        calls.append((premises, hyps))
        return np.array([rows[p] for p in premises], dtype=np.float32)

    return probs, calls


def test_score_claim_takes_max_entail_and_max_contra_independently() -> None:
    # columns follow MORITZ: entail, neutral, contra
    probs, _ = _table(
        {
            "p1": [0.7, 0.2, 0.1],
            "p2": [0.1, 0.1, 0.8],
            "p3": [0.3, 0.6, 0.1],
        }
    )
    entail, contra = score_claim("c", ["p1", "p2", "p3"], probs, LabelIdx(0, 2))
    assert entail == pytest.approx(0.7)  # from p1
    assert contra == pytest.approx(0.8)  # from p2, not p1's contra


def test_score_claim_pairs_each_premise_with_the_claim() -> None:
    probs, calls = _table({"p1": [1, 0, 0], "p2": [1, 0, 0]})
    score_claim("the claim", ["p1", "p2"], probs, LabelIdx(0, 2))
    # premise first, claim as hypothesis — swapping them changes NLI meaning
    assert calls == [(["p1", "p2"], ["the claim", "the claim"])]


def test_score_claim_no_evidence_is_zero_not_error() -> None:
    probs, calls = _table({})
    assert score_claim("c", [], probs, LabelIdx(0, 2)) == (0.0, 0.0)
    assert calls == []  # no model call for nothing


def test_score_claim_returns_python_floats_in_unit_range() -> None:
    probs, _ = _table({"p": [1.0000001, 0.0, 0.0]})  # fp16 softmax can overshoot by an ulp
    entail, contra = score_claim("c", ["p"], probs, LabelIdx(0, 2))
    assert type(entail) is float and type(contra) is float
    assert 0.0 <= entail <= 1.0  # ClaimResult validates ge=0 le=1
