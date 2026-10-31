"""Key-point attribution against a labelled reference (FR-4.5 precision, P5 tuning item)."""

import pytest

from eval.attribution import score_attribution, sentence_index

SENTS = [{"start": 0, "end": 10}, {"start": 11, "end": 25}, {"start": 26, "end": 40}]


def test_sentence_index_finds_the_sentence_a_span_points_at() -> None:
    assert sentence_index(SENTS, {"start": 11, "end": 25}) == 1


def test_sentence_index_of_no_evidence_is_none() -> None:
    assert sentence_index(SENTS, None) is None


def test_sentence_index_rejects_a_span_that_is_not_a_sentence() -> None:
    with pytest.raises(ValueError):
        sentence_index(SENTS, {"start": 3, "end": 30})


def test_score_counts_a_pick_in_the_gold_set_as_correct() -> None:
    m = score_attribution(
        [
            {"system": 1, "gold": [1, 2], "unsupported": False},  # right
            {"system": 0, "gold": [2], "unsupported": False},  # wrong sentence
            {"system": None, "gold": [], "unsupported": True},  # rightly unattributed
            {"system": 2, "gold": [], "unsupported": True},  # attributed a fabrication
        ]
    )
    assert m["n"] == 4
    assert m["precision_at_1"] == pytest.approx(1 / 3)  # of the 3 attributed, 1 right
    assert m["accuracy"] == pytest.approx(2 / 4)  # right pick or right abstention
    assert m["n_unsupported"] == 2


def test_score_with_nothing_attributed_has_nan_precision() -> None:
    m = score_attribution([{"system": None, "gold": [0], "unsupported": False}])
    assert m["precision_at_1"] != m["precision_at_1"]  # NaN
    assert m["accuracy"] == 0.0


def test_rerank_picks_the_candidate_the_nli_model_finds_most_entailing() -> None:
    from eval.attribution import rerank

    assert rerank([7, 3, 9], [0.1, 0.8, 0.5]) == 3


def test_rerank_tie_keeps_the_retriever_order() -> None:
    from eval.attribution import rerank

    # candidates arrive in retriever (cosine) order; equal entailment -> the earlier one
    assert rerank([7, 3, 9], [0.4, 0.4, 0.1]) == 7


def test_rerank_needs_one_score_per_candidate() -> None:
    from eval.attribution import rerank

    with pytest.raises(ValueError):
        rerank([1, 2], [0.5])
    with pytest.raises(ValueError):
        rerank([], [])
