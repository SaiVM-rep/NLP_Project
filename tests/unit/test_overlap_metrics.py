"""PRD §12.5: ROUGE for comparability, Spearman/Kendall vs SummEval ratings (SC4).

ROUGE is reported, never treated as evidence of faithfulness (PRD §12.5).
"""

from __future__ import annotations

import math

import pytest

from eval.metrics import correlation, per_doc_correlation, rouge

# ----------------------------------------------------------------- rouge ---


def test_identical_text_scores_one() -> None:
    r = rouge("The model works well. It is fast.", "The model works well. It is fast.")
    assert r == pytest.approx({"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0, "rougeLsum": 1.0})


def test_case_and_punctuation_do_not_matter() -> None:
    assert rouge("The Cat sat.", "the cat sat")["rouge1"] == pytest.approx(1.0)


def test_ccdv_tokenised_reference_matches_normal_prose() -> None:
    """ccdv abstracts are pre-tokenised ('short - term , ...'); our summaries are prose."""
    r = rouge("Short-term changes, studied.", "short - term changes , studied .")
    assert r["rouge1"] == pytest.approx(1.0)
    assert r["rouge2"] == pytest.approx(1.0)


def test_stemming_is_on() -> None:
    assert rouge("cats running", "cat run")["rouge1"] == pytest.approx(1.0)


def test_bigrams_and_order_matter_for_rouge2_and_l() -> None:
    r = rouge("alpha beta gamma", "gamma beta alpha")
    assert r["rouge1"] == pytest.approx(1.0)
    assert r["rouge2"] == pytest.approx(0.0)
    assert r["rougeL"] == pytest.approx(1 / 3)


def test_lsum_is_sentence_aware_where_l_is_not() -> None:
    # Same sentences, swapped order: plain L pays for the order, Lsum (per sentence) does not.
    a = "Alpha beta gamma delta. Epsilon zeta eta theta."
    b = "Epsilon zeta eta theta. Alpha beta gamma delta."
    r = rouge(a, b)
    assert r["rougeLsum"] == pytest.approx(1.0)
    assert r["rougeL"] < 1.0


def test_multiple_references_take_the_best_match() -> None:
    r = rouge("the cat sat", ["a dog ran", "the cat sat"])
    assert r["rouge1"] == pytest.approx(1.0)
    assert rouge("the cat sat", ["the cat sat"]) == rouge("the cat sat", "the cat sat")


def test_no_references_raises() -> None:
    with pytest.raises(ValueError):
        rouge("x", [])


def test_empty_summary_scores_zero_not_error() -> None:
    assert rouge("", "some reference") == {
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
        "rougeLsum": 0.0,
    }


# ----------------------------------------------------------- correlation ---


def test_perfect_monotonic_agreement() -> None:
    c = correlation([1, 2, 3, 4], [10, 20, 30, 400])
    assert c.spearman == pytest.approx(1.0)
    assert c.kendall == pytest.approx(1.0)
    assert c.n == 4


def test_reversed_order_is_minus_one() -> None:
    c = correlation([1, 2, 3], [3, 2, 1])
    assert (c.spearman, c.kendall) == pytest.approx((-1.0, -1.0))


def test_nan_pairs_dropped_pairwise() -> None:
    """faithfulness() is NaN for a summary with no claims; that pair is skipped, not zeroed."""
    c = correlation([1, float("nan"), 2, 3], [1, 99, 2, 3])
    assert c.n == 3
    assert c.spearman == pytest.approx(1.0)


def test_too_few_points_is_nan() -> None:
    c = correlation([1, 2], [1, 2])
    assert math.isnan(c.spearman) and math.isnan(c.kendall)
    assert c.n == 2


def test_constant_input_is_nan_without_warning() -> None:
    # scipy warns (ConstantInputWarning); pytest runs with warnings-as-errors
    c = correlation([5, 5, 5, 5], [1, 2, 3, 4])
    assert math.isnan(c.spearman) and math.isnan(c.kendall)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        correlation([1, 2, 3], [1, 2])


def test_per_doc_correlation_averages_within_document_scores() -> None:
    """SummEval 'summary-level': correlate the 16 summaries of each doc, then average."""
    groups = {
        "d1": ([1, 2, 3], [1, 2, 3]),  # +1
        "d2": ([1, 2, 3], [3, 2, 1]),  # -1
        "d3": ([1, 2, 3], [7, 7, 7]),  # constant human scores: undefined, skipped
    }
    c = per_doc_correlation(groups)
    assert c.spearman == pytest.approx(0.0)
    assert c.kendall == pytest.approx(0.0)
    assert c.n == 2  # documents that contributed
