"""FR-5.4 label rule and FR-5.5 metrics (faithfulness, coverage, compression, per level)."""

from __future__ import annotations

import math

import pytest

from app.core.schemas import ClaimLabel, ClaimResult, SummaryNode
from app.verify.labels import label
from eval.metrics import compression_ratio, faithfulness, faithfulness_by_level

S, C, U = ClaimLabel.SUPPORTED, ClaimLabel.CONTRADICTED, ClaimLabel.UNSUPPORTED


def test_entail_at_threshold_is_supported() -> None:
    assert label(0.5, 0.0, theta_e=0.5, theta_c=0.5) is S


def test_entail_wins_over_contradiction() -> None:
    # FR-5.4 order: "supported if any premise entails", checked before contradiction.
    assert label(0.9, 0.9, theta_e=0.5, theta_c=0.5) is S


def test_contradiction_at_threshold_when_not_entailed() -> None:
    assert label(0.49, 0.5, theta_e=0.5, theta_c=0.5) is C


def test_neither_is_unsupported() -> None:
    assert label(0.49, 0.49, theta_e=0.5, theta_c=0.5) is U


def test_single_score_verifier_never_contradicts() -> None:
    # MiniCheck gives one support probability; it has no contradiction head.
    assert label(0.1, None, theta_e=0.5, theta_c=0.5) is U


def test_faithfulness_is_supported_over_total() -> None:
    assert faithfulness([S, S, C, U]) == 0.5


def test_faithfulness_of_no_claims_is_nan_not_zero_or_one() -> None:
    # An empty summary is neither perfectly faithful nor perfectly unfaithful;
    # NaN makes the aggregate (pandas mean) skip it instead of biasing it.
    assert math.isnan(faithfulness([]))


def _claim(node: str, lab: ClaimLabel) -> ClaimResult:
    return ClaimResult(
        claim="c",
        summary_sentence_idx=0,
        summary_node_id=node,
        deberta_entail=0.0,
        deberta_contra=0.0,
        minicheck_support=0.0,
        label_deberta=lab,
        label_minicheck=S,
    )


def test_faithfulness_by_level_groups_claims_by_their_node_level() -> None:
    nodes = [
        SummaryNode(id="a", level=0, text=""),
        SummaryNode(id="b", level=0, text=""),
        SummaryNode(id="root", level=1, text=""),
    ]
    claims = [_claim("a", S), _claim("b", U), _claim("root", S), _claim("root", C)]

    by_level = faithfulness_by_level(claims, nodes, verifier="deberta")

    assert by_level == {0: 0.5, 1: 0.5}
    assert faithfulness_by_level(claims, nodes, verifier="minicheck") == {0: 1.0, 1: 1.0}


def test_claim_on_unknown_node_is_an_error() -> None:
    with pytest.raises(KeyError):
        faithfulness_by_level([_claim("ghost", S)], [], verifier="deberta")


def test_compression_ratio_is_output_over_input() -> None:
    assert compression_ratio(out_tokens=100, in_tokens=1000) == 0.1


def test_compression_ratio_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        compression_ratio(out_tokens=1, in_tokens=0)
