"""P8 pilot runner (FR-7.2): E1 Route A vs forced Route B, E5 perturbations. Pure parts only."""

import json
import math
from pathlib import Path

import pytest

from app.core.schemas import ClaimLabel, ClaimResult, Route, SummaryNode
from app.ingest.spans import RawSection
from app.summarize.common import SummaryResult
from app.verify.pipeline import VerifiedSummary
from eval.run import (
    ArmResult,
    aggregate_e1,
    aggregate_e5,
    e1_row,
    e5_items,
    exp_doc,
    run_e1,
)

ABS = RawSection("Abstract", 0, ["We propose X. It works."])
BODY = [
    RawSection("Introduction", 1, ["X is a method for sentence embeddings."]),
    RawSection("Results", 1, ["X is fast and accurate."]),
]


def _claim(node: str, deb: ClaimLabel, mc: ClaimLabel, text: str = "c") -> ClaimResult:
    return ClaimResult(
        claim=text,
        summary_sentence_idx=0,
        summary_node_id=node,
        evidence=[],
        deberta_entail=0.5,
        deberta_contra=0.0,
        minicheck_support=0.5,
        label_deberta=deb,
        label_minicheck=mc,
    )


S, U = ClaimLabel.SUPPORTED, ClaimLabel.UNSUPPORTED


def _arm(route: Route, levels: dict[str, int], claims: list[ClaimResult]) -> ArmResult:
    nodes = [SummaryNode(id=i, level=lv, text="t") for i, lv in levels.items()]
    res = SummaryResult(
        doc_id="d",
        summary="X is fast.",
        route=route,
        tokens=100,
        tau=50,
        prompt_hashes={},
        model="qwen3:8b",
    )
    return ArmResult(res, VerifiedSummary(nodes=nodes, claims=claims), seconds=12.5)


# ------------------------------------------------------------------ exp_doc ---


def test_exp_doc_drops_the_abstract_and_keeps_it_as_reference() -> None:
    d = exp_doc("p", "Title", [ABS, *BODY])
    assert d.reference == "We propose X. It works."
    assert "We propose X" not in d.doc.full_text
    assert "sentence embeddings" in d.doc.full_text
    assert [s.heading for s in d.doc.sections] == ["Introduction", "Results"]


def test_exp_doc_without_an_abstract_is_refused() -> None:
    with pytest.raises(ValueError, match="abstract"):
        exp_doc("p", "Title", BODY)


# ------------------------------------------------------------------- e1_row ---


def test_e1_row_route_a_scores_its_only_level() -> None:
    arm = _arm(Route.A, {"root": 0}, [_claim("root", S, S), _claim("root", U, S)])
    row = e1_row("p", "A", arm, reference="X is fast.")
    assert row["route"] == "A" and row["n_claims"] == 2
    assert row["faith_deberta"] == 0.5 and row["faith_minicheck"] == 1.0
    assert row["seconds"] == 12.5
    assert row["rouge1"] == pytest.approx(1.0)


def test_e1_row_route_b_headline_is_the_root_and_levels_are_kept() -> None:
    # final summary = top level; lower levels feed E3
    arm = _arm(
        Route.B,
        {"c0": 0, "c1": 0, "root": 1},
        [_claim("c0", U, U), _claim("c1", S, S), _claim("root", S, S)],
    )
    row = e1_row("p", "B", arm, reference="X is fast.")
    assert row["faith_deberta"] == 1.0  # root only
    assert row["levels_deberta"] == {"0": 0.5, "1": 1.0}
    assert row["n_nodes"] == 3


def test_e1_row_with_no_claims_is_nan_not_zero() -> None:
    row = e1_row("p", "A", _arm(Route.A, {"root": 0}, []), reference="r")
    assert math.isnan(row["faith_deberta"])


# ------------------------------------------------------------------- run_e1 ---


async def test_run_e1_runs_both_arms_and_forces_route_b(tmp_path: Path) -> None:
    docs = [exp_doc("p1", "T", [ABS, *BODY])]
    calls: list[int | None] = []

    async def run_arm(doc, tau):  # type: ignore[no-untyped-def]
        calls.append(tau)
        route = Route.A if tau is None else Route.B
        return _arm(route, {"root": 0}, [_claim("root", S, S)])

    rows = await run_e1(docs, run_arm, tmp_path / "per_doc.jsonl", tau_b=800)
    assert calls == [None, 800]
    assert [(r["doc_id"], r["arm"]) for r in rows] == [("p1", "A"), ("p1", "B")]
    lines = (tmp_path / "per_doc.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(ln)["arm"] for ln in lines] == ["A", "B"]


async def test_run_e1_resumes_without_redoing_finished_arms(tmp_path: Path) -> None:
    out = tmp_path / "per_doc.jsonl"
    out.write_text(json.dumps({"doc_id": "p1", "arm": "A", "route": "A"}) + "\n", encoding="utf-8")
    calls: list[int | None] = []

    async def run_arm(doc, tau):  # type: ignore[no-untyped-def]
        calls.append(tau)
        return _arm(Route.B, {"root": 0}, [])

    rows = await run_e1([exp_doc("p1", "T", [ABS, *BODY])], run_arm, out, tau_b=800)
    assert calls == [800]
    assert len(rows) == 2


async def test_run_e1_refuses_an_arm_that_took_the_wrong_route(tmp_path: Path) -> None:
    async def run_arm(doc, tau):  # type: ignore[no-untyped-def]
        return _arm(Route.A, {"root": 0}, [])  # B arm came back as A: tau too large

    with pytest.raises(RuntimeError, match="route"):
        await run_e1([exp_doc("p1", "T", [ABS, *BODY])], run_arm, tmp_path / "o.jsonl", tau_b=1)


def test_aggregate_e1_means_per_arm_skipping_nan() -> None:
    rows = [
        {"arm": "A", "faith_deberta": 0.5, "faith_minicheck": 1.0, "rouge1": 0.2, "seconds": 10},
        {
            "arm": "A",
            "faith_deberta": float("nan"),
            "faith_minicheck": 0.0,
            "rouge1": 0.4,
            "seconds": 30,
        },
        {"arm": "B", "faith_deberta": 1.0, "faith_minicheck": 1.0, "rouge1": 0.1, "seconds": 100},
    ]
    agg = {r["arm"]: r for r in aggregate_e1(rows)}
    assert agg["A"]["n_docs"] == 2
    assert agg["A"]["faith_deberta"] == 0.5
    assert agg["A"]["faith_minicheck"] == 0.5
    assert agg["B"]["seconds"] == 100


# ----------------------------------------------------------------------- E5 ---


def test_e5_items_cover_every_applicable_perturbation_with_its_expectation() -> None:
    items = e5_items("d", ["The model is 10 times faster."], per_type=5)
    kinds = {i["type"]: i for i in items}
    assert kinds["negation"]["perturbed"] == "The model is not 10 times faster."
    assert kinds["negation"]["should_support"] is False
    assert kinds["corruption"]["perturbed"] == "The model is 10 times slower."
    assert kinds["assertion"]["should_support"] is False
    assert all(
        i["original"] == "The model is 10 times faster." for i in items if i["type"] != "assertion"
    )


def test_e5_items_skip_claims_a_rule_cannot_change() -> None:
    items = e5_items("d", ["Cats purr."], per_type=5)
    assert {i["type"] for i in items} == {"assertion"}  # no aux verb, no number, no direction word


def test_e5_items_cap_each_type_per_doc() -> None:
    items = e5_items("d", [f"It is {n} ms." for n in range(1, 9)], per_type=3)
    assert sum(i["type"] == "negation" for i in items) == 3
    assert sum(i["type"] == "assertion" for i in items) == 1


def test_aggregate_e5_reports_the_rate_each_verifier_gets_right() -> None:
    rows = [
        {
            "type": "negation",
            "should_support": False,
            "deberta": "contradicted",
            "minicheck": "supported",
        },
        {
            "type": "negation",
            "should_support": False,
            "deberta": "unsupported",
            "minicheck": "unsupported",
        },
        {
            "type": "paraphrase",
            "should_support": True,
            "deberta": "supported",
            "minicheck": "supported",
        },
    ]
    agg = {r["type"]: r for r in aggregate_e5(rows)}
    assert agg["negation"]["n"] == 2
    assert agg["negation"]["deberta_correct"] == 1.0
    assert agg["negation"]["minicheck_correct"] == 0.5
    assert agg["paraphrase"]["deberta_correct"] == 1.0
