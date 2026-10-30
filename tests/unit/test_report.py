"""SC5: every chapter-5 table comes from results/ logs, never typed by hand."""

import math
from pathlib import Path

from eval.report import e1_table, e3_levels, e5_table, latest_run, md_table, verifier_table


def test_md_table_formats_floats_and_nan() -> None:
    out = md_table(["a", "b"], [["x", 0.12345], ["y", math.nan]])
    lines = out.splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| x | 0.123 |"
    assert lines[3] == "| y | n/a |"


def test_latest_run_is_the_newest_finished_run(tmp_path: Path) -> None:
    for name in ("20261031T010000-aaaaaa", "20261031T090000-bbbbbb", "_running"):
        (tmp_path / name).mkdir()
    assert latest_run(tmp_path) == tmp_path / "20261031T090000-bbbbbb"


def test_latest_run_none_when_nothing_finished(tmp_path: Path) -> None:
    (tmp_path / "_running").mkdir()
    assert latest_run(tmp_path) is None
    assert latest_run(tmp_path / "missing") is None


ROWS = [
    {
        "arm": "A",
        "doc_id": "p",
        "faith_deberta": 0.8,
        "faith_minicheck": 0.9,
        "rouge1": 0.4,
        "rouge2": 0.1,
        "rougeL": 0.2,
        "seconds": 100.0,
        "levels_deberta": {"0": 0.8},
        "levels_minicheck": {"0": 0.9},
    },
    {
        "arm": "B",
        "doc_id": "p",
        "faith_deberta": 0.6,
        "faith_minicheck": 0.7,
        "rouge1": 0.3,
        "rouge2": 0.1,
        "rougeL": 0.2,
        "seconds": 900.0,
        "levels_deberta": {"0": 0.7, "1": 0.6},
        "levels_minicheck": {"0": 0.8, "1": 0.7},
    },
    {
        "arm": "B",
        "doc_id": "q",
        "faith_deberta": 0.4,
        "faith_minicheck": 0.5,
        "rouge1": 0.3,
        "rouge2": 0.1,
        "rougeL": 0.2,
        "seconds": 700.0,
        "levels_deberta": {"0": 0.5, "1": 0.4, "2": 0.4},
        "levels_minicheck": {"0": 0.6, "1": 0.5, "2": 0.5},
    },
]


def test_e1_table_has_one_row_per_arm_with_means() -> None:
    out = e1_table(ROWS)
    assert "| A | 1 | 0.800 | 0.900 |" in out
    assert "| B | 2 | 0.500 | 0.600 |" in out


def test_e3_levels_averages_route_b_docs_per_level() -> None:
    lv = e3_levels(ROWS)
    assert lv["deberta"] == {0: (0.6, 2), 1: (0.5, 2), 2: (0.4, 1)}
    assert lv["minicheck"][0] == (0.7, 2)


def test_e5_table_rows_per_perturbation_type() -> None:
    out = e5_table(
        [
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
        ]
    )
    assert "| negation | 2 | 1.000 | 0.500 |" in out


def test_verifier_table_shows_tuned_theta_and_gain_over_placeholder() -> None:
    out = verifier_table(
        {
            "n_rows": 100,
            "nli": {"theta_e": 0.3, "balanced_accuracy": 0.7, "balanced_accuracy_at_0.5": 0.6},
            "minicheck": {
                "theta_e": 0.6,
                "balanced_accuracy": 0.75,
                "balanced_accuracy_at_0.5": 0.74,
            },
        }
    )
    assert "| DeBERTa | 0.300 | 0.600 | 0.700 |" in out
    assert "| MiniCheck | 0.600 | 0.740 | 0.750 |" in out
