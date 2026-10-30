"""Threshold tuning on AggreFact dev (CLAUDE.md M7, SC4). Pure logic; scoring is live-only."""

import math

import pytest

from eval.data import AggreFactItem
from eval.tune import balanced_accuracy, best_theta, stratified_sample


def _item(ds: str, i: int, label: int = 1) -> AggreFactItem:
    return AggreFactItem(dataset=ds, doc=f"doc {ds} {i}", claim=f"claim {i}", label=label)


def test_balanced_accuracy_is_mean_of_class_recalls() -> None:
    # 3 supported (2 right), 1 unsupported (right) -> (2/3 + 1) / 2
    ba = balanced_accuracy([1, 1, 1, 0], [True, True, False, False])
    assert ba == pytest.approx((2 / 3 + 1) / 2)


def test_balanced_accuracy_ignores_class_imbalance() -> None:
    # predicting "supported" always scores 0.5 however skewed the labels are
    assert balanced_accuracy([1] * 9 + [0], [True] * 10) == pytest.approx(0.5)


def test_balanced_accuracy_is_nan_when_a_class_is_missing() -> None:
    assert math.isnan(balanced_accuracy([1, 1], [True, False]))


def test_best_theta_finds_the_separating_threshold() -> None:
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    truth = [0, 0, 0, 1, 1, 1]
    fit = best_theta(scores, truth)
    assert fit.balanced_accuracy == 1.0
    assert 0.3 < fit.theta <= 0.7


def test_best_theta_takes_the_middle_of_a_tied_plateau() -> None:
    # every theta in (0.3, 0.7] is perfect; picking an edge would be fragile
    fit = best_theta([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], [0, 0, 0, 1, 1, 1])
    assert fit.theta == pytest.approx(0.5, abs=0.06)


def test_best_theta_predicts_supported_at_or_above_theta() -> None:
    # a score exactly on theta counts as supported (same rule as verify.labels)
    fit = best_theta([0.5, 0.5, 0.2, 0.2], [1, 1, 0, 0], grid=[0.5])
    assert fit.balanced_accuracy == 1.0


def test_stratified_sample_caps_each_dataset_and_keeps_small_ones() -> None:
    items = [_item("big", i) for i in range(50)] + [_item("small", i) for i in range(3)]
    out = stratified_sample(items, per_dataset=10, seed=0)
    assert sum(i.dataset == "big" for i in out) == 10
    assert sum(i.dataset == "small" for i in out) == 3


def test_stratified_sample_is_deterministic_and_seed_dependent() -> None:
    items = [_item("a", i) for i in range(100)]
    one = stratified_sample(items, per_dataset=10, seed=1)
    assert one == stratified_sample(items, per_dataset=10, seed=1)
    assert one != stratified_sample(items, per_dataset=10, seed=2)


# ------------------------------------------------------------- score_rows ---

from collections.abc import Iterator  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from app.core.config import load_config  # noqa: E402
from app.verify.nli import LabelIdx  # noqa: E402
from app.verify.pipeline import Openers  # noqa: E402
from eval.tune import score_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DOC = "Cats purr softly. Dogs bark loudly. Birds sing at dawn."


def _embed(texts: list[str]) -> Any:
    return np.array(
        [[float(a in t.lower()) for a in ("cat", "dog", "bird")] for t in texts], dtype=np.float32
    )


class _Nli:
    label_idx = LabelIdx(entail=0, contra=2)

    def __call__(self, premises: list[str], hyps: list[str]) -> Any:
        # entails only when the claim repeats the premise's subject word
        return np.array(
            [
                [0.9, 0.05, 0.05] if h.split()[1] in p.lower() else [0.05, 0.05, 0.9]
                for p, h in zip(premises, hyps, strict=True)
            ],
            dtype=np.float32,
        )


def _mc(chunks: list[str], claims: list[str]) -> Any:
    return np.array([0.8 if claims[0].endswith("purrs.") else 0.2] * len(chunks), np.float32)


def _openers(events: list[str]) -> Openers:
    def opener(name: str, obj: Any) -> Any:
        @contextmanager
        def cm() -> Iterator[Any]:
            events.append(f"open {name}")
            yield obj
            events.append(f"close {name}")

        return cm

    return Openers(
        embedder=opener("embedder", _embed), nli=opener("nli", _Nli()), minicheck=opener("mc", _mc)
    )


def test_score_rows_scores_every_row_in_order_with_all_three_verifiers() -> None:
    items = [
        AggreFactItem(dataset="d1", doc=DOC, claim="The cat purrs.", label=1),
        AggreFactItem(dataset="d2", doc=DOC, claim="The dog barks.", label=0),
    ]
    cfg = load_config(ROOT / "configs").verify.model_copy(update={"top_n_evidence": 1})
    rows = score_rows(items, _openers([]), cfg)
    assert [(r.dataset, r.label) for r in rows] == [("d1", 1), ("d2", 0)]
    assert rows[0].entail == pytest.approx(0.9) and rows[0].minicheck == pytest.approx(0.8)
    assert rows[1].minicheck == pytest.approx(0.2)


def test_score_rows_opens_each_model_once_and_never_two_at_a_time() -> None:
    items = [
        AggreFactItem(dataset="d", doc=DOC, claim=f"The cat purrs {i}.", label=1) for i in range(3)
    ]
    events: list[str] = []
    cfg = load_config(ROOT / "configs").verify
    score_rows(items, _openers(events), cfg)
    assert events == [
        "open embedder",
        "close embedder",
        "open nli",
        "close nli",
        "open mc",
        "close mc",
    ]
