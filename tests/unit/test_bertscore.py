"""BertScore wrapper (PRD §12.5). Fake scorer: no download, no GPU (DECISION-9)."""

from __future__ import annotations

import logging
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from eval.bertscore import BertScore  # noqa: E402

MAX = 8  # fake model max length, in tokens


class FakeModel:
    def to(self, device: str) -> FakeModel:
        return self

    def eval(self) -> FakeModel:
        return self


class FakeTokenizer:
    model_max_length = MAX

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [0] * (len(text.split()) + (2 if add_special_tokens else 0))


class FakeScorer:
    def __init__(self) -> None:
        self._model = FakeModel()
        self._tokenizer = FakeTokenizer()
        self.calls: list[tuple[list[str], list[str]]] = []

    def score(self, cands: list[str], refs: list[str]) -> tuple[Any, Any, Any]:
        self.calls.append((cands, refs))
        f = torch.tensor([0.5 + 0.1 * i for i in range(len(cands))])
        return f - 0.2, f + 0.2, f


def test_returns_f1_per_pair_as_floats() -> None:
    f1 = BertScore(FakeScorer(), device="cpu")(["a", "b"], ["x", "y"])
    assert f1 == pytest.approx([0.5, 0.6])
    assert all(type(v) is float for v in f1)


def test_candidate_and_reference_order_kept() -> None:
    s = FakeScorer()
    BertScore(s, device="cpu")(["cand"], ["ref"])
    assert s.calls == [(["cand"], ["ref"])]


def test_empty_input_no_call() -> None:
    s = FakeScorer()
    assert BertScore(s, device="cpu")([], []) == []
    assert s.calls == []


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        BertScore(FakeScorer(), device="cpu")(["a"], ["x", "y"])


def test_truncation_is_counted_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    """bert-score cuts inputs at the model max (512 for RoBERTa) without telling anyone."""
    bs = BertScore(FakeScorer(), device="cpu")
    long = " ".join(["w"] * 10)  # 10 + 2 special > 8
    with caplog.at_level(logging.WARNING):
        bs(["short", long], [long, "short"])
    assert bs.truncated == 2  # one candidate, one reference
    assert any("truncat" in r.getMessage() for r in caplog.records)


def test_no_warning_when_everything_fits(caplog: pytest.LogCaptureFixture) -> None:
    bs = BertScore(FakeScorer(), device="cpu")
    with caplog.at_level(logging.WARNING):
        bs(["a b"], ["c d"])
    assert bs.truncated == 0
    assert caplog.records == []


def test_close_blocks_reuse() -> None:
    with BertScore(FakeScorer(), device="cpu") as bs:
        bs(["a"], ["b"])
    with pytest.raises(RuntimeError, match="closed"):
        bs(["a"], ["b"])
