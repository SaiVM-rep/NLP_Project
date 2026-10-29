"""DebertaNLI wrapper on a tiny random-init model: no download, no GPU (DECISION-9)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from app.verify.nli import DebertaNLI, LabelIdx  # noqa: E402

VOCAB = 64


class StubTokenizer:
    """Char codes as token ids; enough to drive a real forward pass."""

    def __call__(
        self, a: list[str], b: list[str], *, padding: bool, truncation: bool, return_tensors: str
    ) -> dict[str, Any]:
        assert padding and truncation and return_tensors == "pt"
        rows = [[ord(c) % VOCAB for c in x + "|" + y] for x, y in zip(a, b, strict=True)]
        width = max(len(r) for r in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros(len(rows), width, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
            mask[i, : len(r)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def tiny_model() -> Any:
    torch.manual_seed(0)
    cfg = transformers.DebertaV2Config(
        vocab_size=VOCAB,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=128,
        num_labels=3,
        # default 0.02 gives ~uniform probs, too flat to catch e.g. a dropped attention mask
        initializer_range=1.0,
        id2label={0: "contradiction", 1: "entailment", 2: "neutral"},
        label2id={"contradiction": 0, "entailment": 1, "neutral": 2},
    )
    return transformers.DebertaV2ForSequenceClassification(cfg)


PREM = ["The cat sat.", "Dogs bark loudly at night.", "Rain.", "A b c d e f g."]
HYP = ["A cat sat."] * 4


def test_returns_probabilities_per_pair() -> None:
    nli = DebertaNLI(tiny_model(), StubTokenizer(), device="cpu", batch_size=8)
    p = nli(PREM, HYP)
    assert p.shape == (4, 3) and p.dtype == np.float32
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_label_indices_come_from_model_config() -> None:
    nli = DebertaNLI(tiny_model(), StubTokenizer(), device="cpu")
    assert nli.label_idx == LabelIdx(entail=1, contra=0)


def test_batching_does_not_change_scores() -> None:
    # padding differs per batch; attention mask must make that irrelevant
    m = tiny_model()
    whole = DebertaNLI(m, StubTokenizer(), device="cpu", batch_size=8)(PREM, HYP)
    split = DebertaNLI(m, StubTokenizer(), device="cpu", batch_size=1)(PREM, HYP)
    np.testing.assert_allclose(whole, split, atol=1e-5)


def test_eval_mode_is_deterministic() -> None:
    m = tiny_model()
    m.train()  # dropout on, as a freshly built model would be
    nli = DebertaNLI(m, StubTokenizer(), device="cpu")
    np.testing.assert_array_equal(nli(PREM, HYP), nli(PREM, HYP))


def test_empty_input_returns_empty_without_model_call() -> None:
    nli = DebertaNLI(tiny_model(), StubTokenizer(), device="cpu")
    assert nli([], []).shape == (0, 3)


def test_close_releases_model_and_blocks_reuse() -> None:
    # DECISION-3: 8 GB GPU, verifiers load and unload one at a time
    with DebertaNLI(tiny_model(), StubTokenizer(), device="cpu") as nli:
        nli(PREM[:1], HYP[:1])
    with pytest.raises(RuntimeError, match="closed"):
        nli(PREM[:1], HYP[:1])
