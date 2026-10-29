"""SentenceEmbedder: the real `Embed` for evidence retrieval (FR-5.2). Fake model, no download."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.verify.embedder import SentenceEmbedder


class FakeST:
    """Duck-types the bits of SentenceTransformer the wrapper touches."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.training = True

    def to(self, device: str) -> FakeST:
        return self

    def eval(self) -> FakeST:
        self.training = False
        return self

    def get_embedding_dimension(self) -> int:  # sentence-transformers >= 6 name
        return 3

    def encode(self, texts: list[str], **kw: Any) -> Any:
        self.calls.append(kw)
        return np.arange(len(texts) * 3, dtype=np.float64).reshape(len(texts), 3)


def test_returns_float32_matrix_one_row_per_text() -> None:
    v = SentenceEmbedder(FakeST(), device="cpu")(["a", "b"])
    assert v.shape == (2, 3) and v.dtype == np.float32


def test_encode_settings() -> None:
    fake = FakeST()
    SentenceEmbedder(fake, device="cpu", batch_size=7)(["a"])
    kw = fake.calls[0]
    assert kw["batch_size"] == 7
    assert kw["convert_to_numpy"] is True
    assert kw["show_progress_bar"] is False  # progress bars would spam worker logs


def test_model_put_in_eval_mode() -> None:
    fake = FakeST()
    SentenceEmbedder(fake, device="cpu")
    assert fake.training is False


def test_empty_input_keeps_dimension_without_encode() -> None:
    fake = FakeST()
    v = SentenceEmbedder(fake, device="cpu")([])
    assert v.shape == (0, 3)
    assert fake.calls == []


def test_close_blocks_reuse() -> None:
    with SentenceEmbedder(FakeST(), device="cpu") as emb:
        emb(["a"])
    with pytest.raises(RuntimeError, match="closed"):
        emb(["a"])
