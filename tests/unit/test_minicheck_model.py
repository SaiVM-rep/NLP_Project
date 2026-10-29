"""MiniCheckT5 wrapper on a tiny random-init T5: no download, no GPU (DECISION-9)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from app.verify.minicheck import MiniCheckT5  # noqa: E402

VOCAB = 256  # label tokens 3 and 209 must exist


class StubTokenizer:
    eos_token = "</s>"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(
        self,
        texts: list[str],
        *,
        max_length: int,
        truncation: bool,
        padding: bool,
        return_tensors: str,
    ) -> dict[str, Any]:
        assert truncation and padding and return_tensors == "pt"
        self.seen.extend(texts)
        rows = [[ord(c) % VOCAB for c in t][:max_length] for t in texts]
        width = max(len(r) for r in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros(len(rows), width, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
            mask[i, : len(r)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def tiny_model() -> Any:
    torch.manual_seed(0)
    cfg = transformers.T5Config(
        vocab_size=VOCAB,
        d_model=16,
        d_kv=8,
        d_ff=32,
        num_layers=1,
        num_heads=2,
        decoder_start_token_id=0,
        pad_token_id=0,
        # 2.0 spreads outputs (~0.80-0.92); >= 5.0 saturates every pair to 1.0
        initializer_factor=2.0,
    )
    return transformers.T5ForConditionalGeneration(cfg)


CHUNKS = ["The cat sat on the mat.", "Dogs bark.", "Rain fell all day long in the city."]
CLAIMS = ["A cat sat."] * 3


def test_input_format_matches_upstream() -> None:
    tok = StubTokenizer()
    MiniCheckT5(tiny_model(), tok, device="cpu")(["the doc"], ["the claim"])
    assert tok.seen == ["predict: the doc</s>the claim"]


def test_returns_support_probability_per_pair() -> None:
    p = MiniCheckT5(tiny_model(), StubTokenizer(), device="cpu")(CHUNKS, CLAIMS)
    assert p.shape == (3,) and p.dtype == np.float32
    assert ((p >= 0) & (p <= 1)).all()
    assert len(set(p.round(6).tolist())) > 1  # not a constant


def test_support_is_softmax_over_label_tokens_3_and_209() -> None:
    m = tiny_model()
    nli = MiniCheckT5(m, StubTokenizer(), device="cpu")
    with torch.no_grad():
        enc = StubTokenizer()(
            ["predict: x</s>y"], max_length=2048, truncation=True, padding=True, return_tensors="pt"
        )
        logits = m(**enc, decoder_input_ids=torch.zeros((1, 1), dtype=torch.long)).logits[:, 0]
        want = torch.softmax(logits[:, [3, 209]], dim=-1)[:, 1].item()
    assert nli(["x"], ["y"])[0] == pytest.approx(want, abs=1e-6)


def test_batching_does_not_change_scores() -> None:
    m = tiny_model()
    whole = MiniCheckT5(m, StubTokenizer(), device="cpu", batch_size=8)(CHUNKS, CLAIMS)
    split = MiniCheckT5(m, StubTokenizer(), device="cpu", batch_size=1)(CHUNKS, CLAIMS)
    np.testing.assert_allclose(whole, split, atol=1e-5)


def test_eval_mode_is_deterministic() -> None:
    m = tiny_model()
    m.train()
    mc = MiniCheckT5(m, StubTokenizer(), device="cpu")
    np.testing.assert_array_equal(mc(CHUNKS, CLAIMS), mc(CHUNKS, CLAIMS))


def test_empty_input() -> None:
    assert MiniCheckT5(tiny_model(), StubTokenizer(), device="cpu")([], []).shape == (0,)


def test_close_blocks_reuse() -> None:
    with MiniCheckT5(tiny_model(), StubTokenizer(), device="cpu") as mc:
        mc(CHUNKS[:1], CLAIMS[:1])
    with pytest.raises(RuntimeError, match="closed"):
        mc(CHUNKS[:1], CLAIMS[:1])


def test_loading_never_triggers_hub_safetensors_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The checkpoint is .bin-only: transformers then starts a background thread that asks
    huggingface.co for (and can request) a safetensors conversion PR, ignoring
    local_files_only. Found by the FR-5.6 network-disabled test, 2026-10-29."""
    import os

    seen: dict[str, str | None] = {}

    def fake_model(*a: Any, **kw: Any) -> Any:
        seen["env"] = os.environ.get("DISABLE_SAFETENSORS_CONVERSION")
        return tiny_model()

    monkeypatch.delenv("DISABLE_SAFETENSORS_CONVERSION", raising=False)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: StubTokenizer()
    )
    monkeypatch.setattr(transformers.AutoModelForSeq2SeqLM, "from_pretrained", fake_model)
    MiniCheckT5.from_pretrained("x", device="cpu", local_files_only=True)
    assert seen["env"] in ("1", "true")
