"""Live BERTScore with the pinned roberta-large (PRD §12.5). ~1.4 GB download."""

from __future__ import annotations

import pytest
from dotenv import load_dotenv

torch = pytest.importorskip("torch")

from app.core.config import load_config  # noqa: E402
from eval.bertscore import BertScore  # noqa: E402

load_dotenv()  # HF_HOME
CFG = load_config("configs").metrics.bertscore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.network,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


def test_pinned_bertscore_ranks_and_flags_truncation() -> None:
    ref = "The cat sat on the mat."
    long = " ".join(["The model summarises long scientific documents."] * 100)  # ~700 words
    base = torch.cuda.memory_allocated()
    with BertScore.from_config(CFG) as bs:
        same, para, other = bs(
            [ref, "A cat was sitting on a mat.", "Stock prices fell sharply on Monday."],
            [ref, ref, ref],
        )
        assert bs.truncated == 0
        assert same == pytest.approx(1.0, abs=1e-4)
        assert para > other + 0.2
        bs([long], [ref])
        assert bs.truncated == 1
    assert torch.cuda.memory_allocated() - base < 50 * 2**20
