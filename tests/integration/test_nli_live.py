"""Live DeBERTa-v3 NLI checkpoint on the GPU (FR-5.3). Downloads ~870 MB on first run.

Pins the [VERIFY] checkpoint in CLAUDE.md §8: label order, behaviour on clear
entail / contradict / neutral pairs, and that close() really frees VRAM (DECISION-3).
"""

from __future__ import annotations

import pytest
from dotenv import load_dotenv

torch = pytest.importorskip("torch")

from app.core.config import load_config  # noqa: E402
from app.verify.nli import DebertaNLI, LabelIdx, score_claim  # noqa: E402

load_dotenv()  # HF_HOME: weights stay out of the repo

CFG = load_config("configs").verify.nli  # pinned checkpoint, CLAUDE.md §8

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.network,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


def test_checkpoint_behaves_and_frees_vram() -> None:
    base = torch.cuda.memory_allocated()
    with DebertaNLI.from_pretrained(CFG.checkpoint, revision=CFG.revision, device="cuda") as nli:
        assert nli.label_idx == LabelIdx(entail=0, contra=2)
        assert torch.cuda.memory_allocated() - base > 500 * 2**20  # weights really on GPU

        entail, _ = score_claim(
            "A person is playing an instrument.",
            ["The sky was clear.", "A man is playing a guitar on stage."],
            nli,
            nli.label_idx,
        )
        assert entail > 0.9

        _, contra = score_claim(
            "The meeting was held on Friday.",
            ["The meeting was held on Monday."],
            nli,
            nli.label_idx,
        )
        assert contra > 0.9

        entail, contra = score_claim(
            "The tower was painted blue in 2020.",
            ["The Eiffel Tower is in Paris."],
            nli,
            nli.label_idx,
        )
        assert entail < 0.5 and contra < 0.5  # neutral -> "unsupported" under label()

    assert torch.cuda.memory_allocated() - base < 50 * 2**20
