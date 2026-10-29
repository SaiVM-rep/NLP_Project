"""Live MiniCheck-Flan-T5-Large on the GPU (FR-5.3). Downloads ~3.1 GB on first run.

Fidelity check for the reimplementation (DECISION-13): reproduce the probabilities
printed in upstream's README (Liyan06/MiniCheck@b58b9fa, fp32) within bf16 tolerance.
"""

from __future__ import annotations

import pytest
from dotenv import load_dotenv

torch = pytest.importorskip("torch")

from app.core.config import load_config  # noqa: E402
from app.core.schemas import Document, Span  # noqa: E402
from app.verify.minicheck import MiniCheckT5, chunk_spans, support_score  # noqa: E402

load_dotenv()  # HF_HOME: weights stay out of the repo

CFG = load_config("configs").verify.minicheck  # pinned checkpoint, CLAUDE.md §8

DOC = "A group of students gather in the school library to study for their upcoming final exams."
UPSTREAM = {
    "The students are preparing for an examination.": 0.9805923700332642,
    "The students are on vacation.": 0.007121330592781305,
}

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.network,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


def test_matches_upstream_readme_and_frees_vram() -> None:
    doc = Document(
        doc_id="d", source="pdf", title="t", full_text=DOC, sentences=[Span(start=0, end=len(DOC))]
    )
    chunks = [DOC[s.start : s.end] for s in chunk_spans(doc)]
    base = torch.cuda.memory_allocated()
    with MiniCheckT5.from_pretrained(CFG.checkpoint, revision=CFG.revision, device="cuda") as mc:
        assert torch.cuda.memory_allocated() - base > 1000 * 2**20
        for claim, want in UPSTREAM.items():
            assert support_score(claim, chunks, mc) == pytest.approx(want, abs=0.02)
    assert torch.cuda.memory_allocated() - base < 50 * 2**20
