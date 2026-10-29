"""Live sentence-transformer + evidence retrieval on the GPU (FR-5.2). ~90 MB download."""

from __future__ import annotations

import pytest
from dotenv import load_dotenv

torch = pytest.importorskip("torch")

from app.core.config import load_config  # noqa: E402
from app.core.schemas import Document, Span  # noqa: E402
from app.verify.embedder import SentenceEmbedder  # noqa: E402
from app.verify.evidence import retrieve_evidence  # noqa: E402

load_dotenv()  # HF_HOME: weights stay out of the repo
CFG = load_config("configs").verify.embedder

SENTS = [
    "The Transformer relies entirely on self-attention.",
    "We trained on eight NVIDIA P100 GPUs for twelve hours.",
    "The model reached 28.4 BLEU on WMT 2014 English-German.",
    "Recurrent networks process tokens sequentially.",
]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.network,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


def doc() -> Document:
    text = " ".join(SENTS)
    spans, pos = [], 0
    for s in SENTS:
        spans.append(Span(start=pos, end=pos + len(s)))
        pos += len(s) + 1
    return Document(doc_id="d", source="pdf", title="t", full_text=text, sentences=spans)


def test_paraphrased_claims_retrieve_their_source_sentence() -> None:
    d = doc()
    claims = [
        "Training took half a day on 8 GPUs.",
        "It scored 28.4 BLEU on English to German translation.",
        "The architecture uses only attention, no recurrence.",
    ]
    base = torch.cuda.memory_allocated()
    with SentenceEmbedder.from_pretrained(CFG.checkpoint, revision=CFG.revision) as emb:
        assert emb(["x"]).shape == (1, 384)
        ev = retrieve_evidence(claims, d, emb, n=1)
    got = [d.full_text[e[0].start : e[0].end] for e in ev]
    assert got == [SENTS[1], SENTS[2], SENTS[0]]
    assert torch.cuda.memory_allocated() - base < 50 * 2**20
