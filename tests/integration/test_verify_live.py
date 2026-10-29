"""End-to-end verify stage on a real paper with the real, config-pinned models (PRD §11.8).

Claims are fixed (no LLM) so the test pins the verifiers, not the decomposer.
Document is the SBERT paper from stored GROBID TEI, so no docker is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

torch = pytest.importorskip("torch")

from app.core.config import load_config  # noqa: E402
from app.core.schemas import ClaimLabel  # noqa: E402
from app.ingest.spans import build_document  # noqa: E402
from app.ingest.tei import parse_tei  # noqa: E402
from app.verify.decompose import Claim  # noqa: E402
from app.verify.pipeline import openers_from_config, verify_claims  # noqa: E402

load_dotenv()  # HF_HOME
ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.network,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

S = ClaimLabel.SUPPORTED
C = ClaimLabel.CONTRADICTED
U = ClaimLabel.UNSUPPORTED

# claim -> (DeBERTa label, MiniCheck label), checked against the SBERT paper by hand
CASES = {
    "SBERT uses siamese and triplet network structures.": (S, S),
    "SBERT finds the most similar pair in about 5 seconds.": (S, S),
    # "SBERT was pre-trained only on Wikipedia (via BERT) and on NLI data." -> contradicted
    "SBERT was trained on ImageNet photographs.": (C, U),
}
# Role swap: 65 h is BERT's time, SBERT's is 5 s. DeBERTa catches it (contra 0.998).
# MiniCheck scored it 0.653 = supported on 2026-10-29: a real false positive, measured in
# E5 / the perturbation suite, deliberately NOT pinned here.
ROLE_SWAP = "SBERT takes about 65 hours to find the most similar pair."


def test_labels_on_real_paper_and_gpu_left_clean() -> None:
    tei = parse_tei((ROOT / "tests/fixtures/tei/sbert.tei.xml").read_bytes())
    doc = build_document("sbert", "pdf", tei.title, tei.sections, tei.references_text)
    cfg = load_config(ROOT / "configs").verify
    claims = [("n0", Claim(text=t, sentence_idx=i)) for i, t in enumerate([*CASES, ROLE_SWAP])]

    base = torch.cuda.memory_allocated()
    out = verify_claims(doc, claims, openers_from_config(cfg), cfg)
    assert torch.cuda.memory_allocated() - base < 50 * 2**20  # all three closed

    got = {r.claim: (r.label_deberta, r.label_minicheck) for r in out}
    assert got[ROLE_SWAP][0] == C
    del got[ROLE_SWAP]
    assert got == CASES
    for r in out:  # evidence is real offsets into the paper
        assert all(doc.full_text[s.start : s.end] for s in r.evidence)
        assert len(r.evidence) == cfg.top_n_evidence
