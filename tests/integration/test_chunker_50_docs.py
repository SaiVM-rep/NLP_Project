"""FR-3.2 acceptance: chunking property test on 50 real documents.

19 papers with real sections (5 stored TEI fixtures + 14 PDFs parsed by live GROBID)
and 31 arXiv test documents (ccdv; long, unstructured). Real token estimator, default
budget and a small one that forces splits everywhere.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv

from app.core.config import ChunkCfg, load_config
from app.core.schemas import Document
from app.ingest.grobid import GrobidClient, ingest_pdf
from app.ingest.spans import build_document
from app.ingest.tei import parse_tei
from app.route.chunker import chunk_sections, chunk_uniform
from app.route.tokens import estimate_tokens
from eval.data import load_long_docs, to_document
from tests.unit.test_chunker import check_invariants

load_dotenv()
ROOT = Path(__file__).resolve().parents[2]
GROBID = os.getenv("GROBID_URL", "http://localhost:8070")
PDFS = [
    *sorted((ROOT / "data" / "p1_pdfs").glob("*.pdf")),
    ROOT / "NIPS-2017-attention-is-all-you-need-Paper.pdf",
]

pytestmark = [pytest.mark.integration, pytest.mark.network]


@functools.cache
def corpus() -> list[Document]:
    docs = []
    for p in sorted((ROOT / "tests/fixtures/tei").glob("*.tei.xml")):
        t = parse_tei(p.read_bytes())
        docs.append(build_document(p.name, "pdf", t.title, t.sections, t.references_text))
    try:
        httpx.get(f"{GROBID}/api/isalive", timeout=5).raise_for_status()
    except httpx.HTTPError:
        pytest.skip("GROBID not running (make up)")
    with httpx.Client(timeout=300) as http:
        client = GrobidClient(GROBID, client=http)
        docs += [ingest_pdf(p, client) for p in PDFS if p.exists()]
    docs += [to_document(d, "arxiv") for d in load_long_docs("arxiv", limit=50 - len(docs))]
    return docs


def test_corpus_is_fifty_documents_with_real_sections_among_them() -> None:
    docs = corpus()
    assert len(docs) == 50
    assert sum(d.structure_ok for d in docs) >= 19


@pytest.mark.parametrize("chunk,overlap", [(3000, 150), (500, 50)])
def test_invariants_hold_on_every_document(chunk: int, overlap: int) -> None:
    margin = load_config(ROOT / "configs").route.token_margin
    count = functools.partial(estimate_tokens, margin=margin)
    c = ChunkCfg(chunk_tokens=chunk, overlap_tokens=overlap, fan_in_k=4)
    split = 0
    for d in corpus():
        sec = chunk_sections(d, c, count)
        check_invariants(d, sec, c, sectioned=True, count=count)
        check_invariants(d, chunk_uniform(d, c, count), c, sectioned=False, count=count)
        split += len(sec) > len(d.sections)
    assert split > 0  # the budget really forced section splits somewhere
