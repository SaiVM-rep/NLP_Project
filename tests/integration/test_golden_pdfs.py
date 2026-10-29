"""Golden test (PRD §15): live GROBID on 5 fixture PDFs vs stored expectations.

Catches GROBID image/version changes (DECISION-10). Expectations live in
tests/fixtures/golden; regenerate with `python scripts/make_fixtures.py` and
review the diff before committing.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from app.ingest.grobid import GrobidClient, ingest_pdf

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PDFS = sorted((FIXTURES / "pdfs").glob("*.pdf"))
GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070")

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def grobid() -> Iterator[GrobidClient]:
    try:
        httpx.get(f"{GROBID_URL}/api/isalive", timeout=5).raise_for_status()
    except httpx.HTTPError:
        pytest.skip(f"GROBID not reachable at {GROBID_URL}; run `make up`")
    with httpx.Client(timeout=300) as http:
        yield GrobidClient(GROBID_URL, client=http)


def test_five_fixture_pdfs_exist() -> None:
    assert len(PDFS) == 5


@pytest.mark.parametrize("pdf", PDFS, ids=lambda p: p.stem)
def test_pdf_matches_golden(pdf: Path, grobid: GrobidClient) -> None:
    golden = json.loads((FIXTURES / "golden" / f"{pdf.stem}.json").read_text(encoding="utf-8"))
    doc = ingest_pdf(pdf, grobid)
    assert doc.title == golden["title"]
    assert doc.structure_ok == golden["structure_ok"]
    assert [s.heading for s in doc.sections] == golden["headings"]
