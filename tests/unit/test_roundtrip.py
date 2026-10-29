"""SC2 / FR-1.4 on real papers: every span reproduces the parsed text exactly.

Runs offline on GROBID TEI stored in tests/fixtures/tei (regenerate with
`python scripts/make_fixtures.py`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.schemas import Document
from app.ingest.spans import build_document
from app.ingest.tei import ParsedTEI, parse_tei

TEI_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "tei"
TEI_FILES = sorted(TEI_DIR.glob("*.tei.xml"))


def test_five_fixture_papers_exist() -> None:
    assert len(TEI_FILES) == 5, f"expected 5 TEI fixtures in {TEI_DIR}"


def _load(path: Path) -> tuple[ParsedTEI, Document]:
    parsed = parse_tei(path.read_bytes())
    doc = build_document(path.name, "pdf", parsed.title, parsed.sections, parsed.references_text)
    return parsed, doc


@pytest.mark.parametrize("path", TEI_FILES, ids=lambda p: p.name.split(".")[0])
def test_every_heading_and_paragraph_round_trips(path: Path) -> None:
    parsed, doc = _load(path)
    raw = [
        (s.heading.strip(), [p.strip() for p in s.paragraphs if p.strip()]) for s in parsed.sections
    ]
    raw = [(h, ps) for h, ps in raw if h or ps]
    assert len(doc.sections) == len(raw)
    for sec, (heading, paras) in zip(doc.sections, raw, strict=True):
        assert sec.heading == heading
        assert doc.text_of(sec.span).startswith(heading)
        assert [doc.text_of(p) for p in sec.paragraphs] == paras


@pytest.mark.parametrize("path", TEI_FILES, ids=lambda p: p.name.split(".")[0])
def test_spans_nest_and_never_cross_section_boundaries(path: Path) -> None:
    _, doc = _load(path)
    paras = [p for s in doc.sections for p in s.paragraphs]
    for a, b in zip(doc.sections, doc.sections[1:], strict=False):
        assert a.span.end <= b.span.start
    for sec in doc.sections:
        assert all(sec.span.start <= p.start and p.end <= sec.span.end for p in sec.paragraphs)
    for sent in doc.sentences:
        text = doc.text_of(sent)
        assert text and text == text.strip()
        assert sum(p.start <= sent.start and sent.end <= p.end for p in paras) == 1


@pytest.mark.parametrize("path", TEI_FILES, ids=lambda p: p.name.split(".")[0])
def test_references_kept_out_of_full_text(path: Path) -> None:
    _, doc = _load(path)
    assert doc.references_text, "every fixture paper has a bibliography"
    first_ref = doc.references_text.splitlines()[0]
    assert first_ref not in doc.full_text
