"""FR-1.4 / SC2: full_text stored once, every structure is a span that slices back exactly."""

from __future__ import annotations

import logging

import pytest

from app.ingest.spans import RawSection, build_document

SECTIONS = [
    RawSection("Abstract", 0, ["We study X. It works well."]),
    RawSection("1 Introduction", 1, ["Para one. Dr. Smith et al. agree.", "Para two is here."]),
    RawSection("2 Method", 1, ["We use Y, i.e. a model. Results in Fig. 2 show gains."]),
]


def _doc(sections: list[RawSection] = SECTIONS, structure_ok: bool = True):  # type: ignore[no-untyped-def]
    return build_document("d1", "pdf", "A Title", sections, structure_ok=structure_ok)


def test_full_text_joins_headings_and_paragraphs_with_blank_lines() -> None:
    doc = _doc([RawSection("H1", 1, ["P1", "P2"]), RawSection("H2", 1, ["P3"])])
    assert doc.full_text == "H1\n\nP1\n\nP2\n\nH2\n\nP3"


def test_every_paragraph_span_round_trips() -> None:
    doc = _doc()
    got = [doc.text_of(p) for s in doc.sections for p in s.paragraphs]
    assert got == [p for s in SECTIONS for p in s.paragraphs]


def test_section_span_starts_at_heading_and_covers_its_paragraphs() -> None:
    doc = _doc()
    for raw, sec in zip(SECTIONS, doc.sections, strict=True):
        assert sec.heading == raw.heading
        assert doc.text_of(sec.span).startswith(raw.heading)
        assert sec.span.end == sec.paragraphs[-1].end


def test_sections_do_not_overlap() -> None:
    doc = _doc()
    for a, b in zip(doc.sections, doc.sections[1:], strict=False):
        assert a.span.end <= b.span.start


def test_sentences_round_trip_and_stay_inside_one_paragraph() -> None:
    doc = _doc()
    paras = [p for s in doc.sections for p in s.paragraphs]
    assert doc.sentences
    for sent in doc.sentences:
        text = doc.text_of(sent)
        assert text and text == text.strip()
        assert any(p.start <= sent.start and sent.end <= p.end for p in paras)


def test_sentence_splitter_respects_abbreviations() -> None:
    doc = _doc([RawSection("H", 0, ["Dr. Smith et al. agree. See Fig. 2 for details."])])
    assert [doc.text_of(s) for s in doc.sentences] == [
        "Dr. Smith et al. agree.",
        "See Fig. 2 for details.",
    ]


def test_blank_paragraphs_are_dropped() -> None:
    doc = _doc([RawSection("H", 0, ["  ", "", "Real."])])
    assert [doc.text_of(p) for p in doc.sections[0].paragraphs] == ["Real."]


def test_headless_section_contributes_no_heading_text() -> None:
    doc = _doc([RawSection("", 0, ["Lead paragraph."]), RawSection("H", 1, ["Body."])])
    assert doc.full_text == "Lead paragraph.\n\nH\n\nBody."
    assert doc.sections[0].span.start == 0


def test_references_are_metadata_not_full_text() -> None:
    doc = build_document("d1", "pdf", "T", SECTIONS, references_text="[1] Some Ref. 2020.")
    assert doc.references_text == "[1] Some Ref. 2020."
    assert "Some Ref" not in doc.full_text


def test_structure_not_ok_is_flagged_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        doc = _doc(structure_ok=False)
    assert doc.structure_ok is False
    assert any("structure" in r.getMessage() for r in caplog.records)


def test_section_ids_are_stable_and_unique() -> None:
    ids = [s.id for s in _doc().sections]
    assert ids == ["s0", "s1", "s2"]
