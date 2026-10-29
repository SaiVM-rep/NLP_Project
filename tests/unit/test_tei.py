"""FR-1.2 / FR-1.5 / FR-1.6: GROBID TEI -> title, abstract, sections, references."""

from __future__ import annotations

import pytest

from app.ingest.tei import IngestError, parse_tei

TEI = b"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc><titleStmt>
      <title level="a" type="main">Compressing   Long Text</title>
    </titleStmt></fileDesc>
    <profileDesc>
      <abstract><div><p>We compress <hi>long</hi>
        documents.</p></div></abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div><head n="1">Introduction</head>
        <p>Intro text <ref type="bibr" target="#b0">[1]</ref>.</p>
        <formula>E = mc^2</formula>
        <p>Second para.</p>
      </div>
      <div><head n="2.1">Setup</head><p>Setup text.</p></div>
      <figure><head>Figure 1</head><figDesc>A caption.</figDesc></figure>
    </body>
    <back>
      <div type="references"><listBibl>
        <biblStruct xml:id="b0"><analytic><title>Prior Work</title></analytic>
          <monogr><imprint><date when="2020"/></imprint></monogr></biblStruct>
        <biblStruct xml:id="b1"><analytic><title>Other Work</title></analytic></biblStruct>
      </listBibl></div>
    </back>
  </text>
</TEI>"""


def test_title_is_whitespace_normalised() -> None:
    assert parse_tei(TEI).title == "Compressing Long Text"


def test_abstract_becomes_first_section() -> None:
    first = parse_tei(TEI).sections[0]
    assert (first.heading, first.level, first.paragraphs) == (
        "Abstract",
        0,
        ["We compress long documents."],
    )


def test_body_divs_become_sections_with_levels_from_numbering() -> None:
    body = parse_tei(TEI).sections[1:]
    assert [(s.heading, s.level) for s in body] == [("Introduction", 1), ("Setup", 2)]


def test_paragraphs_keep_inline_text_and_skip_formulas_and_figures() -> None:
    intro, setup = parse_tei(TEI).sections[1:]
    assert intro.paragraphs == ["Intro text [1].", "Second para."]
    assert setup.paragraphs == ["Setup text."]


def test_references_go_to_metadata_one_per_line() -> None:
    parsed = parse_tei(TEI)
    assert parsed.references_text is not None
    lines = parsed.references_text.splitlines()
    assert len(lines) == 2 and "Prior Work" in lines[0]
    assert all("Prior Work" not in p for s in parsed.sections for p in s.paragraphs)


def test_two_headed_body_divs_is_structure_ok() -> None:
    assert parse_tei(TEI).structure_ok is True


def test_flat_body_is_not_structure_ok() -> None:
    flat = b"""<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader/><text><body>
      <div><p>Only text, no headings.</p></div></body></text></TEI>"""
    parsed = parse_tei(flat)
    assert parsed.structure_ok is False
    assert parsed.sections[0].heading == "" and parsed.references_text is None


def test_malformed_xml_raises_ingest_error() -> None:
    with pytest.raises(IngestError):
        parse_tei(b"<TEI><unclosed>")


def test_loose_text_before_first_abstract_paragraph_is_kept() -> None:
    # Real GROBID 0.8.1 output on the SBERT paper: the sentence starts outside <p>.
    tei = TEI.replace(
        b"<abstract><div><p>We compress <hi>long</hi>\n        documents.</p></div></abstract>",
        b"<abstract><div> BERT (Devlin et al., 2018) <p>and RoBERTa set a record.</p>"
        b"<p>Second.</p></div></abstract>",
    )
    assert parse_tei(tei).sections[0].paragraphs == [
        "BERT (Devlin et al., 2018) and RoBERTa set a record.",
        "Second.",
    ]
