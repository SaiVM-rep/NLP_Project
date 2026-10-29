"""GROBID TEI/XML -> RawSections. PRD FR-1.2, FR-1.5, FR-1.6, §11.4 step 2.

shortcut: only <abstract> and <body> divs feed full_text; <back> divs other than
references (acknowledgements, annexes) are dropped. Revisit if annexes matter.
"""

from __future__ import annotations

from dataclasses import dataclass

from lxml import etree

from app.ingest.spans import RawSection

NS = {"t": "http://www.tei-c.org/ns/1.0"}
MIN_SECTIONS = 2  # §11.4 step 5


class IngestError(ValueError):
    """Source could not be parsed into a Document."""


@dataclass
class ParsedTEI:
    title: str
    sections: list[RawSection]
    references_text: str | None
    structure_ok: bool


def _text(el: etree._Element) -> str:
    return " ".join("".join(el.itertext()).split())


def _paragraphs(div: etree._Element) -> list[str]:
    # Direct <p> children only: formulas, figures and tables are siblings, not paragraphs.
    paras = [t for p in div.findall("t:p", NS) if (t := _text(p))]
    # GROBID sometimes leaves a sentence's opening words loose before the first <p>
    # (seen on the SBERT abstract); they belong to that paragraph.
    head = div.find("t:head", NS)
    tail = head.tail if head is not None else None
    loose = " ".join(f"{div.text or ''} {tail or ''}".split())
    if loose and paras:
        paras[0] = f"{loose} {paras[0]}"
    return paras


def _level(head: etree._Element | None) -> int:
    n = head.get("n") if head is not None else None
    return n.strip(".").count(".") + 1 if n else 0


def parse_tei(xml: bytes) -> ParsedTEI:
    try:
        root = etree.fromstring(
            xml, parser=etree.XMLParser(resolve_entities=False, no_network=True)
        )
    except etree.XMLSyntaxError as e:
        raise IngestError(f"malformed TEI: {e}") from e

    title_el = root.find(".//t:titleStmt/t:title", NS)
    title = _text(title_el) if title_el is not None else ""

    sections: list[RawSection] = []
    abstract = [p for div in root.iterfind(".//t:abstract/t:div", NS) for p in _paragraphs(div)]
    if any(abstract):
        sections.append(RawSection("Abstract", 0, [a for a in abstract if a]))

    headed = 0
    for div in root.iterfind(".//t:body/t:div", NS):
        head = div.find("t:head", NS)
        heading = _text(head) if head is not None else ""
        headed += bool(heading)
        sections.append(RawSection(heading, _level(head), _paragraphs(div)))

    refs = [_text(b) for b in root.iterfind(".//t:listBibl/t:biblStruct", NS)]
    return ParsedTEI(
        title=title,
        sections=sections,
        references_text="\n".join(refs) if refs else None,
        structure_ok=headed >= MIN_SECTIONS,
    )
