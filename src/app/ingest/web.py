"""URL -> trafilatura XML -> Document. PRD FR-1.1, FR-1.3, §11.4 web path.

shortcut: tables, quotes and code blocks in trafilatura's output are dropped; only
<p> and <list> feed full_text. Revisit if web reference set loses content.
"""

from __future__ import annotations

import hashlib
import re

import httpx
import trafilatura
from lxml import etree

from app.core.schemas import Document
from app.ingest.spans import RawSection, build_document
from app.ingest.tei import MIN_SECTIONS, IngestError, ParsedTEI

REFERENCE_HEADINGS = {
    "references",
    "bibliography",
    "works cited",
    "sources",
    "notes and references",
}
# Blog chrome trafilatura keeps as content (simonwillison.net: "More recent articles").
NON_CONTENT_HEADINGS = {
    "more recent articles",
    "recent articles",
    "recent posts",
    "related posts",
    "related articles",
    "you may also like",
    "read next",
}
HEADING_TAG = re.compile(r"h[1-6]")

# (level, text): level 0 is a paragraph, 1-6 a heading.
Block = tuple[int, str]


def _text(el: etree._Element) -> str:
    return " ".join("".join(el.itertext()).split())


def _heading(el: etree._Element) -> str:
    # Sphinx and many blogs append a permalink glyph to headings.
    return _text(el).rstrip("¶#§ ").strip()


def _level(head: etree._Element) -> int:
    rend = head.get("rend", "")
    return int(rend[1]) if len(rend) == 2 and rend[0] == "h" and rend[1].isdigit() else 1


def parse_web(html: str) -> ParsedTEI:
    xml = trafilatura.extract(
        html, output_format="xml", include_comments=False, include_tables=False, with_metadata=True
    )
    if not xml:
        raise IngestError("no article text found in page")
    root = etree.fromstring(
        xml.encode(), parser=etree.XMLParser(resolve_entities=False, no_network=True)
    )

    main = root.find("main")
    blocks: list[Block] = []
    for el in main if main is not None else []:
        if el.tag == "head":
            blocks.append((_level(el), _heading(el)))
        elif el.tag == "p":
            blocks.append((0, _text(el)))
        elif el.tag == "list":
            blocks.append((0, "\n".join(t for item in el.iter("item") if (t := _text(item)))))
    blocks = [b for b in blocks if b[1]]
    if not any(level for level, _ in blocks):
        blocks = recover_headings(html, blocks)

    sections: list[RawSection] = []
    for level, text in blocks:
        if level:
            sections.append(RawSection(text, level))
            continue
        if not sections:
            sections.append(RawSection("", 0))
        sections[-1].paragraphs.append(text)
    sections = [s for s in sections if s.heading.lower() not in NON_CONTENT_HEADINGS]

    # FR-1.6: reference lists stay out of full_text, as on the PDF path.
    refs = [s for s in sections if s.heading.lower() in REFERENCE_HEADINGS]
    sections = [s for s in sections if s.heading.lower() not in REFERENCE_HEADINGS]
    ref_lines = [p for s in refs for p in s.paragraphs]

    if not any(s.paragraphs for s in sections):
        raise IngestError("no article text found in page")
    return ParsedTEI(
        title=root.get("title", "").strip(),
        sections=sections,
        references_text="\n".join(ref_lines) if ref_lines else None,
        structure_ok=sum(bool(s.heading) for s in sections) >= MIN_SECTIONS,
    )


def recover_headings(html: str, blocks: list[Block]) -> list[Block]:
    """Put the page's own <h1>-<h6> back between trafilatura's paragraphs.

    trafilatura drops every heading on some pages (distill.pub, Project Gutenberg).
    Headings and paragraphs are placed by their offset in the page's raw text; only
    headings between the first and last located paragraph count, so site headers
    and footers stay out.
    """
    # encoding pinned: we encode the str ourselves; lxml would otherwise guess latin-1.
    parser = etree.HTMLParser(encoding="utf-8", no_network=True)
    page = etree.fromstring(html.encode(), parser=parser)
    if page is None:
        return blocks
    raw: list[str] = []
    size = 0
    heads: list[tuple[int, int, str]] = []
    for event, el in etree.iterwalk(page, events=("start", "end")):
        if not isinstance(el.tag, str):  # comments, processing instructions
            text = el.tail if event == "end" else None
        elif event == "start":
            if HEADING_TAG.fullmatch(el.tag) and (heading := _heading(el)):
                heads.append((size, int(el.tag[1]), heading))
            text = el.text
        else:
            text = el.tail
        if text:
            raw.append(text)
            size += len(text)
    full = "".join(raw)

    placed: list[tuple[int, int, str]] = []
    found: list[int] = []
    cursor = 0
    for level, text in blocks:
        words = text.split()[:6]
        m = re.compile(r"\s+".join(map(re.escape, words))).search(full, cursor)
        if m:
            found.append(m.start())
            cursor = m.end()
        # Unlocated paragraphs (text rewritten by trafilatura) stay with the one before.
        placed.append((found[-1] if found else 0, level, text))
    if not found:
        return blocks
    placed += [h for h in heads if found[0] < h[0] < found[-1]]
    placed.sort(key=lambda b: (b[0], b[1] == 0))
    return [(level, text) for _, level, text in placed]


def ingest_url(url: str, client: httpx.Client, max_mb: float = 25) -> Document:
    if not url.lower().startswith(("http://", "https://")):
        raise IngestError(f"{url!r} is not an http(s) URL")
    try:
        resp = client.get(url, follow_redirects=True)
    except httpx.TransportError as e:
        raise IngestError(f"{url} unreachable ({e})") from e
    if resp.status_code != 200:
        raise IngestError(f"{url} returned HTTP {resp.status_code}")
    ctype = resp.headers.get("content-type", "text/html").split(";")[0].strip()
    if ctype not in ("text/html", "application/xhtml+xml"):
        raise IngestError(f"{url} is {ctype}, not an HTML page")
    if len(resp.content) > max_mb * 1024 * 1024:
        raise IngestError(f"{url} is {len(resp.content) / 1e6:.1f} MB, limit is {max_mb} MB")

    parsed = parse_web(resp.text)
    return build_document(
        doc_id=hashlib.sha256(resp.content).hexdigest()[:16],
        source="web",
        title=parsed.title or url,
        sections=parsed.sections,
        references_text=parsed.references_text,
        structure_ok=parsed.structure_ok,
    )
