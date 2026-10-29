"""Span builder. PRD FR-1.4, §11.4 step 3-4.

Parsers (TEI, web) hand over plain section/paragraph strings; this module is the
only place `full_text` and its spans are built, so SC2 holds for every source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cache

import pysbd

from app.core.schemas import Document, Section, Span

log = logging.getLogger(__name__)

SEP = "\n\n"


@dataclass
class RawSection:
    heading: str  # "" when the parser found text with no heading
    level: int
    paragraphs: list[str] = field(default_factory=list)


@cache
def _segmenter() -> pysbd.Segmenter:
    return pysbd.Segmenter(language="en", clean=False)


def sentence_spans(text: str, offset: int) -> list[Span]:
    spans: list[Span] = []
    cursor = 0
    for raw in _segmenter().segment(text):
        sent = raw.strip()
        if not sent:
            continue
        idx = text.find(sent, cursor)
        if idx < 0:
            # pySBD changed the text; keep the rest as one sentence rather than lose it.
            rest = text[cursor:].strip()
            if rest:
                start = text.index(rest, cursor)
                spans.append(Span(start=offset + start, end=offset + start + len(rest)))
            break
        spans.append(Span(start=offset + idx, end=offset + idx + len(sent)))
        cursor = idx + len(sent)
    return spans


def build_document(
    doc_id: str,
    source: str,
    title: str,
    sections: list[RawSection],
    references_text: str | None = None,
    structure_ok: bool = True,
) -> Document:
    parts: list[str] = []
    pos = 0

    def append(text: str) -> Span:
        nonlocal pos
        if parts:
            parts.append(SEP)
            pos += len(SEP)
        parts.append(text)
        span = Span(start=pos, end=pos + len(text))
        pos = span.end
        return span

    out_sections: list[Section] = []
    sentences: list[Span] = []
    for raw in sections:
        paras = [p.strip() for p in raw.paragraphs if p.strip()]
        heading = raw.heading.strip()
        if not paras and not heading:
            continue
        head_span = append(heading) if heading else None
        para_spans = [append(p) for p in paras]
        for p, span in zip(paras, para_spans, strict=True):
            sentences.extend(sentence_spans(p, span.start))
        first = head_span or para_spans[0]
        last = para_spans[-1] if para_spans else first
        out_sections.append(
            Section(
                id=f"s{len(out_sections)}",
                heading=heading,
                level=raw.level,
                span=Span(start=first.start, end=last.end),
                paragraphs=para_spans,
            )
        )

    if not structure_ok:
        log.warning("structure not recovered", extra={"structure_ok": False, "source": source})

    return Document(
        doc_id=doc_id,
        source=source,
        title=title,
        full_text="".join(parts),
        sections=out_sections,
        sentences=sentences,
        references_text=references_text,
        structure_ok=structure_ok,
    )
