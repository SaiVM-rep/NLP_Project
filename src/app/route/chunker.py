"""Route B chunking. PRD FR-3.2, §11.5 steps 1-4.

Units are paragraphs (a paragraph over budget falls back to its sentences; a sentence
over budget stays whole and is flagged `oversized`). Units are packed greedily up to
`chunk_tokens`; each later piece is prefixed with whole trailing sentences of the
previous piece, up to `overlap_tokens`.

`chunk_sections` never lets a chunk cross a section boundary. `chunk_uniform` (baseline
B2) runs the *same* packer with the same budget and overlap but ignores sections, so E4
isolates exactly one variable. B2 cuts at sentence boundaries, not mid-sentence.

Chunks are spans over `full_text` (SC2): chunk text is always `full_text[span]`.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from app.core.config import ChunkCfg
from app.core.schemas import Document, Section, Span

Count = Callable[[str], int]


class Chunk(BaseModel):
    id: str
    index: int
    section_ids: list[str]  # exactly one for section-aware chunks
    span: Span  # includes the overlap
    overlap: Span | None = None  # leading text repeated from the previous chunk
    tokens: int
    oversized: bool = False  # a single unit (sentence/heading) larger than the budget


def _units(doc: Document, sec: Section, budget: int, count: Count) -> list[Span]:
    units: list[Span] = []
    if sec.heading:
        units.append(Span(start=sec.span.start, end=sec.span.start + len(sec.heading)))
    for p in sec.paragraphs:
        if count(doc.text_of(p)) <= budget:
            units.append(p)
            continue
        sents = [s for s in doc.sentences if p.start <= s.start and s.end <= p.end]
        units.extend(sents or [p])
    return units


def _pack(doc: Document, units: list[Span], budget: int, count: Count) -> list[list[Span]]:
    pieces: list[list[Span]] = []
    cur: list[Span] = []
    for u in units:
        if cur and count(doc.full_text[cur[0].start : u.end]) > budget:
            pieces.append(cur)
            cur = []
        cur.append(u)
    if cur:
        pieces.append(cur)
    return pieces


def _overlap(doc: Document, prev: Span, limit: int, count: Count) -> Span | None:
    """Longest run of whole trailing sentences of `prev` within `limit` tokens."""
    start = None
    inside = [s for s in doc.sentences if prev.start <= s.start and s.end <= prev.end]
    for s in reversed(inside):
        if count(doc.full_text[s.start : prev.end]) > limit:
            break
        start = s.start
    return Span(start=start, end=prev.end) if start is not None else None


def _chunks(doc: Document, groups: list[list[Span]], cfg: ChunkCfg, count: Count) -> list[Chunk]:
    """`groups`: unit lists that pieces must not span (one per section, or one overall)."""
    out: list[Chunk] = []
    for units in groups:
        prev: Span | None = None
        for piece in _pack(doc, units, cfg.chunk_tokens, count):
            new = Span(start=piece[0].start, end=piece[-1].end)
            ov = _overlap(doc, prev, cfg.overlap_tokens, count) if prev else None
            span = Span(start=ov.start, end=new.end) if ov else new
            out.append(
                Chunk(
                    id=f"c{len(out)}",
                    index=len(out),
                    section_ids=[
                        s.id
                        for s in doc.sections
                        if s.span.start < span.end and span.start < s.span.end
                    ],
                    span=span,
                    overlap=ov,
                    tokens=count(doc.text_of(span)),
                    oversized=len(piece) == 1 and count(doc.text_of(new)) > cfg.chunk_tokens,
                )
            )
            prev = new
    return out


def chunk_sections(doc: Document, cfg: ChunkCfg, count: Count) -> list[Chunk]:
    groups = [_units(doc, s, cfg.chunk_tokens, count) for s in doc.sections]
    return _chunks(doc, groups, cfg, count)


def chunk_uniform(doc: Document, cfg: ChunkCfg, count: Count) -> list[Chunk]:
    units = [u for s in doc.sections for u in _units(doc, s, cfg.chunk_tokens, count)]
    return _chunks(doc, [units] if units else [], cfg, count)
