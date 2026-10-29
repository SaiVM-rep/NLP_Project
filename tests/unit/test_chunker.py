"""FR-3.2 section-aware chunking + B2 uniform baseline. PRD §11.5 steps 1-4.

Token counter is injected; tests count whitespace words so sizes are easy to read.
"""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.core.config import ChunkCfg
from app.core.schemas import Document
from app.ingest.spans import RawSection, build_document
from app.route.chunker import Chunk, Count, chunk_sections, chunk_uniform


def words(text: str) -> int:
    return len(text.split())


def cfg(chunk: int, overlap: int) -> ChunkCfg:
    return ChunkCfg(chunk_tokens=chunk, overlap_tokens=overlap, fan_in_k=4)


def doc(*sections: tuple[str, list[str]]) -> Document:
    return build_document(
        "d", "pdf", "T", [RawSection(heading=h, level=1, paragraphs=ps) for h, ps in sections]
    )


def text(d: Document, c: Chunk) -> str:
    return d.full_text[c.span.start : c.span.end]


S = "w w w w."  # a 4-word sentence


# ------------------------------------------------------------ hand cases ---


def test_small_section_is_one_chunk_with_its_heading() -> None:
    d = doc(("Intro", [S, S]))
    (c,) = chunk_sections(d, cfg(100, 10), words)
    assert c.span == d.sections[0].span
    assert text(d, c).startswith("Intro")
    assert c.section_ids == ["s0"] and c.overlap is None and not c.oversized


def test_sections_are_never_merged_even_when_they_would_fit() -> None:
    d = doc(("A", [S]), ("B", [S]))
    cs = chunk_sections(d, cfg(100, 10), words)
    assert [c.section_ids for c in cs] == [["s0"], ["s1"]]


def test_uniform_baseline_ignores_section_boundaries() -> None:
    d = doc(("A", [S]), ("B", [S]))
    (c,) = chunk_uniform(d, cfg(100, 10), words)
    assert c.section_ids == ["s0", "s1"]
    assert text(d, c) == d.full_text


def test_oversized_section_splits_at_paragraph_boundaries() -> None:
    para = f"{S} {S}"  # 8 words
    d = doc(("Long", [para, para, para]))
    cs = chunk_sections(d, cfg(17, 0), words)  # heading(1)+8+8 fits, third does not
    assert len(cs) == 2
    assert text(d, cs[0]) == f"Long\n\n{para}\n\n{para}"
    assert text(d, cs[1]) == para


def test_overlap_is_trailing_whole_sentences_of_the_previous_piece() -> None:
    # three 3-word sentences (real words: pySBD reads "a." / "b." as initials)
    para = "Cats purr softly. Dogs bark loudly. Birds sing often."
    d = doc(("H", [para, para]))
    cs = chunk_sections(d, cfg(10, 6), words)
    assert len(cs) == 2
    # overlap: last two sentences of piece 1 (6 words), never a partial sentence
    assert cs[1].overlap is not None
    assert d.full_text[cs[1].overlap.start : cs[1].overlap.end] == (
        "Dogs bark loudly. Birds sing often."
    )
    assert text(d, cs[1]) == f"Dogs bark loudly. Birds sing often.\n\n{para}"
    # the overlap is a copy of the end of the previous chunk
    assert cs[1].overlap.end == cs[0].span.end


def test_oversized_paragraph_falls_back_to_sentences() -> None:
    big = " ".join([S] * 5)  # one 20-word paragraph
    d = doc(("", [big]))
    cs = chunk_sections(d, cfg(8, 0), words)
    assert [words(text(d, c)) for c in cs] == [8, 8, 4]


def test_single_sentence_over_budget_is_its_own_flagged_chunk() -> None:
    huge = " ".join(["x"] * 30) + "."
    d = doc(("", [S, huge, S]))
    cs = chunk_sections(d, cfg(10, 0), words)
    assert [c.oversized for c in cs] == [False, True, False]
    assert text(d, cs[1]) == huge


def test_no_overlap_across_sections() -> None:
    para = "Cats purr softly. Dogs bark loudly."
    d = doc(("A", [para, para]), ("B", [para]))
    cs = chunk_sections(d, cfg(7, 3), words)
    first_of_b = next(c for c in cs if c.section_ids == ["s1"])
    assert first_of_b.overlap is None


def test_ids_and_indexes_are_sequential() -> None:
    d = doc(("A", [S]), ("B", [S]), ("C", [S]))
    cs = chunk_sections(d, cfg(100, 0), words)
    assert [(c.id, c.index) for c in cs] == [("c0", 0), ("c1", 1), ("c2", 2)]


def test_empty_document_has_no_chunks() -> None:
    d = Document(doc_id="d", source="pdf", title="", full_text="")
    assert chunk_sections(d, cfg(10, 0), words) == []
    assert chunk_uniform(d, cfg(10, 0), words) == []


# ------------------------------------------------------- property tests ---

sentence = st.integers(min_value=1, max_value=12).map(lambda n: " ".join(["w"] * n) + ".")
paragraph = st.lists(sentence, min_size=1, max_size=6).map(" ".join)
section = st.tuples(
    st.sampled_from(["", "Intro", "Method", "Results"]),
    st.lists(paragraph, min_size=0, max_size=5),
)
documents = st.lists(section, min_size=1, max_size=6).map(lambda secs: doc(*secs))
budgets = st.tuples(st.integers(min_value=3, max_value=60), st.integers(min_value=0, max_value=20))


def check_invariants(
    d: Document, cs: list[Chunk], c: ChunkCfg, *, sectioned: bool, count: Count = words
) -> None:
    sec_of = {s.id: s for s in d.sections}
    for prev, cur in itertools.pairwise(cs):
        new_start = cur.overlap.end if cur.overlap else cur.span.start
        assert new_start >= prev.span.end or (cur.overlap and cur.overlap.end == prev.span.end)
        assert cur.index == prev.index + 1
    for ch in cs:
        assert 0 <= ch.span.start < ch.span.end <= len(d.full_text)
        assert ch.tokens == count(d.full_text[ch.span.start : ch.span.end])
        if sectioned:
            # FR-3.2: never crosses a section boundary
            (sid,) = ch.section_ids
            s = sec_of[sid].span
            assert s.start <= ch.span.start and ch.span.end <= s.end
        if ch.overlap:
            assert ch.span.start == ch.overlap.start < ch.overlap.end <= ch.span.end
            assert count(d.full_text[ch.overlap.start : ch.overlap.end]) <= c.overlap_tokens
        if not ch.oversized:
            assert ch.tokens <= c.chunk_tokens + c.overlap_tokens
        # boundaries sit on sentence edges, or a heading's (headings are not sentences)
        heads = [s for s in d.sections if s.heading]
        starts = {s.start for s in d.sentences} | {s.span.start for s in d.sections}
        ends = {s.end for s in d.sentences} | {s.span.start + len(s.heading) for s in heads}
        assert ch.span.start in starts and ch.span.end in ends
    # coverage: no sentence is lost
    for s in d.sentences:
        assert any(ch.span.start <= s.start and s.end <= ch.span.end for ch in cs)
    # every heading kept too
    for sec in d.sections:
        if sec.heading:
            assert any(ch.span.start <= sec.span.start < ch.span.end for ch in cs)


@settings(max_examples=300, deadline=None)
@given(documents, budgets)
def test_section_aware_invariants(d: Document, b: tuple[int, int]) -> None:
    c = ChunkCfg(chunk_tokens=b[0] + b[1] + 1, overlap_tokens=b[1], fan_in_k=4)
    cs = chunk_sections(d, c, words)
    check_invariants(d, cs, c, sectioned=True)
    assert chunk_sections(d, c, words) == cs  # deterministic


@settings(max_examples=300, deadline=None)
@given(documents, budgets)
def test_uniform_invariants(d: Document, b: tuple[int, int]) -> None:
    c = ChunkCfg(chunk_tokens=b[0] + b[1] + 1, overlap_tokens=b[1], fan_in_k=4)
    check_invariants(d, chunk_uniform(d, c, words), c, sectioned=False)


@settings(max_examples=200, deadline=None)
@given(documents)
def test_uniform_never_has_more_chunks_than_section_aware(d: Document) -> None:
    """Same packer and budget; sections can only add boundaries."""
    c = cfg(40, 0)
    assert len(chunk_uniform(d, c, words)) <= len(chunk_sections(d, c, words))


@pytest.mark.parametrize("chunk,overlap", [(3000, 150), (500, 50)])
def test_real_paper_fixture(chunk: int, overlap: int) -> None:
    from pathlib import Path

    from app.ingest.tei import parse_tei

    root = Path(__file__).resolve().parents[1] / "fixtures" / "tei"
    for p in sorted(root.glob("*.tei.xml")):
        t = parse_tei(p.read_bytes())
        d = build_document(p.name, "pdf", t.title, t.sections, t.references_text)
        c = cfg(chunk, overlap)
        check_invariants(d, chunk_sections(d, c, words), c, sectioned=True)
        check_invariants(d, chunk_uniform(d, c, words), c, sectioned=False)
