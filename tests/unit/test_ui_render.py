"""P7 rendering (FR-6.4, FR-6.5): source text with highlighted spans, claim colours.

Offsets are the backbone (SC2): the highlighted text must be exactly full_text[start:end].
"""

import html
import re

from app.core.schemas import ClaimLabel
from app.ui.render import claim_choice, label_icon, render_source


def _marks(out: str) -> list[str]:
    return [html.unescape(m) for m in re.findall(r"<mark[^>]*>(.*?)</mark>", out, flags=re.S)]


def test_marks_exactly_the_span_text() -> None:
    text = "alpha beta gamma delta"
    out = render_source(text, [(6, 10)])
    assert _marks(out) == ["beta"]


def test_text_outside_the_spans_is_kept_in_order() -> None:
    out = render_source("alpha beta gamma", [(6, 10)])
    plain = html.unescape(re.sub(r"<[^>]+>", "", out))
    assert plain == "alpha beta gamma"


def test_escapes_html_in_the_document() -> None:
    out = render_source("a <script>x</script> b", [(2, 10)])
    assert "<script>" not in out
    assert _marks(out) == ["<script>"]


def test_only_the_first_mark_is_the_scroll_target() -> None:
    out = render_source("one two three four", [(0, 3), (8, 13)])
    assert out.count('id="active-hl"') == 1
    assert out.index('id="active-hl"') < out.index("three")


def test_overlapping_spans_merge_into_one_mark() -> None:
    out = render_source("0123456789", [(2, 6), (4, 8)])
    assert _marks(out) == ["234567"]


def test_out_of_range_and_empty_spans_are_skipped_not_crashed() -> None:
    out = render_source("short", [(3, 99), (4, 4), (-1, 2)])
    assert _marks(out) == []


def test_no_spans_gives_plain_text() -> None:
    assert _marks(render_source("plain text", [])) == []


def test_unicode_offsets_are_character_offsets() -> None:
    text = "naïve café – ok"  # noqa: RUF001 (non-ASCII on purpose)
    start = text.index("café")
    assert _marks(render_source(text, [(start, start + 4)])) == ["café"]


def test_label_icons_differ_per_label() -> None:
    icons = {label_icon(label) for label in ClaimLabel}
    assert len(icons) == 3


def test_claim_choice_shows_both_verifiers_and_the_claim() -> None:
    text = claim_choice(ClaimLabel.SUPPORTED, ClaimLabel.CONTRADICTED, "SBERT uses triplet loss.")
    assert label_icon(ClaimLabel.SUPPORTED) in text
    assert label_icon(ClaimLabel.CONTRADICTED) in text
    assert text.endswith("SBERT uses triplet loss.")


# ------------------------------------------------- tables & formulas (DECISION-40) ---

from app.ui.render import render_objects  # noqa: E402

TABLE = {
    "kind": "table",
    "page": 5,
    "caption": "Table 2: <b>STS</b> results.",
    "content": "| Model | Spearman |\n|---|---|\n| SBERT | 85.35 |",
    "image_png": "QUJD",
}
FORMULA = {
    "kind": "formula",
    "page": 3,
    "caption": None,
    "content": "o = W_t u",
    "image_png": "REVG",
}
UNREAD = {"kind": "formula", "page": 4, "caption": None, "content": "", "image_png": "R0hJ"}


def test_each_object_shows_the_page_crop_beside_the_extraction() -> None:
    out = render_objects([TABLE])
    crop, extracted = out.index('src="data:image/png;base64,QUJD"'), out.index("| SBERT | 85.35 |")
    assert crop < extracted  # crop left, extraction right
    assert "page 5" in out and "Original" in out and "Extracted" in out


def test_captions_are_escaped() -> None:
    out = render_objects([TABLE])
    assert "<b>STS</b>" not in out and "&lt;b&gt;STS&lt;/b&gt;" in out


def test_formulas_render_as_display_math_and_unread_ones_say_so() -> None:
    out = render_objects([FORMULA, UNREAD])
    assert "$$o = W_t u$$" in out
    assert "could not read" in out and 'src="data:image/png;base64,R0hJ"' in out


def test_markdown_inside_the_side_by_side_block_is_separated_by_blank_lines() -> None:
    # CommonMark: markdown inside an HTML block is parsed only after a blank line
    out = render_objects([TABLE])
    assert "\n\n| Model | Spearman |" in out and "| SBERT | 85.35 |\n\n</div>" in out


def test_counts_in_the_heading_and_an_empty_paper() -> None:
    assert "1 table · 2 formulas" in render_objects([TABLE, FORMULA, UNREAD])
    assert "No tables or formulas" in render_objects([])


def test_each_object_is_tagged_with_its_kind_for_layout() -> None:
    out = render_objects([TABLE, FORMULA])
    assert 'class="obj obj-table"' in out and 'class="obj obj-formula"' in out
