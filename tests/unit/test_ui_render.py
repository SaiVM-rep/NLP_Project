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
