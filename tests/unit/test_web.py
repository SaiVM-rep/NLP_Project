"""FR-1.1 / FR-1.3: web article -> trafilatura -> Document, no network (MockTransport)."""

from __future__ import annotations

import httpx
import pytest

from app.ingest.tei import IngestError
from app.ingest.web import ingest_url, parse_web, recover_headings

FILLER = "This sentence is padding so trafilatura treats the paragraph as real content."

HTML = f"""<html><head><title>Site | Compressing Text</title>
<meta property="og:title" content="Compressing Text"></head><body>
<nav><a href="/">Home</a> <a href="/about">About us</a> <a href="/login">Login</a></nav>
<article>
<h1>Compressing Text</h1>
<p>Lead paragraph with <b>bold</b> words. {FILLER}</p>
<h2>Background</h2>
<p>Background one. {FILLER}</p>
<p>Background two. {FILLER}</p>
<h3>Detail</h3>
<p>Detail text. {FILLER}</p>
<ul><li>first item</li><li>second item</li></ul>
<h2>Results</h2>
<p>Results text. {FILLER}</p>
</article>
<footer>Copyright 2026 Footer Corp. All rights reserved.</footer>
</body></html>"""

URL = "https://example.org/post"
# content=, not text=: httpx forces text/plain when given text=.
HTML_HEADERS = {"content-type": "text/html; charset=utf-8"}


def _http(handler) -> httpx.Client:  # type: ignore[no-untyped-def]
    return httpx.Client(transport=httpx.MockTransport(handler))


def _html_ok(req: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=HTML.encode(), headers=HTML_HEADERS)


def test_title_comes_from_page_metadata() -> None:
    assert parse_web(HTML).title == "Compressing Text"


def test_headings_become_sections_with_levels_from_tag() -> None:
    secs = parse_web(HTML).sections
    assert [(s.heading, s.level) for s in secs] == [
        ("Compressing Text", 1),
        ("Background", 2),
        ("Detail", 3),
        ("Results", 2),
    ]


def test_paragraphs_keep_inline_text_and_lists_become_one_paragraph() -> None:
    secs = parse_web(HTML).sections
    assert secs[0].paragraphs == [f"Lead paragraph with bold words. {FILLER}"]
    assert len(secs[1].paragraphs) == 2
    assert secs[2].paragraphs[1] == "first item\nsecond item"


def test_boilerplate_is_absent() -> None:
    parsed = parse_web(HTML)
    text = " ".join(p for s in parsed.sections for p in [s.heading, *s.paragraphs])
    for junk in ("About us", "Login", "Footer Corp", "All rights reserved"):
        assert junk not in text


def test_structure_ok_needs_two_headings() -> None:
    assert parse_web(HTML).structure_ok
    flat = f"<html><body><article><p>Only text. {FILLER}</p><p>More. {FILLER}</p></article>"
    parsed = parse_web(flat)
    assert not parsed.structure_ok
    assert [s.heading for s in parsed.sections] == [""]


def test_page_with_no_extractable_text_raises() -> None:
    with pytest.raises(IngestError, match="no article text"):
        parse_web("<html><body><nav>Home</nav></body></html>")


def test_ingest_url_builds_document_with_round_trip_spans() -> None:
    doc = ingest_url(URL, _http(_html_ok))
    assert doc.source == "web"
    assert doc.title == "Compressing Text"
    assert doc.references_text is None
    assert doc.structure_ok
    assert len(doc.doc_id) == 16
    parsed = parse_web(HTML)
    for sec, raw in zip(doc.sections, parsed.sections, strict=True):
        assert doc.text_of(sec.span).startswith(raw.heading)
        assert [doc.text_of(p) for p in sec.paragraphs] == raw.paragraphs
        assert all(sec.span.start <= p.start and p.end <= sec.span.end for p in sec.paragraphs)
    assert doc.sentences and all(doc.text_of(s).strip() == doc.text_of(s) for s in doc.sentences)


def test_ingest_url_falls_back_to_url_when_page_has_no_title() -> None:
    # No <title>, no headings: trafilatura otherwise promotes the first heading to title.
    bare = f"<html><body><article><p>x. {FILLER}</p><p>y. {FILLER}</p></article></body></html>"
    doc = ingest_url(
        URL, _http(lambda r: httpx.Response(200, content=bare.encode(), headers=HTML_HEADERS))
    )
    assert doc.title == URL


@pytest.mark.parametrize("bad", ["ftp://example.org/x", "file:///etc/passwd", "example.org"])
def test_rejects_non_http_urls(bad: str) -> None:
    with pytest.raises(IngestError, match="http"):
        ingest_url(bad, _http(_html_ok))


def test_rejects_non_html_content() -> None:
    pdf = _http(
        lambda r: httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"})
    )
    with pytest.raises(IngestError, match="application/pdf"):
        ingest_url(URL, pdf)


def test_rejects_oversized_page() -> None:
    with pytest.raises(IngestError, match="limit"):
        ingest_url(URL, _http(_html_ok), max_mb=0.0001)


def test_http_error_and_unreachable_become_ingest_errors() -> None:
    with pytest.raises(IngestError, match="404"):
        ingest_url(URL, _http(lambda r: httpx.Response(404, text="nope")))

    def down(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(IngestError, match="unreachable"):
        ingest_url(URL, _http(down))


def test_references_section_goes_to_metadata_not_full_text() -> None:
    # FR-1.6 parity with the PDF path: found live on a real blog post.
    page = HTML.replace(
        "</article>",
        "<h2>References</h2><p>[1] Smith. Prior work on compression, 2020.</p>"
        "<p>[2] Jones. Other work on summarization, 2021.</p></article>",
    )
    parsed = parse_web(page)
    assert "References" not in [s.heading for s in parsed.sections]
    assert parsed.references_text is not None and "Prior work" in parsed.references_text


def test_heading_permalink_glyph_is_stripped() -> None:
    # Sphinx (docs.python.org) appends a "¶" permalink to every heading.
    page = HTML.replace(
        "<h2>Results</h2>", '<h2>Results<a class="headerlink" href="#results">¶</a></h2>'
    )
    assert "Results" in [s.heading for s in parse_web(page).sections]


def test_related_posts_section_is_dropped() -> None:
    # Found live on simonwillison.net: trafilatura keeps the sidebar list.
    page = HTML.replace(
        "</article>",
        f"<h2>More recent articles</h2><p>Some other post. {FILLER}</p></article>",
    )
    parsed = parse_web(page)
    assert "More recent articles" not in [s.heading for s in parsed.sections]
    assert not any("Some other post" in p for s in parsed.sections for p in s.paragraphs)


def test_recover_headings_puts_page_headings_back_between_paragraphs() -> None:
    page = (
        "<html><body><header><h1>Site name</h1></header>"
        "<p>Intro words here and more.</p>"
        "<h2>First <b>part</b>¶</h2><p>Alpha  text\n spans lines.</p><p>Beta text.</p>"
        "<h3>Sub</h3><p>Gamma text.</p>"
        "<footer><h2>Footer heading</h2></footer></body></html>"
    )
    blocks = [
        (0, "Intro words here and more."),
        (0, "Alpha text spans lines."),
        (0, "Beta text."),
        (0, "Not in the page at all."),
        (0, "Gamma text."),
    ]
    assert recover_headings(page, blocks) == [
        (0, "Intro words here and more."),
        (2, "First part"),
        (0, "Alpha text spans lines."),
        (0, "Beta text."),
        (0, "Not in the page at all."),
        (3, "Sub"),
        (0, "Gamma text."),
    ]
