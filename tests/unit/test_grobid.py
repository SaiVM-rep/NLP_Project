"""FR-1.1 / FR-1.2: PDF -> GROBID -> Document, no network (MockTransport)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.ingest.grobid import GrobidClient, ingest_pdf
from app.ingest.tei import IngestError

TEI = b"""<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt>
<title>Paper</title></titleStmt></fileDesc></teiHeader><text><body>
<div><head n="1">Intro</head><p>Hello world. Second sentence.</p></div>
<div><head n="2">End</head><p>Bye.</p></div></body></text></TEI>"""

PDF = b"%PDF-1.4\n fake body"


def _client(handler) -> GrobidClient:  # type: ignore[no-untyped-def]
    return GrobidClient(
        "http://grobid:8070",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retry_delay=0,
    )


def _pdf(tmp_path: Path, data: bytes = PDF, name: str = "p.pdf") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_posts_pdf_to_fulltext_endpoint_and_builds_document(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, content=TEI)

    doc = ingest_pdf(_pdf(tmp_path), _client(handler))
    assert seen[0].url.path == "/api/processFulltextDocument"
    assert b"fake body" in seen[0].content
    assert doc.source == "pdf" and doc.title == "Paper" and doc.structure_ok
    assert [s.heading for s in doc.sections] == ["Intro", "End"]
    assert doc.text_of(doc.sentences[0]) == "Hello world."


def test_doc_id_is_content_hash(tmp_path: Path) -> None:
    client = _client(lambda r: httpx.Response(200, content=TEI))
    a = ingest_pdf(_pdf(tmp_path, name="a.pdf"), client)
    b = ingest_pdf(_pdf(tmp_path, name="b.pdf"), client)
    assert a.doc_id == b.doc_id and len(a.doc_id) == 16


def test_retries_503_busy_then_succeeds(tmp_path: Path) -> None:
    calls = iter([httpx.Response(503), httpx.Response(200, content=TEI)])
    doc = ingest_pdf(_pdf(tmp_path), _client(lambda r: next(calls)))
    assert doc.title == "Paper"


def test_gives_up_after_retries(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="503"):
        ingest_pdf(_pdf(tmp_path), _client(lambda r: httpx.Response(503)))


def test_unreachable_grobid_is_a_clear_error(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(IngestError, match="docker compose up"):
        ingest_pdf(_pdf(tmp_path), _client(handler))


def test_rejects_non_pdf(tmp_path: Path) -> None:
    client = _client(lambda r: httpx.Response(200, content=TEI))
    with pytest.raises(IngestError, match="not a PDF"):
        ingest_pdf(_pdf(tmp_path, data=b"hello", name="x.pdf"), client)


def test_rejects_oversize_file(tmp_path: Path) -> None:
    client = _client(lambda r: httpx.Response(200, content=TEI))
    with pytest.raises(IngestError, match="MB"):
        ingest_pdf(_pdf(tmp_path, data=PDF + b"x" * 2_000_000), client, max_mb=1)
