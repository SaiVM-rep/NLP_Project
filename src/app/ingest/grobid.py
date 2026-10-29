"""PDF -> GROBID TEI -> Document. PRD FR-1.1, FR-1.2, §11.4 step 1."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import httpx

from app.core.schemas import Document
from app.ingest.spans import build_document
from app.ingest.tei import IngestError, parse_tei


class GrobidClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8070",
        client: httpx.Client | None = None,
        timeout: float = 300.0,
        retries: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)
        self.retries = retries
        self.retry_delay = retry_delay

    def process_fulltext(self, pdf: bytes) -> bytes:
        url = f"{self.base_url}/api/processFulltextDocument"
        for attempt in range(self.retries):
            try:
                files = {"input": ("doc.pdf", pdf, "application/pdf")}
                resp = self._client.post(url, files=files)
            except httpx.TransportError as e:
                hint = "run `docker compose up -d grobid`"
                raise IngestError(f"GROBID unreachable at {self.base_url} ({e}); {hint}") from e
            # 503 = GROBID's worker pool is full; it says so explicitly and wants a retry.
            if resp.status_code == 503 and attempt < self.retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))
                continue
            if resp.status_code != 200:
                raise IngestError(f"GROBID {resp.status_code}: {resp.text[:200]}")
            return resp.content
        raise AssertionError("unreachable")


def ingest_pdf(path: Path, client: GrobidClient, max_mb: float = 25) -> Document:
    data = path.read_bytes()
    if not data.startswith(b"%PDF"):
        raise IngestError(f"{path.name} is not a PDF")
    if len(data) > max_mb * 1024 * 1024:
        raise IngestError(f"{path.name} is {len(data) / 1e6:.1f} MB, limit is {max_mb} MB")
    parsed = parse_tei(client.process_fulltext(data))
    return build_document(
        doc_id=hashlib.sha256(data).hexdigest()[:16],
        source="pdf",
        title=parsed.title or path.stem,
        sections=parsed.sections,
        references_text=parsed.references_text,
        structure_ok=parsed.structure_ok,
    )
