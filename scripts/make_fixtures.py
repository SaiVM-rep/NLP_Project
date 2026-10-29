"""Regenerate tests/fixtures/{tei,golden} from tests/fixtures/pdfs via live GROBID.

Run after changing the GROBID image (DECISION-10), then review `git diff` on the
golden files before committing: a heading change there is a parser regression
until proven otherwise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.ingest.grobid import GrobidClient
from app.ingest.spans import build_document
from app.ingest.tei import parse_tei

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def main() -> None:
    client = GrobidClient(os.getenv("GROBID_URL", "http://localhost:8070"))
    (FIXTURES / "tei").mkdir(exist_ok=True)
    (FIXTURES / "golden").mkdir(exist_ok=True)
    for pdf in sorted((FIXTURES / "pdfs").glob("*.pdf")):
        tei = client.process_fulltext(pdf.read_bytes())
        (FIXTURES / "tei" / f"{pdf.stem}.tei.xml").write_bytes(tei)
        parsed = parse_tei(tei)
        # Same title fallback as ingest_pdf, so the golden test compares like with like.
        doc = build_document(pdf.stem, "pdf", parsed.title or pdf.stem, parsed.sections)
        golden = {
            "title": doc.title,
            "structure_ok": parsed.structure_ok,
            "headings": [s.heading for s in doc.sections],
        }
        out = FIXTURES / "golden" / f"{pdf.stem}.json"
        out.write_text(json.dumps(golden, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{pdf.stem}: {len(golden['headings'])} sections, structure_ok={parsed.structure_ok}")


if __name__ == "__main__":
    main()
