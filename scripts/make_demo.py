"""Build P7 demo bundles from stored GROBID TEI, with no Docker, API or database.

    .venv/Scripts/python.exe scripts/make_demo.py sbert [frank ...] [--min-free-gb 6]

Runs the real worker pipeline (`LiveStages`: Qwen summary, key points, verification) on one
in-process `MemoryStore` job per paper, so the result JSON is exactly what the API returns.
Only ingest differs: the paper's stored TEI (tests/fixtures/tei) instead of a GROBID call.
Writes demo/<name>.json. Needs Ollama running and the `make models` weights.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tune_aggrefact import available_gb  # noqa: E402

from app.core.config import load_config  # noqa: E402
from app.core.schemas import Document  # noqa: E402
from app.db.memory import MemoryStore  # noqa: E402
from app.ingest.spans import build_document  # noqa: E402
from app.ingest.tei import parse_tei  # noqa: E402
from app.ui.view import Bundle  # noqa: E402
from app.worker.pipeline import JobOptions, LiveStages, make_pipeline  # noqa: E402
from app.worker.runner import run_once  # noqa: E402

TEI = ROOT / "tests" / "fixtures" / "tei"


class TeiStages(LiveStages):
    def ingest(self, input: dict[str, Any]) -> Document:
        tei = parse_tei((TEI / f"{input['tei']}.tei.xml").read_bytes())
        return build_document(input["tei"], "pdf", tei.title, tei.sections, tei.references_text)


def document_payload(doc: Document) -> dict[str, Any]:
    """Same fields as GET /v1/documents/{id}/text."""
    return {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "full_text": doc.full_text,
        "structure_ok": doc.structure_ok,
        "sections": [
            s.model_dump(include={"id", "heading", "level", "span"}) for s in doc.sections
        ],
        "sentences": [s.model_dump() for s in doc.sentences],
    }


async def build(name: str) -> Bundle:
    store = MemoryStore()
    jid = store.create_job({"tei": name}, JobOptions().model_dump())
    pipeline = make_pipeline(lambda o: load_config(ROOT / "configs", o), TeiStages)
    await run_once(store, "demo", pipeline, lease_s=7200, max_attempts=1)
    job = store.get_job(jid)
    assert job is not None
    if job.status != "done" or job.doc_id is None:
        raise RuntimeError(f"{name}: job {job.status}: {job.error}")
    doc, result = store.document(job.doc_id), store.result(jid)
    assert doc is not None and result is not None
    return Bundle(title=doc.title or name, document=document_payload(doc), result=result)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("docs", nargs="+", help=f"names in {TEI}")
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    free = available_gb()
    if free < a.min_free_gb:
        print(f"only {free:.1f} GB RAM available, need {a.min_free_gb:g}: close apps first")
        return 2
    out_dir = ROOT / "demo"
    out_dir.mkdir(exist_ok=True)
    for name in a.docs:
        t0 = time.perf_counter()
        b = asyncio.run(build(name))
        (out_dir / f"{name}.json").write_text(b.model_dump_json(), encoding="utf-8")
        m = b.result["metrics"]
        print(
            f"{name}: {time.perf_counter() - t0:.0f} s, route {b.result['route']}, "
            f"{len(b.result['claims'])} claims, {len(b.result['key_points'])} key points, "
            f"faithfulness DeBERTa {m.get('faithfulness_deberta')} "
            f"MiniCheck {m.get('faithfulness_minicheck')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
