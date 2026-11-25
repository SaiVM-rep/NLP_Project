"""Add tables & formulas (DECISION-40) to saved demo bundles: `make demo-objects`.

Each `demo/<name>.json` gets `result.objects` from its source PDF (found by name under
tests/fixtures/pdfs and data/p1_pdfs). Bundles with no matching PDF are left alone.
Extraction is cached by PDF hash in .cache/objects, so a rerun is quick.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.ingest.objects import extract_objects  # noqa: E402

PDF_DIRS = [ROOT / "tests" / "fixtures" / "pdfs", ROOT / "data" / "p1_pdfs"]


def main(names: list[str]) -> int:
    bundles = [ROOT / "demo" / f"{n}.json" for n in names] or sorted((ROOT / "demo").glob("*.json"))
    for b in bundles:
        pdf = next((d / f"{b.stem}.pdf" for d in PDF_DIRS if (d / f"{b.stem}.pdf").exists()), None)
        if pdf is None:
            print(f"{b.stem}: no source PDF found, skipped")
            continue
        objs = extract_objects(pdf, cache_dir=ROOT / ".cache" / "objects")
        data = json.loads(b.read_text(encoding="utf-8"))
        data["result"]["objects"] = [o.model_dump() for o in objs]
        b.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tables = sum(o.kind == "table" for o in objs)
        print(f"{b.stem}: {tables} tables, {len(objs) - tables} formulas", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
