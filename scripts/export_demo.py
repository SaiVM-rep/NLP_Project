"""Save a finished job as a P7 demo bundle: demo/<name>.json (result + document text).

    .venv/Scripts/python.exe scripts/export_demo.py JOB_ID [--name sbert] [--api http://127.0.0.1:8000]

Demo mode then opens it with no GPU, database or network (needs `make api` only while exporting).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.ui.view import bundle_from_api  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("--name", help="file name (default: from the title)")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    a = ap.parse_args()
    with httpx.Client(base_url=a.api, timeout=60) as c:
        b = bundle_from_api(c, a.job_id)
    name = a.name or re.sub(r"[^a-z0-9]+", "_", b.title.lower()).strip("_")[:40] or a.job_id
    out = ROOT / "demo" / f"{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(b.model_dump_json(), encoding="utf-8")
    n_claims, n_kps = len(b.result["claims"]), len(b.result["key_points"])
    print(f"{b.title!r}: {n_claims} claims, {n_kps} key points -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
