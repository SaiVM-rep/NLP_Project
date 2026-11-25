"""DECISION-40, live: the real extractor in .venv-docling on SBERT (7 tables, 2 formulas).

Needs `make docling-env` and a GPU (CPU works, slowly). Runs the subprocess the worker runs.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from app.ingest.objects import DEFAULT_PYTHON, extract_objects

ROOT = Path(__file__).resolve().parents[2]
PNG = b"\x89PNG\r\n\x1a\n"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not DEFAULT_PYTHON.exists(), reason="run `make docling-env` first"),
]


def test_sbert_tables_and_formulas_with_page_crops(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(args: list[str], env: dict[str, str], timeout: float) -> None:
        import subprocess

        calls.append(args)
        subprocess.run(args, env=env, timeout=timeout, check=True, capture_output=True)

    objs = extract_objects(
        ROOT / "tests/fixtures/pdfs/sbert.pdf", cache_dir=tmp_path, runner=runner
    )
    assert len(calls) == 1
    tables = [o for o in objs if o.kind == "table"]
    formulas = [o for o in objs if o.kind == "formula"]
    assert len(tables) == 7 and len(formulas) == 2
    # captions are titles, not repeated inside the table text
    t2 = next(t for t in tables if (t.caption or "").startswith("Table 2"))
    assert t2.page == 5 and not t2.content.startswith("Table 2")
    assert "softmax" in formulas[0].content
    assert all(base64.b64decode(o.image_png).startswith(PNG) for o in objs)
    assert [o.page for o in objs] == sorted(o.page for o in objs)  # document order
