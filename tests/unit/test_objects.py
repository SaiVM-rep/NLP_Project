"""Tables & formulas (DECISION-40): Docling runs in its own env as a subprocess; display only."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import pytest

from app.ingest.objects import PaperObject, extract_objects

TABLE = {
    "kind": "table",
    "page": 5,
    "caption": "Table 2: Results.",
    "content": "| a | b |\n|---|---|\n| 1 | 2 |",
    "image_png": "iVBORw0KGgo=",
}
FORMULA = {"kind": "formula", "page": 3, "caption": None, "content": "x^2", "image_png": "AA=="}


class FakeRunner:
    """Stands in for `python scripts/docling_extract.py PDF OUT`; writes OUT."""

    def __init__(self, payload: Any = None, error: Exception | None = None) -> None:
        self.payload = [TABLE, FORMULA] if payload is None else payload
        self.error = error
        self.calls: list[tuple[list[str], dict[str, str], float]] = []

    def __call__(self, args: list[str], env: dict[str, str], timeout: float) -> None:
        self.calls.append((args, env, timeout))
        if self.error is not None:
            raise self.error
        Path(args[-1]).write_text(json.dumps(self.payload), encoding="utf-8")


def _pdf(tmp_path: Path, body: bytes = b"%PDF-1.4 a") -> Path:
    p = tmp_path / "paper.pdf"
    p.write_bytes(body)
    return p


def test_objects_come_back_in_document_order(tmp_path: Path) -> None:
    run = FakeRunner()
    got = extract_objects(_pdf(tmp_path), cache_dir=tmp_path / "c", runner=run)
    assert got == [PaperObject(**TABLE), PaperObject(**FORMULA)]
    args, _, timeout = run.calls[0]
    assert args[-2].endswith("paper.pdf") and timeout > 0


def test_same_pdf_bytes_are_extracted_once(tmp_path: Path) -> None:
    run = FakeRunner()
    pdf = _pdf(tmp_path)
    extract_objects(pdf, cache_dir=tmp_path / "c", runner=run)
    again = extract_objects(pdf, cache_dir=tmp_path / "c", runner=run)
    assert len(run.calls) == 1 and len(again) == 2


def test_subprocess_runs_offline_with_the_default_model_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Docling's weights sit in the default HF cache; the worker's HF_HOME points elsewhere.
    monkeypatch.setenv("HF_HOME", "C:/elsewhere")
    run = FakeRunner()
    extract_objects(_pdf(tmp_path), cache_dir=tmp_path / "c", runner=run)
    env = run.calls[0][1]
    assert env["HF_HUB_OFFLINE"] == "1" and "HF_HOME" not in env


@pytest.mark.parametrize(
    "error",
    [
        subprocess.CalledProcessError(1, "x"),
        subprocess.TimeoutExpired("x", 1),
        FileNotFoundError("no .venv-docling"),
    ],
)
def test_extraction_failure_never_fails_the_job(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, error: Exception
) -> None:
    with caplog.at_level(logging.WARNING):
        got = extract_objects(
            _pdf(tmp_path), cache_dir=tmp_path / "c", runner=FakeRunner(error=error)
        )
    assert got == []
    assert any("tables and formulas" in r.getMessage() for r in caplog.records)


def test_malformed_output_is_dropped_not_cached(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bad = FakeRunner(payload=[{"kind": "figure"}])
    with caplog.at_level(logging.WARNING):
        assert extract_objects(_pdf(tmp_path), cache_dir=tmp_path / "c", runner=bad) == []
    good = FakeRunner()
    assert len(extract_objects(_pdf(tmp_path), cache_dir=tmp_path / "c", runner=good)) == 2


def test_results_stored_before_tables_existed_render_with_no_objects() -> None:
    from uuid import uuid4

    from app.db.store import Job, render_result

    old = {"summary": "s", "tokens": 1, "tau": 1, "model": "m", "prompt_hashes": {}, "metrics": {}}
    job = Job(job_id=uuid4(), status="done", input={}, config={})
    assert render_result(job, old, [], [], [])["objects"] == []
