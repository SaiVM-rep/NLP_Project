"""P2 exit: `python -m app.cli summarize FILE` -> summary on stdout + JSON on disk."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import cli
from app.core.schemas import Document, Section, Span
from app.llm.base import LLMResult

TEXT = "Intro\n\nSome body text here."


class FakeProvider:
    name = "gemini"

    async def generate(self, prompt, *, model, params, seed):  # type: ignore[no-untyped-def]
        return LLMResult(
            text='{"summary": "A summary."}',
            tokens_in=1,
            tokens_out=1,
            provider="gemini",
            model=model,
        )


def _fake_ingest(path, client, max_mb=25):  # type: ignore[no-untyped-def]
    return Document(
        doc_id="abc",
        source="pdf",
        title="T",
        full_text=TEXT,
        sections=[Section(id="s0", heading="Intro", level=1, span=Span(start=0, end=len(TEXT)))],
    )


def test_summarize_prints_summary_and_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(cli, "ingest_pdf", _fake_ingest)
    monkeypatch.setattr(cli, "make_provider", lambda profile: FakeProvider())
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path / "cache"))

    code = cli.main(["summarize", str(pdf), "--out", str(tmp_path / "out")])

    assert code == 0
    assert "A summary." in capsys.readouterr().out
    saved = json.loads((tmp_path / "out" / "abc.json").read_text(encoding="utf-8"))
    assert saved["summary"] == "A summary." and saved["route"] == "A"
    assert saved["document"]["title"] == "T"


def test_ollama_gets_the_profile_context_size() -> None:
    # Without num_ctx Ollama defaults to 4096 and silently drops the START of the
    # prompt (the instructions), so the model loops until "token repeat limit".
    cfg = cli.load_config(cli.ROOT / "configs")
    profile = cfg.profiles["qwen3_8b_local"]
    provider = cli.make_provider(profile)
    assert provider.num_ctx == profile.effective_context  # type: ignore[attr-defined]


def test_url_argument_goes_through_web_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[str] = []

    def fake_url(url, client, max_mb=25):  # type: ignore[no-untyped-def]
        seen.append(url)
        return _fake_ingest(url, client)

    monkeypatch.setattr(cli, "ingest_url", fake_url)
    monkeypatch.setattr(cli, "make_provider", lambda profile: FakeProvider())
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path / "cache"))

    code = cli.main(["summarize", "https://example.org/a", "--out", str(tmp_path / "out")])

    assert code == 0 and seen == ["https://example.org/a"]
    assert "A summary." in capsys.readouterr().out


def test_missing_file_is_a_clean_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["summarize", "nope.pdf"]) == 1
    assert "not found" in capsys.readouterr().err


def test_ingest_failure_is_logged_as_json_with_its_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # P1 exit / NFR-5: a failed document must leave a structured log line naming it,
    # not just a human-readable print, so batch runs can be audited afterwards.
    bad = tmp_path / "notpdf.pdf"
    bad.write_bytes(b"hello")

    code = cli.main(["summarize", str(bad), "--out", str(tmp_path / "out")])

    assert code == 1
    records = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line[:1] == "{"]
    errors = [r for r in records if r["level"] == "ERROR"]
    assert errors and errors[0]["source"] == str(bad)
    assert "not a PDF" in errors[0]["msg"]
