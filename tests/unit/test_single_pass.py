"""FR-3.1 / FR-2.4: Route A single-pass summary, JSON validated with one repair retry."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import load_config
from app.core.schemas import Document, Route, Section, Span
from app.llm.base import LLMResult
from app.llm.cache import DiskCache
from app.llm.client import LLMClient
from app.summarize.single_pass import SummaryError, summarize

ROOT = Path(__file__).resolve().parents[2]
TEXT = "Intro\n\n" + " ".join(["word"] * 200)


class ScriptedProvider:
    name = "gemini"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.params: list[dict] = []  # type: ignore[type-arg]

    async def generate(self, prompt, *, model, params, seed):  # type: ignore[no-untyped-def]
        self.prompts.append(prompt)
        self.params.append(params)
        return LLMResult(
            text=self.replies.pop(0), tokens_in=1, tokens_out=1, provider="gemini", model=model
        )


def _doc() -> Document:
    return Document(
        doc_id="d1",
        source="pdf",
        title="T",
        full_text=TEXT,
        sections=[Section(id="s0", heading="Intro", level=1, span=Span(start=0, end=len(TEXT)))],
    )


def _run(provider: ScriptedProvider, tmp_path: Path, **kw):  # type: ignore[no-untyped-def]
    cfg = load_config(ROOT / "configs")
    client = LLMClient(provider, cfg.active_profile, DiskCache(tmp_path))
    return summarize(_doc(), client, cfg, prompt_dir=ROOT / "prompts", **kw)


async def test_returns_summary_with_routing_metadata(tmp_path: Path) -> None:
    p = ScriptedProvider('{"summary": "Short."}')
    res = await _run(p, tmp_path)
    assert res.summary == "Short."
    assert res.route is Route.A and res.tokens > 0 and res.tau == 838860
    assert res.prompt == "single_pass.v1.txt" and len(res.prompt_hash) == 64


async def test_prompt_carries_document_target_length_and_json_mode(tmp_path: Path) -> None:
    p = ScriptedProvider('{"summary": "x"}')
    await _run(p, tmp_path)
    assert TEXT in p.prompts[0]
    assert "about 20 words (10% of the input)" in p.prompts[0]
    assert "{document}" not in p.prompts[0]
    assert p.params[0]["json_schema"]["required"] == ["summary"]


async def test_invalid_json_gets_one_repair_retry(tmp_path: Path) -> None:
    p = ScriptedProvider("not json", '{"summary": "fixed"}')
    res = await _run(p, tmp_path)
    assert res.summary == "fixed" and len(p.prompts) == 2
    assert "not valid JSON" in p.prompts[1]


async def test_tolerates_markdown_code_fence(tmp_path: Path) -> None:
    p = ScriptedProvider('```json\n{"summary": "fenced"}\n```')
    assert (await _run(p, tmp_path)).summary == "fenced"


async def test_second_invalid_reply_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SummaryError, match="JSON"):
        await _run(ScriptedProvider("nope", '{"wrong": 1}'), tmp_path)


async def test_route_b_documents_are_refused_until_p4(tmp_path: Path) -> None:
    with pytest.raises(SummaryError, match="Route B"):
        await _run(ScriptedProvider(), tmp_path, tau_override=10)
