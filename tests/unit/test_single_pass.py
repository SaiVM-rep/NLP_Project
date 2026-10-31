"""FR-3.1 / FR-2.4: Route A single-pass summary, JSON validated with one repair retry."""

from __future__ import annotations

import json
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
    # tau = 0.8 x qwen3_8b_local effective_context 16384 (DECISION-27)
    assert res.route is Route.A and res.tokens > 0 and res.tau == 13107
    # FR-3.5: template hash per run, same sha256 as manifest.prompt_hashes
    assert list(res.prompt_hashes) == ["single_pass.v1.txt"]
    assert len(res.prompt_hashes["single_pass.v1.txt"]) == 64
    assert res.nodes == [] and res.chunks == []


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


async def test_route_b_documents_go_to_map_reduce(tmp_path: Path) -> None:
    from app.ingest.spans import RawSection, build_document

    cfg = load_config(ROOT / "configs")
    p = ScriptedProvider('{"summary": "chunk summary"}')
    client = LLMClient(p, cfg.active_profile, DiskCache(tmp_path))
    doc = build_document(
        "d1", "pdf", "T", [RawSection("Intro", 1, [" ".join(["word"] * 200) + "."])]
    )
    res = await summarize(doc, client, cfg, prompt_dir=ROOT / "prompts", tau_override=10)
    assert res.route is Route.B and res.summary == "chunk summary"
    assert len(res.nodes) == 1 and len(res.chunks) == 1  # 200 words: one chunk, no merge
    assert set(res.prompt_hashes) == {"chunk_summary.v1.txt", "merge.v1.txt"}


async def test_malformed_reply_is_not_replayed_from_cache(tmp_path: Path) -> None:
    """Live Route B: a wrong-shape reply was cached, and every rerun failed on it."""
    with pytest.raises(SummaryError):
        await _run(ScriptedProvider('{"x": "summary"}', '{"x": "summary"}'), tmp_path)
    p = ScriptedProvider('{"summary": "fresh"}')
    assert (await _run(p, tmp_path)).summary == "fresh"
    assert len(p.prompts) == 1


async def test_summary_call_sends_the_configured_think_flag(tmp_path: Path) -> None:
    p = ScriptedProvider('{"summary": "Short."}')
    await _run(p, tmp_path)
    assert p.params[0]["think"] is False  # configs/default.yaml summary.think


async def test_word_cap_limits_the_route_a_target(tmp_path: Path) -> None:
    import re

    p = ScriptedProvider('{"summary": "Short."}')
    cfg = load_config(ROOT / "configs", {"summary": {"max_words": 3}})
    client = LLMClient(p, cfg.active_profile, DiskCache(tmp_path))
    await summarize(_doc(), client, cfg, prompt_dir=ROOT / "prompts")
    m = re.search(r"about (\d+) words", p.prompts[0])
    assert m and int(m.group(1)) <= 3


async def test_capped_summary_far_over_the_cap_is_compressed(tmp_path: Path) -> None:
    # DECISION-38: Qwen copies its input when asked for fewer words (T5: 2,100 words for a
    # 500-word target, even when re-asked), but a sentence-capped schema is enforced.
    long = " ".join(["x"] * 40)
    p = ScriptedProvider(json.dumps({"summary": long}), '{"sentences": ["One.", "Two."]}')
    cfg = load_config(ROOT / "configs", {"summary": {"max_words": 10}})
    client = LLMClient(p, cfg.active_profile, DiskCache(tmp_path))
    res = await summarize(_doc(), client, cfg, prompt_dir=ROOT / "prompts")
    assert res.summary == "One. Two."
    assert len(p.prompts) == 2 and "at most 3 sentences" in p.prompts[1] and long in p.prompts[1]
    schema = p.params[1]["json_schema"]
    assert schema["properties"]["sentences"]["maxItems"] == 3  # enforced while decoding
    assert "compress.v1.txt" in res.prompt_hashes


async def test_capped_summary_within_slack_is_not_compressed(tmp_path: Path) -> None:
    p = ScriptedProvider(json.dumps({"summary": " ".join(["x"] * 12)}))  # 12 <= 1.25 x 10
    cfg = load_config(ROOT / "configs", {"summary": {"max_words": 10}})
    client = LLMClient(p, cfg.active_profile, DiskCache(tmp_path))
    res = await summarize(_doc(), client, cfg, prompt_dir=ROOT / "prompts")
    assert len(p.prompts) == 1 and "compress.v1.txt" not in res.prompt_hashes


async def test_uncapped_runs_are_never_compressed(tmp_path: Path) -> None:
    long = " ".join(["x"] * 400)  # far over the 10% target of a 201-word doc
    p = ScriptedProvider(json.dumps({"summary": long}))
    res = await _run(p, tmp_path)
    assert res.summary == long and len(p.prompts) == 1  # experiments: PRD behaviour
