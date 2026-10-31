"""FR-4.1: candidate key points per section. PRD §11.7 step 1, prompt `keypoints.v1`."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import load_config
from app.core.schemas import Document
from app.ingest.spans import RawSection, build_document
from app.keypoints.candidates import Candidate, CandidateError, generate_candidates
from app.llm.base import LLMResult
from app.llm.cache import DiskCache
from app.llm.client import LLMClient

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "configs")


class ScriptedProvider:
    name = "ollama"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.params: list[dict] = []  # type: ignore[type-arg]

    async def generate(self, prompt, *, model, params, seed):  # type: ignore[no-untyped-def]
        self.prompts.append(prompt)
        self.params.append(params)
        return LLMResult(
            text=self.replies.pop(0), tokens_in=1, tokens_out=1, provider="ollama", model=model
        )


def _client(p: ScriptedProvider, tmp_path: Path) -> LLMClient:
    return LLMClient(p, CFG.profiles["qwen3_8b_local"], DiskCache(tmp_path))


def _doc() -> Document:
    return build_document(
        "d",
        "pdf",
        "T",
        [
            RawSection(heading="Introduction", level=1, paragraphs=["Cats purr. Dogs bark."]),
            RawSection(heading="Method", level=1, paragraphs=["We count purrs."]),
        ],
    )


async def test_one_call_per_section_points_keep_section_and_document_order(
    tmp_path: Path,
) -> None:
    p = ScriptedProvider(
        '{"points": ["Cats purr.", "Dogs bark."]}', '{"points": ["Purrs are counted."]}'
    )
    out = await generate_candidates(_doc(), _client(p, tmp_path), CFG, ROOT / "prompts")
    assert out == [
        Candidate(text="Cats purr.", section_id="s0", position=0),
        Candidate(text="Dogs bark.", section_id="s0", position=1),
        Candidate(text="Purrs are counted.", section_id="s1", position=2),
    ]


async def test_prompt_carries_heading_section_text_and_n(tmp_path: Path) -> None:
    p = ScriptedProvider('{"points": []}', '{"points": []}')
    await generate_candidates(_doc(), _client(p, tmp_path), CFG, ROOT / "prompts")
    n = CFG.keypoints.candidates_per_section
    assert f"List up to {n} key points" in p.prompts[0]
    assert "SECTION (Introduction):\nIntroduction\n\nCats purr. Dogs bark." not in p.prompts[0]
    assert "SECTION (Introduction):\nCats purr. Dogs bark." in p.prompts[0]
    assert "SECTION (Method):\nWe count purrs." in p.prompts[1]


async def test_decoding_comes_from_config_like_the_summaries(tmp_path: Path) -> None:
    p = ScriptedProvider('{"points": []}', '{"points": []}')
    await generate_candidates(_doc(), _client(p, tmp_path), CFG, ROOT / "prompts")
    assert p.params[0]["temperature"] == CFG.decoding.temperature
    assert p.params[0]["top_p"] == CFG.decoding.top_p


async def test_candidates_disable_model_thinking(tmp_path: Path) -> None:
    """DECISION-25: Qwen3 thinking made each section call 26-60 s on SBERT (2026-10-30)."""
    p = ScriptedProvider('{"points": []}', '{"points": []}')
    await generate_candidates(_doc(), _client(p, tmp_path), CFG, ROOT / "prompts")
    assert [q["think"] for q in p.params] == [False, False]


async def test_more_points_than_asked_are_cut_to_n(tmp_path: Path) -> None:
    n = CFG.keypoints.candidates_per_section
    many = ", ".join(f'"Point {i}."' for i in range(n + 3))
    p = ScriptedProvider(f'{{"points": [{many}]}}', '{"points": []}')
    out = await generate_candidates(_doc(), _client(p, tmp_path), CFG, ROOT / "prompts")
    assert len(out) == n


async def test_section_without_body_text_is_skipped(tmp_path: Path) -> None:
    doc = build_document(
        "d",
        "pdf",
        "T",
        [
            RawSection(heading="Empty", level=1, paragraphs=[]),
            RawSection(heading="Body", level=1, paragraphs=["Text here."]),
        ],
    )
    p = ScriptedProvider('{"points": ["Text is here."]}')
    out = await generate_candidates(doc, _client(p, tmp_path), CFG, ROOT / "prompts")
    assert len(p.prompts) == 1
    assert out == [Candidate(text="Text is here.", section_id="s1", position=0)]


async def test_invalid_json_gets_one_repair_retry_then_fails_naming_the_section(
    tmp_path: Path,
) -> None:
    p = ScriptedProvider("nope", '{"points": ["ok"]}', "nope", "still nope")
    with pytest.raises(CandidateError, match="s1"):
        await generate_candidates(_doc(), _client(p, tmp_path), CFG, ROOT / "prompts")
    assert len(p.prompts) == 4


async def test_all_section_calls_share_one_small_window(tmp_path: Path) -> None:
    # One window for the whole stage: a change between calls would reload the model.
    p = ScriptedProvider('{"points": ["Cats purr."]}', '{"points": ["Purrs counted."]}')
    await generate_candidates(_doc(), _client(p, tmp_path), CFG, prompt_dir=ROOT / "prompts")
    assert [x["num_ctx"] for x in p.params] == [4096, 4096]
