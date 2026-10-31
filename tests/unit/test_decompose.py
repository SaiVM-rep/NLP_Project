"""FR-5.1: summary -> sentences -> atomic claims, each tagged with its parent sentence."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import load_config
from app.llm.base import LLMResult
from app.llm.cache import DiskCache
from app.llm.client import LLMClient
from app.verify.decompose import Claim, DecomposeError, decompose

ROOT = Path(__file__).resolve().parents[2]


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
    cfg = load_config(ROOT / "configs")
    return LLMClient(p, cfg.profiles["qwen3_8b_local"], DiskCache(tmp_path))


async def test_each_claim_keeps_its_sentence_index(tmp_path: Path) -> None:
    p = ScriptedProvider(
        '{"claims": ["BART is a denoising autoencoder.", "BART is pretrained."]}',
        '{"claims": ["BART beats RoBERTa on SQuAD."]}',
    )
    claims = await decompose(
        "BART is a pretrained denoising autoencoder. It beats RoBERTa on SQuAD.",
        _client(p, tmp_path),
        prompt_dir=ROOT / "prompts",
    )
    assert claims == [
        Claim(text="BART is a denoising autoencoder.", sentence_idx=0),
        Claim(text="BART is pretrained.", sentence_idx=0),
        Claim(text="BART beats RoBERTa on SQuAD.", sentence_idx=1),
    ]


async def test_previous_sentence_is_passed_as_context(tmp_path: Path) -> None:
    p = ScriptedProvider('{"claims": ["a"]}', '{"claims": ["b"]}')
    await decompose("First one here. It does things.", _client(p, tmp_path), ROOT / "prompts")
    assert "SENTENCE:\nFirst one here." in p.prompts[0]
    assert "do NOT extract claims from it):\n(none)" in p.prompts[0]
    assert "SENTENCE:\nIt does things." in p.prompts[1]
    assert "do NOT extract claims from it):\nFirst one here." in p.prompts[1]


async def test_decoding_is_greedy_so_claims_are_stable_across_runs(tmp_path: Path) -> None:
    p = ScriptedProvider('{"claims": ["a"]}')
    await decompose("One sentence.", _client(p, tmp_path), ROOT / "prompts")
    assert p.params[0]["temperature"] == 0.0


async def test_decomposition_disables_model_thinking(tmp_path: Path) -> None:
    # measurement, not generation: hidden reasoning is 7x the time and varies the claims
    p = ScriptedProvider('{"claims": ["a"]}')
    await decompose("One sentence.", _client(p, tmp_path), ROOT / "prompts")
    assert p.params[0]["think"] is False


async def test_invalid_json_gets_one_repair_retry(tmp_path: Path) -> None:
    p = ScriptedProvider("not json", '```json\n{"claims": ["ok"]}\n```')
    claims = await decompose("One sentence.", _client(p, tmp_path), ROOT / "prompts")
    assert [c.text for c in claims] == ["ok"]
    assert "not valid JSON" in p.prompts[1]


async def test_invalid_twice_raises(tmp_path: Path) -> None:
    p = ScriptedProvider("nope", '{"claims": "not a list"}')
    with pytest.raises(DecomposeError):
        await decompose("One sentence.", _client(p, tmp_path), ROOT / "prompts")


async def test_blank_claims_are_dropped(tmp_path: Path) -> None:
    p = ScriptedProvider('{"claims": ["  ", "Real claim."]}')
    claims = await decompose("One sentence.", _client(p, tmp_path), ROOT / "prompts")
    assert [c.text for c in claims] == ["Real claim."]


async def test_empty_summary_makes_no_calls(tmp_path: Path) -> None:
    p = ScriptedProvider()
    assert await decompose("   ", _client(p, tmp_path), ROOT / "prompts") == []
    assert p.prompts == []


async def test_uses_v2_prompt_with_context_before_sentence(tmp_path: Path) -> None:
    """DECISION-15: v1 put CONTEXT last and Qwen3 (no thinking) pulled claims from it."""
    p = ScriptedProvider('{"claims": ["a"]}')
    await decompose("One sentence.", _client(p, tmp_path), ROOT / "prompts")
    prompt = p.prompts[0]
    assert prompt.index("CONTEXT") < prompt.index("SENTENCE:")
    assert "never from the CONTEXT" in prompt


def test_v1_prompt_is_still_frozen_on_disk() -> None:
    """Old runs must stay reproducible: a prompt change is a new file, never an edit."""
    v1 = (ROOT / "prompts" / "claim_decompose.v1.txt").read_text(encoding="utf-8")
    assert v1.startswith("Split the sentence into the smallest independent factual claims.")
    assert v1.rstrip().endswith("{context}")


async def test_malformed_reply_is_not_replayed_from_cache(tmp_path: Path) -> None:
    with pytest.raises(DecomposeError):
        await decompose(
            "One sentence.", _client(ScriptedProvider("[1]", "[1]"), tmp_path), ROOT / "prompts"
        )
    p = ScriptedProvider('{"claims": ["ok"]}')
    claims = await decompose("One sentence.", _client(p, tmp_path), ROOT / "prompts")
    assert [c.text for c in claims] == ["ok"]


async def test_decomposition_runs_in_a_small_window(tmp_path: Path) -> None:
    p = ScriptedProvider('{"claims": ["Cats purr."]}', '{"claims": ["Dogs bark."]}')
    await decompose("Cats purr. Dogs bark.", _client(p, tmp_path), prompt_dir=ROOT / "prompts")
    assert [x["num_ctx"] for x in p.params] == [4096, 4096]
