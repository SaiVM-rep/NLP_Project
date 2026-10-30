"""PRD §11.7 end to end with a scripted LLM and a hand-made embedder: FR-4.1..4.5 wired."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.core.config import AppConfig, load_config
from app.core.schemas import Document, KeyPoint
from app.ingest.spans import RawSection, build_document
from app.keypoints.pipeline import extract_keypoints
from app.llm.base import LLMResult
from app.llm.cache import DiskCache
from app.llm.client import LLMClient

ROOT = Path(__file__).resolve().parents[2]
AXES = ["cat", "dog", "bird"]


def fake_embed(texts: list[str]) -> NDArray[np.float32]:
    return np.array(
        [[1.0 if a in t.lower() else 0.0 for a in AXES] for t in texts], dtype=np.float32
    )


class ScriptedProvider:
    name = "ollama"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)

    async def generate(self, prompt, *, model, params, seed):  # type: ignore[no-untyped-def]
        return LLMResult(
            text=self.replies.pop(0), tokens_in=1, tokens_out=1, provider="ollama", model=model
        )


def _cfg(**kp: float) -> AppConfig:
    cfg = load_config(ROOT / "configs")
    for k, val in kp.items():
        setattr(cfg.keypoints, k, val)
    return cfg


def _doc() -> Document:
    return build_document(
        "d",
        "pdf",
        "T",
        [
            RawSection(heading="Cats", level=1, paragraphs=["Cats purr softly."]),
            RawSection(heading="Pets", level=1, paragraphs=["Cats and dogs live here."]),
            RawSection(heading="Birds", level=1, paragraphs=["Birds sing."]),
        ],
    )


REPLIES = (
    '{"points": ["A cat purrs.", "The dog barks."]}',
    '{"points": ["Cats purr a lot."]}',
    '{"points": ["Many fish swim."]}',  # no source sentence about fish -> unattributed
)


class Events:
    def __init__(self) -> None:
        self.log: list[str] = []

    def opener(self) -> object:
        @contextmanager
        def cm() -> Iterator[object]:
            self.log.append("open embedder")
            yield fake_embed
            self.log.append("close embedder")

        return cm


async def _run(tmp_path: Path, cfg: AppConfig, ev: Events | None = None) -> list[KeyPoint]:
    ev = ev or Events()
    client = LLMClient(
        ScriptedProvider(*REPLIES), cfg.profiles["qwen3_8b_local"], DiskCache(tmp_path)
    )

    async def release() -> None:
        ev.log.append("release llm")

    return await extract_keypoints(
        _doc(),
        client,
        ev.opener(),  # type: ignore[arg-type]
        cfg,
        release_llm=release,
        prompt_dir=ROOT / "prompts",
    )


async def test_duplicates_collapse_and_bigger_clusters_rank_first(tmp_path: Path) -> None:
    out = await _run(tmp_path, _cfg(top_k=8))
    d = _doc()
    assert [(k.rank, k.cluster_size) for k in out] == [(1, 2), (2, 1), (3, 1)]
    assert out[0].text in {"A cat purrs.", "Cats purr a lot."}
    assert [k.text for k in out[1:]] == ["The dog barks.", "Many fish swim."]
    assert out[0].evidence is not None and d.text_of(out[0].evidence) == "Cats purr softly."
    assert out[1].evidence is not None and d.text_of(out[1].evidence) == "Cats and dogs live here."


async def test_point_unlike_any_source_sentence_is_unattributed(tmp_path: Path) -> None:
    fish = (await _run(tmp_path, _cfg()))[-1]
    assert fish.text == "Many fish swim."
    assert fish.evidence is None and fish.similarity == 0.0


async def test_top_k_cuts_the_list(tmp_path: Path) -> None:
    assert len(await _run(tmp_path, _cfg(top_k=2))) == 2


async def test_llm_released_before_the_embedder_loads_once(tmp_path: Path) -> None:
    ev = Events()
    await _run(tmp_path, _cfg(), ev)
    assert ev.log == ["release llm", "open embedder", "close embedder"]


async def test_no_candidates_loads_no_model(tmp_path: Path) -> None:
    cfg = _cfg()
    client = LLMClient(
        ScriptedProvider('{"points": []}', '{"points": []}', '{"points": []}'),
        cfg.profiles["qwen3_8b_local"],
        DiskCache(tmp_path),
    )
    ev = Events()
    out = await extract_keypoints(
        _doc(), client, ev.opener(), cfg, prompt_dir=ROOT / "prompts"  # type: ignore[arg-type]
    )
    assert out == [] and ev.log == []
