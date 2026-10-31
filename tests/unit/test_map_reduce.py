"""Route B map-reduce. PRD FR-3.3, FR-3.4, FR-3.5, §11.5.

The fake provider answers from the prompt itself, so the merge tree is readable in the
root summary: E[...] = chunk summary, M[a+b] = merge of a and b.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from app.core.config import AppConfig, load_config
from app.core.schemas import Document
from app.ingest.spans import RawSection, build_document
from app.llm.base import LLMResult
from app.llm.cache import DiskCache
from app.llm.client import LLMClient
from app.llm.limiter import SlidingWindow
from app.route.tokens import estimate_tokens
from app.summarize.map_reduce import summarize_b

ROOT = Path(__file__).resolve().parents[2]


class EchoProvider:
    name = "gemini"

    def __init__(self, clock: FakeClock | None = None, delays: dict[str, float] | None = None):
        self.prompts: list[str] = []
        self.params: list[dict] = []  # type: ignore[type-arg]
        self.times: list[tuple[float, str]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.clock = clock
        self.delays = delays or {}

    async def generate(self, prompt, *, model, params, seed):  # type: ignore[no-untyped-def]
        self.prompts.append(prompt)
        self.params.append({**params, "seed": seed})
        if self.clock:
            self.times.append((self.clock.now, prompt))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if "EXCERPT:" in prompt:
                body = prompt.split("EXCERPT:\n", 1)[1].strip()
                await asyncio.sleep(self.delays.get(body.split()[0], 0))
                out = f"E[{body.split()[0]}]"
            else:
                parts = prompt.split("PARTIAL SUMMARIES:\n", 1)[1].strip().split("\n\n")
                out = "M[" + "+".join(parts) + "]"
        finally:
            self.in_flight -= 1
        return LLMResult(
            text=f'{{"summary": "{out}"}}',
            tokens_in=1,
            tokens_out=1,
            provider="gemini",
            model=model,
        )


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
        await asyncio.sleep(0)


def config(chunk_tokens: int = 40, k: int = 2) -> AppConfig:
    return load_config(
        ROOT / "configs",
        {"chunk": {"chunk_tokens": chunk_tokens, "overlap_tokens": 0, "fan_in_k": k}},
    )


def sections(n: int, words_each: int = 20) -> Document:
    """n sections, each one paragraph starting with a marker word S0, S1, ..."""
    return build_document(
        "d",
        "pdf",
        "T",
        [
            RawSection(
                heading=f"H{i}", level=1, paragraphs=[f"S{i} " + " ".join(["w"] * words_each) + "."]
            )
            for i in range(n)
        ],
    )


async def run(doc: Document, cfg: AppConfig, provider: EchoProvider, tmp_path: Path, **kw):  # type: ignore[no-untyped-def]
    client = LLMClient(provider, cfg.active_profile, DiskCache(tmp_path), **kw)
    return await summarize_b(
        doc, client, cfg, prompt_dir=ROOT / "prompts", count=lambda t: len(t.split())
    )


# ------------------------------------------------------------------ tree ---


async def test_single_chunk_needs_no_merge(tmp_path: Path) -> None:
    p = EchoProvider()
    res = await run(sections(1), config(), p, tmp_path)
    assert len(p.prompts) == 1
    assert res.summary == "E[H0]"  # heading starts the chunk text
    (root,) = res.nodes
    assert (root.level, root.parent_id, root.source_chunk_ids) == (0, None, ["c0"])


async def test_merge_tree_levels_parents_and_order(tmp_path: Path) -> None:
    p = EchoProvider()
    res = await run(sections(3), config(k=2), p, tmp_path)
    # 3 leaves -> (L0 a+b), c carried up unmerged -> root merges both
    assert res.summary == "M[M[E[H0]+E[H1]]+E[H2]]"
    assert len(p.prompts) == 3 + 2  # a group of one is carried up, not re-summarized
    by_id = {n.id: n for n in res.nodes}
    root = by_id[res.root_id]
    assert root.parent_id is None and root.level == 2
    assert root.source_chunk_ids == ["c0", "c1", "c2"]
    leaves = [n for n in res.nodes if n.level == 0]
    assert [n.source_chunk_ids for n in leaves] == [["c0"], ["c1"], ["c2"]]
    for n in res.nodes:
        if n.id != res.root_id:
            assert n.parent_id in by_id  # every node reaches the root (E3)
            assert by_id[n.parent_id].level > n.level


async def test_fan_in_k_groups(tmp_path: Path) -> None:
    p = EchoProvider()
    res = await run(sections(5), config(k=4), p, tmp_path)
    assert res.summary == "M[M[E[H0]+E[H1]+E[H2]+E[H3]]+E[H4]]"
    assert max(n.level for n in res.nodes) == 2


async def test_merge_order_is_document_order_not_completion_order(tmp_path: Path) -> None:
    # first chunk finishes last
    p = EchoProvider(delays={"H0": 0.05, "H1": 0.0})
    res = await run(sections(2), config(k=2), p, tmp_path)
    assert res.summary == "M[E[H0]+E[H1]]"


async def test_uniform_chunker_is_pluggable(tmp_path: Path) -> None:
    from app.route.chunker import chunk_uniform

    p = EchoProvider()
    cfg = config(chunk_tokens=1000)
    client = LLMClient(p, cfg.active_profile, DiskCache(tmp_path))
    res = await summarize_b(
        sections(3),
        client,
        cfg,
        prompt_dir=ROOT / "prompts",
        count=lambda t: len(t.split()),
        chunker=chunk_uniform,
    )
    assert len(res.chunks) == 1 and res.chunks[0].section_ids == ["s0", "s1", "s2"]
    assert 'section: "H0 / H1 / H2"' in p.prompts[0]


# --------------------------------------------------------------- prompts ---


def _target(prompt: str) -> int:
    m = re.search(r"about (\d+) words", prompt)
    assert m, prompt
    return int(m.group(1))


async def test_target_words_leaf_proportional_and_merge_is_sum_of_children(
    tmp_path: Path,
) -> None:
    p = EchoProvider()
    await run(sections(2, words_each=200), config(chunk_tokens=500, k=2), p, tmp_path)
    leaves = [q for q in p.prompts if "EXCERPT:" in q]
    merges = [q for q in p.prompts if "PARTIAL SUMMARIES:" in q]
    # 10% of 203 words (heading + marker + 200 + nothing) ~ 20
    assert [_target(q) for q in leaves] == [20, 20]
    assert _target(merges[0]) == 40


async def test_tiny_chunks_get_a_floor_target(tmp_path: Path) -> None:
    p = EchoProvider()
    await run(sections(1, words_each=3), config(), p, tmp_path)
    assert _target(p.prompts[0]) == 10


async def test_chunk_prompt_carries_heading_and_excerpt(tmp_path: Path) -> None:
    p = EchoProvider()
    d = sections(1)
    await run(d, config(), p, tmp_path)
    assert 'section: "H0"' in p.prompts[0]
    assert p.prompts[0].rstrip().endswith(d.full_text)


async def test_same_decoding_and_schema_as_route_a(tmp_path: Path) -> None:
    """FR-3.5: identical decoding parameters and output schema across routes."""
    p = EchoProvider()
    cfg = config(k=2)
    await run(sections(3), cfg, p, tmp_path)
    dec = cfg.decoding
    for params in p.params:
        assert (params["temperature"], params["top_p"], params["seed"]) == (
            dec.temperature,
            dec.top_p,
            dec.seed,
        )
        assert params["json_schema"]["required"] == ["summary"]


async def test_prompt_hashes_recorded(tmp_path: Path) -> None:
    res = await run(sections(2), config(k=2), EchoProvider(), tmp_path)
    import hashlib

    for name in ("chunk_summary.v1.txt", "merge.v1.txt"):
        assert (
            res.prompt_hashes[name]
            == hashlib.sha256((ROOT / "prompts" / name).read_bytes()).hexdigest()
        )


# ----------------------------------------------------- limits (FR-3.3) ---


async def test_concurrency_is_capped(tmp_path: Path) -> None:
    p = EchoProvider(delays={f"H{i}": 0.01 for i in range(8)})
    cfg = config()
    client = LLMClient(p, cfg.active_profile, DiskCache(tmp_path))
    await summarize_b(
        sections(8),
        client,
        cfg,
        prompt_dir=ROOT / "prompts",
        count=lambda t: len(t.split()),
        max_concurrency=3,
    )
    assert p.max_in_flight == 3


async def test_call_log_never_exceeds_rpm_or_tpm_in_any_minute(tmp_path: Path) -> None:
    """FR-3.3 acceptance. 25 chunks against rpm=10 / tpm=60 on a simulated clock."""
    clock = FakeClock()
    p = EchoProvider(clock=clock)
    cfg = config(k=4)
    rpm, tpm = 10, 600
    await run(
        sections(25),
        cfg,
        p,
        tmp_path,
        rpm_bucket=SlidingWindow(rpm, clock=clock, sleep=clock.sleep),
        tpm_bucket=SlidingWindow(tpm, clock=clock, sleep=clock.sleep),
        sleep=clock.sleep,
    )
    times = sorted(p.times)
    # 25 leaves; 25 -> 7 nodes (6 merges + 1 carried), 7 -> 2, 2 -> 1
    assert len(times) == 25 + 6 + 2 + 1
    for t0, _ in times:
        window = [q for t, q in times if t0 <= t < t0 + 60]
        assert len(window) <= rpm
        assert sum(estimate_tokens(q, 0.0) for q in window) <= tpm


async def test_one_failed_chunk_cancels_the_others(tmp_path: Path) -> None:
    """gather() raised but left sibling calls running: they kept spending quota after the
    run had failed, and leaked connections at exit (live run 2026-10-30)."""
    from app.summarize.common import SummaryError

    class FailFirst(EchoProvider):
        def __init__(self) -> None:
            super().__init__()
            self.finished: list[str] = []

        async def generate(self, prompt, *, model, params, seed):  # type: ignore[no-untyped-def]
            if "EXCERPT:\nH0" in prompt:
                return LLMResult(
                    text="not json", tokens_in=1, tokens_out=1, provider="gemini", model=model
                )
            await asyncio.sleep(0.2)  # slow siblings
            self.finished.append(prompt)
            return await super().generate(prompt, model=model, params=params, seed=seed)

    p = FailFirst()
    with pytest.raises(SummaryError):
        await run(sections(4), config(), p, tmp_path)
    await asyncio.sleep(0.3)
    assert p.finished == []


async def test_concurrency_defaults_to_the_active_profile(tmp_path: Path) -> None:
    """Ollama serves one request at a time: queued calls must not pile up past the timeout."""
    p = EchoProvider(delays={f"H{i}": 0.01 for i in range(8)})
    cfg = config()
    cfg.profiles[cfg.profile] = cfg.active_profile.model_copy(update={"max_concurrency": 2})
    client = LLMClient(p, cfg.active_profile, DiskCache(tmp_path))
    await summarize_b(
        sections(8), client, cfg, prompt_dir=ROOT / "prompts", count=lambda t: len(t.split())
    )
    assert p.max_in_flight == 2


async def test_word_cap_limits_merge_targets(tmp_path: Path) -> None:
    p = EchoProvider()
    cfg = load_config(
        ROOT / "configs",
        {
            "chunk": {"chunk_tokens": 500, "overlap_tokens": 0, "fan_in_k": 2},
            "summary": {"max_words": 25},
        },
    )
    await run(sections(2, words_each=200), cfg, p, tmp_path)
    merges = [q for q in p.prompts if "PARTIAL SUMMARIES:" in q]
    assert _target(merges[0]) == 25  # sum of children would be 40
