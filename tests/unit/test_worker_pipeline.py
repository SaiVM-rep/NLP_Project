"""Job pipeline orchestration with fake stages: order, progress, options, metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.core.config import AppConfig, load_config
from app.core.schemas import Document, KeyPoint
from app.db.memory import MemoryStore
from app.ingest.objects import PaperObject
from app.summarize.common import SummaryResult
from app.verify.pipeline import VerifiedSummary, summary_nodes
from app.worker.pipeline import make_pipeline
from app.worker.runner import run_once
from tests.store_contract import PAPER_OBJECT, make_doc, make_output

ROOT = Path(__file__).resolve().parents[2]


MAX_WORDS: list[int | None] = []  # cfg.summary.max_words each job was opened with
TOP_K: list[int | None] = []  # cfg.verify.minicheck.top_k_chunks per job


class FakeStages:
    def __init__(self, cfg: AppConfig, log: list[Any]) -> None:
        self.log = log
        log.append(("open", cfg.profile))
        MAX_WORDS.append(cfg.summary.max_words)
        TOP_K.append(cfg.verify.minicheck.top_k_chunks)
        self.out = make_output()

    def ingest(self, input: dict[str, Any]) -> Document:
        self.log.append(("ingest", input))
        return make_doc()

    def objects(self, input: dict[str, Any]) -> list[PaperObject]:
        self.log.append(("objects", input["path"]))
        return [PAPER_OBJECT]

    async def summarize(self, doc: Document, tau: int | None) -> SummaryResult:
        self.log.append(("summarize", tau))
        return self.out.summary

    async def keypoints(self, doc: Document) -> list[KeyPoint]:
        self.log.append(("keypoints",))
        return self.out.key_points

    async def verify(
        self, doc: Document, result: SummaryResult, *, final_only: bool = False
    ) -> VerifiedSummary:
        self.log.append(("verify",) if not final_only else ("verify", "final"))
        return VerifiedSummary(nodes=self.out.nodes, claims=self.out.claims)

    async def aclose(self) -> None:
        self.log.append(("close",))


def _run(
    options: dict[str, Any],
    log: list[Any],
    stages: Any = FakeStages,
    input: dict[str, Any] | None = None,
) -> MemoryStore:
    import asyncio

    store = MemoryStore()
    store.create_job(input or {"url": "https://x"}, options)
    pipe = make_pipeline(lambda o: load_config(ROOT / "configs", o), lambda cfg: stages(cfg, log))
    asyncio.run(run_once(store, "w", pipe, lease_s=60, max_attempts=3))
    return store


def _only_job(store: MemoryStore) -> Any:
    (jid,) = store._jobs
    return jid


def test_stages_run_in_order_with_progress_and_close() -> None:
    log: list[Any] = []
    store = _run({}, log)
    jid = _only_job(store)
    assert log == [
        ("open", load_config(ROOT / "configs").profile),
        ("ingest", {"url": "https://x"}),
        ("summarize", None),
        ("keypoints",),
        ("verify", "final"),  # job default: the summary the user reads (DECISION-37)
        ("close",),
    ]
    ev = store.events(jid, 0)
    assert [e.stage for e in ev] == ["ingest", "summarize", "keypoints", "verify", "done"]
    assert [e.progress for e in ev] == sorted(e.progress for e in ev)
    assert "route B" in ev[2].message or "route B" in ev[1].message


def test_result_carries_metrics_and_everything_stored() -> None:
    store = _run({}, [])
    r = store.result(_only_job(store))
    assert r is not None
    assert r["metrics"]["faithfulness_deberta"] == 1.0
    assert r["metrics"]["faithfulness_minicheck"] == 1.0
    # summary "Cats purr and dogs bark." = 24 chars ~ 6 tokens over 7 source tokens
    assert 0 < r["metrics"]["compression_ratio"] < 1.5
    assert len(r["key_points"]) == 2 and len(r["claims"]) == 1


def test_options_pick_profile_and_tau() -> None:
    log: list[Any] = []
    _run({"profile": "qwen3_8b_local", "tau": 1000}, log)
    assert log[0] == ("open", "qwen3_8b_local") and ("summarize", 1000) in log


def test_verify_and_keypoints_can_be_switched_off() -> None:
    log: list[Any] = []
    store = _run({"verify": False, "keypoints": False}, log)
    assert not any(e[0] in ("verify", "keypoints") for e in log)
    r = store.result(_only_job(store))
    assert r is not None and r["claims"] == [] and r["key_points"] == []
    assert "faithfulness_minicheck" not in r["metrics"]  # no claims: no number, not NaN
    want = summary_nodes(make_output().summary)
    assert len(r["summary"]["nodes"]) == len(want)


def test_stage_failure_fails_job_and_still_closes() -> None:
    class Broken(FakeStages):
        async def summarize(self, doc: Document, tau: int | None) -> SummaryResult:
            raise RuntimeError("quota")

    log: list[Any] = []
    store = _run({}, log, Broken)
    job = store.get_job(_only_job(store))
    assert job is not None and (job.status, job.error) == ("failed", "RuntimeError: quota")
    assert log[-1] == ("close",)


@pytest.mark.parametrize("bad", [{"profile": "nope"}, {"tau": 0}, {"colour": "red"}])
def test_bad_options_fail_the_job_cleanly(bad: dict[str, Any]) -> None:
    store = _run(bad, [])
    job = store.get_job(_only_job(store))
    assert job is not None and job.status == "failed" and job.error


def test_live_stages_free_the_gpu_when_the_job_ends() -> None:
    # Ollama now keeps the model loaded between calls; a finished job must not leave
    # Qwen on the 8 GB GPU for the next job's verifiers (DECISION-3).
    import asyncio

    from app.worker.pipeline import LiveStages

    log: list[str] = []

    class FakeProvider:
        async def unload(self, model: str) -> None:
            log.append(f"unload {model}")

        async def aclose(self) -> None:
            log.append("aclose")

    st = LiveStages.__new__(LiveStages)
    st.cfg = load_config(ROOT / "configs")
    st.provider = FakeProvider()  # type: ignore[assignment]
    asyncio.run(st.aclose())
    assert log == [f"unload {st.cfg.active_profile.model}", "aclose"]


def test_uploads_cap_the_summary_and_verify_only_the_final_summary() -> None:
    # DECISION-37: job defaults are the fast product path; experiments call LiveStages
    # directly and keep the 10% rule and per-level verification.
    log: list[Any] = []
    MAX_WORDS.clear()
    _run({}, log)
    assert MAX_WORDS == [500] and ("verify", "final") in log
    log = []
    _run({"max_summary_words": None, "verify_scope": "all"}, log)
    assert MAX_WORDS == [500, None] and ("verify",) in log


def test_uploads_check_each_claim_against_its_top_5_chunks() -> None:
    TOP_K.clear()
    _run({}, [])
    _run({"minicheck_top_k": None}, [])
    _run({"minicheck_top_k": 2}, [])
    assert TOP_K == [5, None, 2]


def test_pdf_jobs_extract_tables_and_formulas_right_after_ingest() -> None:
    log: list[Any] = []
    store = _run({}, log, input={"path": "p.pdf"})
    names = [e[0] for e in log]
    assert names[names.index("ingest") + 1] == "objects" and ("objects", "p.pdf") in log
    jid = _only_job(store)
    assert "tables" in [e.stage for e in store.events(jid, 0)]
    r = store.result(jid)
    assert r is not None and r["objects"] == [PAPER_OBJECT.model_dump()]


def test_tables_skipped_for_web_pages_and_when_switched_off() -> None:
    log: list[Any] = []
    _run({}, log)
    _run({"tables": False}, log, input={"path": "p.pdf"})
    assert not any(e[0] == "objects" for e in log)


def test_live_stages_trim_ram_before_docling_and_after_the_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import app.ingest.objects as objects_mod
    import app.worker.pipeline as wp

    events: list[str] = []
    monkeypatch.setattr(wp, "trim_working_set", lambda: events.append("trim") or True)
    monkeypatch.setattr(
        objects_mod, "extract_objects", lambda *a, **k: events.append("docling") or []
    )

    class P:
        async def aclose(self) -> None:
            events.append("aclose")

    st = wp.LiveStages.__new__(wp.LiveStages)
    st.cfg = load_config(ROOT / "configs")
    st.provider = P()  # type: ignore[assignment]
    st.objects({"path": "p.pdf"})
    asyncio.run(st.aclose())
    assert events[:2] == ["trim", "docling"] and events[-1] == "trim"
