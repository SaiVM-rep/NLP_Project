"""Job pipeline orchestration with fake stages: order, progress, options, metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.core.config import AppConfig, load_config
from app.core.schemas import Document, KeyPoint
from app.db.memory import MemoryStore
from app.summarize.common import SummaryResult
from app.verify.pipeline import VerifiedSummary, summary_nodes
from app.worker.pipeline import make_pipeline
from app.worker.runner import run_once
from tests.store_contract import make_doc, make_output

ROOT = Path(__file__).resolve().parents[2]


class FakeStages:
    def __init__(self, cfg: AppConfig, log: list[Any]) -> None:
        self.log = log
        log.append(("open", cfg.profile))
        self.out = make_output()

    def ingest(self, input: dict[str, Any]) -> Document:
        self.log.append(("ingest", input))
        return make_doc()

    async def summarize(self, doc: Document, tau: int | None) -> SummaryResult:
        self.log.append(("summarize", tau))
        return self.out.summary

    async def keypoints(self, doc: Document) -> list[KeyPoint]:
        self.log.append(("keypoints",))
        return self.out.key_points

    async def verify(self, doc: Document, result: SummaryResult) -> VerifiedSummary:
        self.log.append(("verify",))
        return VerifiedSummary(nodes=self.out.nodes, claims=self.out.claims)

    async def aclose(self) -> None:
        self.log.append(("close",))


def _run(options: dict[str, Any], log: list[Any], stages: Any = FakeStages) -> MemoryStore:
    import asyncio

    store = MemoryStore()
    store.create_job({"url": "https://x"}, options)
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
        ("verify",),
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
    assert ("verify",) not in log and ("keypoints",) not in log
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
