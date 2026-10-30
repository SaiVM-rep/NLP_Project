"""Worker loop on the in-memory store with a fake pipeline (FR-6.1, FR-6.2, NFR-5)."""

from __future__ import annotations

import logging
import time
from uuid import UUID

import pytest

from app.db.memory import MemoryStore
from app.db.store import Job, JobOutput
from app.worker.runner import JobContext, LeaseLost, run_once
from tests.store_contract import make_doc, make_output


async def ok_pipeline(job: Job, ctx: JobContext) -> JobOutput:
    ctx.progress("ingest", 0.1, "parsing")
    ctx.attach_document(make_doc())
    ctx.progress("summarize", 0.5, "2 chunks")
    return make_output()


async def test_no_job_returns_false() -> None:
    assert await run_once(MemoryStore(), "w", ok_pipeline, lease_s=60, max_attempts=3) is False


async def test_job_runs_to_done_with_ordered_progress() -> None:
    store = MemoryStore()
    jid = store.create_job({"url": "u"}, {})
    assert await run_once(store, "w", ok_pipeline, lease_s=60, max_attempts=3) is True
    job = store.get_job(jid)
    assert job is not None and job.status == "done" and job.doc_id is not None
    assert [e.stage for e in store.events(jid, 0)] == ["ingest", "summarize", "done"]
    r = store.result(jid)
    assert r is not None and r["doc_id"] == str(job.doc_id)


async def test_pipeline_error_fails_the_job_and_logs_doc_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def boom(job: Job, ctx: JobContext) -> JobOutput:
        ctx.attach_document(make_doc())
        raise ValueError("GROBID said no")

    store = MemoryStore()
    jid = store.create_job({}, {})
    with caplog.at_level(logging.ERROR):
        await run_once(store, "w", boom, lease_s=60, max_attempts=3)
    job = store.get_job(jid)
    assert job is not None and job.status == "failed"
    assert job.error == "ValueError: GROBID said no"
    rec = next(r for r in caplog.records if r.levelno == logging.ERROR)
    assert getattr(rec, "job_id") == str(jid)  # noqa: B009
    assert getattr(rec, "doc_id") == str(job.doc_id)  # noqa: B009


async def test_heartbeat_holds_the_lease_through_a_long_blocking_stage() -> None:
    store = MemoryStore()
    jid = store.create_job({}, {})
    stolen: list[Job | None] = []

    async def slow(job: Job, ctx: JobContext) -> JobOutput:
        time.sleep(0.8)  # blocks the event loop, like torch or GROBID would
        stolen.append(store.claim("thief", lease_s=0.3, max_attempts=3))
        return make_output()

    await run_once(store, "w", slow, lease_s=0.3, max_attempts=3, heartbeat_s=0.1)
    assert stolen == [None]
    job = store.get_job(jid)
    assert job is not None and job.status == "done"


async def test_worker_that_lost_its_lease_writes_nothing() -> None:
    """A stalled worker comes back after someone else took the job: it must stop.
    The heartbeat thread never fires here (10 s); `progress` checks ownership itself."""
    store = MemoryStore()
    jid = store.create_job({}, {})

    async def stalls(job: Job, ctx: JobContext) -> JobOutput:
        time.sleep(0.4)
        assert store.claim("w2", lease_s=0.2, max_attempts=3) is not None
        ctx.progress("summarize", 0.5, "late")
        return make_output()

    await run_once(store, "w1", stalls, lease_s=0.2, max_attempts=3, heartbeat_s=10)
    job = store.get_job(jid)
    assert job is not None and (job.status, job.locked_by) == ("running", "w2")
    assert store.events(jid, 0) == []


def test_progress_raises_once_lease_is_lost() -> None:
    store = MemoryStore()
    jid = store.create_job({}, {})
    job = store.claim("w", lease_s=60, max_attempts=3)
    assert job is not None
    ctx = JobContext(store, job, "w")
    ctx.lost.set()
    with pytest.raises(LeaseLost):
        ctx.progress("x", 0.1, "")
    assert isinstance(jid, UUID)
