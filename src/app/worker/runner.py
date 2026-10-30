"""Worker loop: claim one job, run the pipeline under a lease, store the outcome (FR-6.2).

The heartbeat is a thread, not a task: pipeline stages block the event loop (GROBID calls,
torch on the GPU), and a task would starve exactly when the lease matters. Every progress
report also re-checks ownership, so a worker that stalled past its lease and was replaced
stops at its next report instead of writing over the new owner's work.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Awaitable, Callable
from uuid import UUID

from app.core.schemas import Document
from app.db.store import Job, JobOutput, Store

log = logging.getLogger("app.worker")


class LeaseLost(RuntimeError):
    """Another worker owns the job now; this one must write nothing more."""


class JobContext:
    def __init__(self, store: Store, job: Job, worker: str) -> None:
        self.store, self.job, self.worker = store, job, worker
        self.lost = threading.Event()
        self.doc_id: UUID | None = None

    def _check(self) -> None:
        if self.lost.is_set() or not self.store.heartbeat(self.job.job_id, self.worker):
            self.lost.set()
            raise LeaseLost(f"job {self.job.job_id} was taken over")

    def progress(self, stage: str, progress: float, message: str = "") -> None:
        self._check()
        self.store.progress(self.job.job_id, stage, progress, message)

    def attach_document(self, doc: Document) -> UUID:
        self._check()
        self.doc_id = self.store.attach_document(self.job.job_id, doc)
        return self.doc_id


Pipeline = Callable[[Job, JobContext], Awaitable[JobOutput]]


def _heartbeat(ctx: JobContext, every_s: float, stop: threading.Event) -> None:
    while not stop.wait(every_s):
        if not ctx.store.heartbeat(ctx.job.job_id, ctx.worker):
            ctx.lost.set()
            return


async def run_once(
    store: Store,
    worker: str,
    pipeline: Pipeline,
    *,
    lease_s: float,
    max_attempts: int,
    heartbeat_s: float | None = None,
) -> bool:
    """Process at most one job. False when the queue was empty."""
    job = store.claim(worker, lease_s=lease_s, max_attempts=max_attempts)
    if job is None:
        return False
    ctx = JobContext(store, job, worker)
    stop = threading.Event()
    beat = threading.Thread(
        target=_heartbeat, args=(ctx, heartbeat_s or lease_s / 3, stop), daemon=True
    )
    beat.start()
    try:
        out = await pipeline(job, ctx)
        ctx.progress("done", 1.0, "")
        store.finish(job.job_id, out)
    except LeaseLost:
        log.warning("lease lost; leaving the job to its new owner", extra=_ids(ctx))
    except Exception as e:  # any stage failure ends the job cleanly (FR-6.2)
        if ctx.lost.is_set():
            log.warning("lease lost; leaving the job to its new owner", extra=_ids(ctx))
        else:
            log.exception("job failed", extra=_ids(ctx))
            store.fail(job.job_id, f"{type(e).__name__}: {e}")
    finally:
        stop.set()
        beat.join()
    return True


def _ids(ctx: JobContext) -> dict[str, str | None]:
    return {
        "job_id": str(ctx.job.job_id),
        "doc_id": None if ctx.doc_id is None else str(ctx.doc_id),
        "worker": ctx.worker,
    }
