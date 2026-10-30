"""In-process `Store`: unit tests for the API and worker, and the precomputed demo (P7).

Same behaviour as `PgStore` (shared contract tests). Thread-safe, since the API serves
sync handlers from a thread pool.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4

from app.core.schemas import Document
from app.db.store import (
    Event,
    Job,
    JobOutput,
    lost_error,
    output_rows,
    render_result,
    stats,
)


class MemoryStore:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._jobs: dict[UUID, Job] = {}
        self._order: list[UUID] = []
        self._touched: dict[UUID, float] = {}
        self._events: dict[UUID, list[Event]] = {}
        self._next_event = 1
        self._docs: dict[UUID, Document] = {}
        self._results: dict[UUID, dict[str, Any]] = {}

    def ping(self) -> None:
        return None

    def create_job(self, input: dict[str, Any], config: dict[str, Any]) -> UUID:
        jid = uuid4()
        with self._lock:
            self._jobs[jid] = Job(job_id=jid, status="queued", input=input, config=config)
            self._order.append(jid)
            self._touched[jid] = self._clock()
        return jid

    def claim(self, worker: str, *, lease_s: float, max_attempts: int) -> Job | None:
        now = self._clock()
        with self._lock:
            for jid in self._order:
                job = self._jobs[jid]
                stale = job.status == "running" and now - self._touched[jid] > lease_s
                if stale and job.attempts >= max_attempts:
                    job.status, job.locked_by = "failed", None
                    job.error = lost_error(job.attempts)
            for jid in self._order:
                job = self._jobs[jid]
                stale = job.status == "running" and now - self._touched[jid] > lease_s
                if job.status == "queued" or stale:
                    job.status, job.locked_by = "running", worker
                    job.attempts += 1
                    self._touched[jid] = now
                    return job.model_copy()
        return None

    def heartbeat(self, job_id: UUID, worker: str) -> bool:
        with self._lock:
            job = self._jobs[job_id]
            if job.status != "running" or job.locked_by != worker:
                return False
            self._touched[job_id] = self._clock()
            return True

    def progress(self, job_id: UUID, stage: str, progress: float, message: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.stage, job.progress = stage, progress
            self._touched[job_id] = self._clock()
            ev = Event(id=self._next_event, stage=stage, progress=progress, message=message)
            self._next_event += 1
            self._events.setdefault(job_id, []).append(ev)

    def attach_document(self, job_id: UUID, doc: Document) -> UUID:
        doc_id = uuid4()
        with self._lock:
            self._docs[doc_id] = doc.model_copy(update={"doc_id": str(doc_id)})
            self._jobs[job_id].doc_id = doc_id
        return doc_id

    def finish(self, job_id: UUID, out: JobOutput) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status, job.progress, job.route = "done", 1.0, out.summary.route.value
            job.locked_by = None
            self._results[job_id] = render_result(job, stats(out), *output_rows(job_id, out))

    def fail(self, job_id: UUID, error: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status, job.error, job.locked_by = "failed", error, None

    def get_job(self, job_id: UUID) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else job.model_copy()

    def events(self, job_id: UUID, after: int) -> list[Event]:
        with self._lock:
            return [e for e in self._events.get(job_id, []) if e.id > after]

    def result(self, job_id: UUID) -> dict[str, Any] | None:
        with self._lock:
            return self._results.get(job_id)

    def document(self, doc_id: UUID) -> Document | None:
        with self._lock:
            return self._docs.get(doc_id)
