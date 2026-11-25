"""Job store: the contract the API and the worker share (FR-6.1, FR-6.2).

Two backends behave identically (`tests/store_contract.py`): `PgStore` (PRD §11.2 schema)
and `MemoryStore` (unit tests, demo mode).

Leases (DECISION-26): a claimed job is `running` with `locked_by` and `updated_at`. The
worker heartbeats; a job not touched for `lease_s` belongs to a dead worker and is claimed
again, from the start (the LLM cache makes the repeat cheap). After `max_attempts` claims
it fails with a clear error instead of looping forever.

Summary-node ids from the pipeline ("c0", "m1-0") become `uuid5(job_id, id)`: the PRD
schema keys nodes by UUID, and the mapping is stable so claims keep pointing at them.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol
from uuid import UUID, uuid5

from pydantic import BaseModel

from app.core.schemas import ClaimResult, Document, KeyPoint, SummaryNode
from app.ingest.objects import PaperObject
from app.summarize.common import SummaryResult

Status = Literal["queued", "running", "done", "failed"]


class Job(BaseModel):
    job_id: UUID
    status: Status
    stage: str | None = None
    progress: float = 0.0
    route: str | None = None
    doc_id: UUID | None = None
    input: dict[str, Any]
    config: dict[str, Any]
    error: str | None = None
    attempts: int = 0
    locked_by: str | None = None


class Event(BaseModel):
    id: int
    stage: str
    progress: float
    message: str


class JobOutput(BaseModel):
    """Everything a finished job stores. `nodes` are the verified nodes (Route A: one root)."""

    summary: SummaryResult
    nodes: list[SummaryNode]
    claims: list[ClaimResult]
    key_points: list[KeyPoint]
    metrics: dict[str, float]
    objects: list[PaperObject] = []  # tables & formulas, display only (DECISION-40)


class Store(Protocol):
    def ping(self) -> None: ...
    def create_job(self, input: dict[str, Any], config: dict[str, Any]) -> UUID: ...
    def claim(self, worker: str, *, lease_s: float, max_attempts: int) -> Job | None: ...
    def heartbeat(self, job_id: UUID, worker: str) -> bool: ...
    def progress(self, job_id: UUID, stage: str, progress: float, message: str) -> None: ...
    def attach_document(self, job_id: UUID, doc: Document) -> UUID: ...
    def finish(self, job_id: UUID, out: JobOutput) -> None: ...
    def fail(self, job_id: UUID, error: str) -> None: ...
    def get_job(self, job_id: UUID) -> Job | None: ...
    def events(self, job_id: UUID, after: int) -> list[Event]: ...
    def result(self, job_id: UUID) -> dict[str, Any] | None: ...
    def document(self, doc_id: UUID) -> Document | None: ...


def lost_error(attempts: int) -> str:
    return f"worker lost the job {attempts} times ({attempts} attempts); giving up"


def node_uuid(job_id: UUID, node_id: str | None) -> str | None:
    return None if node_id is None else str(uuid5(job_id, node_id))


def stats(out: JobOutput) -> dict[str, Any]:
    """What `jobs.stats` keeps beyond the per-row tables."""
    s = out.summary
    return {
        "summary": s.summary,
        "tokens": s.tokens,
        "tau": s.tau,
        "model": s.model,
        "prompt_hashes": s.prompt_hashes,
        "metrics": out.metrics,
        "objects": [o.model_dump() for o in out.objects],
    }


def render_result(
    job: Job,
    stats: dict[str, Any],
    nodes: list[dict[str, Any]],
    key_points: list[dict[str, Any]],
    claims: list[dict[str, Any]],
) -> dict[str, Any]:
    """PRD §11.10 result JSON. Rows are already in API form (UUID strings, span dicts)."""
    return {
        "job_id": str(job.job_id),
        "doc_id": None if job.doc_id is None else str(job.doc_id),
        "route": job.route,
        "tokens": stats["tokens"],
        "tau": stats["tau"],
        "model": stats["model"],
        "prompt_hashes": stats["prompt_hashes"],
        "summary": {"text": stats["summary"], "nodes": nodes},
        "key_points": key_points,
        "claims": claims,
        "metrics": stats["metrics"],
        "objects": stats.get("objects", []),  # jobs finished before DECISION-40 have none
    }


def output_rows(
    job_id: UUID, out: JobOutput
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """(nodes, key_points, claims) in API form, ids mapped with `node_uuid`."""
    nodes = [
        {
            "id": node_uuid(job_id, n.id),
            "level": n.level,
            "parent_id": node_uuid(job_id, n.parent_id),
            "text": n.text,
            "source_chunk_ids": n.source_chunk_ids,
        }
        for n in out.nodes
    ]
    kps = [
        {
            "rank": k.rank,
            "text": k.text,
            "cluster_size": k.cluster_size,
            "evidence": None if k.evidence is None else k.evidence.model_dump(),
            "similarity": k.similarity,
        }
        for k in out.key_points
    ]
    claims = [
        {
            "claim": c.claim,
            "summary_node_id": node_uuid(job_id, c.summary_node_id),
            "sentence_idx": c.summary_sentence_idx,
            "evidence": [e.model_dump() for e in c.evidence],
            "deberta_entail": c.deberta_entail,
            "deberta_contra": c.deberta_contra,
            "minicheck_support": c.minicheck_support,
            "label_deberta": c.label_deberta.value,
            "label_minicheck": c.label_minicheck.value,
        }
        for c in out.claims
    ]
    return nodes, kps, claims
