"""`Store` on PostgreSQL (PRD Â§11.2 + `migrations/0002_service.sql`).

One short connection per call: the worker is a single loop and the API polls; a pool
would add a dependency for no measurable gain at this scale. Times come from the database
clock (`clock_timestamp()`), so API and worker processes agree on lease expiry.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.core.schemas import Document
from app.db.store import Event, Job, JobOutput, node_uuid, render_result, stats

STALE = "status = 'running' AND updated_at < clock_timestamp() - make_interval(secs => %(lease)s)"


class PgStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    @contextmanager
    def _conn(self) -> Iterator[psycopg.Connection[dict[str, Any]]]:
        with psycopg.connect(self._dsn, row_factory=dict_row, connect_timeout=5) as conn:
            yield conn  # commits on clean exit, rolls back on error

    def ping(self) -> None:
        with self._conn() as c:
            c.execute("SELECT 1")

    def create_job(self, input: dict[str, Any], config: dict[str, Any]) -> UUID:
        jid = uuid4()
        with self._conn() as c:
            c.execute(
                "INSERT INTO jobs (job_id, status, input, config) VALUES (%s, 'queued', %s, %s)",
                (jid, Jsonb(input), Jsonb(config)),
            )
        return jid

    def claim(self, worker: str, *, lease_s: float, max_attempts: int) -> Job | None:
        p = {"lease": lease_s, "max": max_attempts, "worker": worker}
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET status = 'failed', locked_by = NULL,"
                " updated_at = clock_timestamp(),"
                " error = format('worker lost the job %%s times (%%s attempts); giving up',"
                " attempts, attempts)"
                f" WHERE {STALE} AND attempts >= %(max)s",
                p,
            )
            row = c.execute(
                "UPDATE jobs SET status = 'running', locked_by = %(worker)s,"
                " attempts = attempts + 1, updated_at = clock_timestamp()"
                " WHERE job_id = (SELECT job_id FROM jobs"
                f" WHERE status = 'queued' OR ({STALE})"
                " ORDER BY seq FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING *",
                p,
            ).fetchone()
        return None if row is None else Job(**row)

    def heartbeat(self, job_id: UUID, worker: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "UPDATE jobs SET updated_at = clock_timestamp()"
                " WHERE job_id = %s AND status = 'running' AND locked_by = %s RETURNING 1",
                (job_id, worker),
            ).fetchone()
        return row is not None

    def progress(self, job_id: UUID, stage: str, progress: float, message: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET stage = %s, progress = %s, updated_at = clock_timestamp()"
                " WHERE job_id = %s",
                (stage, progress, job_id),
            )
            c.execute(
                "INSERT INTO job_events (job_id, stage, progress, message) VALUES (%s, %s, %s, %s)",
                (job_id, stage, progress, message),
            )

    def attach_document(self, job_id: UUID, doc: Document) -> UUID:
        doc_id = uuid4()
        structure = doc.model_dump(
            mode="json", include={"sections", "sentences", "references_text"}
        )
        with self._conn() as c:
            c.execute(
                "INSERT INTO documents (doc_id, source, title, full_text, structure,"
                " structure_ok, token_count) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    doc_id,
                    doc.source,
                    doc.title,
                    doc.full_text,
                    Jsonb(structure),
                    doc.structure_ok,
                    doc.token_count,
                ),
            )
            c.execute("UPDATE jobs SET doc_id = %s WHERE job_id = %s", (doc_id, job_id))
        return doc_id

    def finish(self, job_id: UUID, out: JobOutput) -> None:
        with self._conn() as c, c.cursor() as cur:
            cur.executemany(
                "INSERT INTO summary_nodes (id, job_id, level, parent_id, text,"
                " source_chunk_ids, ord) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [
                    (
                        node_uuid(job_id, n.id),
                        job_id,
                        n.level,
                        node_uuid(job_id, n.parent_id),
                        n.text,
                        Jsonb(n.source_chunk_ids),
                        i,
                    )
                    for i, n in enumerate(out.nodes)
                ],
            )
            cur.executemany(
                "INSERT INTO key_points (id, job_id, rank, text, cluster_size, ev_start, ev_end,"
                " similarity) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    (
                        uuid4(),
                        job_id,
                        k.rank,
                        k.text,
                        k.cluster_size,
                        None if k.evidence is None else k.evidence.start,
                        None if k.evidence is None else k.evidence.end,
                        k.similarity,
                    )
                    for k in out.key_points
                ],
            )
            cur.executemany(
                "INSERT INTO claims (id, job_id, summary_node_id, sentence_idx, claim, evidence,"
                " deberta_entail, deberta_contra, minicheck_support, label_deberta,"
                " label_minicheck, ord) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    (
                        uuid4(),
                        job_id,
                        node_uuid(job_id, k.summary_node_id),
                        k.summary_sentence_idx,
                        k.claim,
                        Jsonb([e.model_dump() for e in k.evidence]),
                        k.deberta_entail,
                        k.deberta_contra,
                        k.minicheck_support,
                        k.label_deberta.value,
                        k.label_minicheck.value,
                        i,
                    )
                    for i, k in enumerate(out.claims)
                ],
            )
            cur.execute(
                "UPDATE jobs SET status = 'done', progress = 1, route = %s, stats = %s,"
                " locked_by = NULL, updated_at = clock_timestamp() WHERE job_id = %s",
                (out.summary.route.value, Jsonb(stats(out)), job_id),
            )

    def fail(self, job_id: UUID, error: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET status = 'failed', error = %s, locked_by = NULL,"
                " updated_at = clock_timestamp() WHERE job_id = %s",
                (error, job_id),
            )

    def get_job(self, job_id: UUID) -> Job | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE job_id = %s", (job_id,)).fetchone()
        return None if row is None else Job(**row)

    def events(self, job_id: UUID, after: int) -> list[Event]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, stage, progress, message FROM job_events"
                " WHERE job_id = %s AND id > %s ORDER BY id",
                (job_id, after),
            ).fetchall()
        return [Event(**r) for r in rows]

    def result(self, job_id: UUID) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE job_id = %s", (job_id,)).fetchone()
            if row is None or row["status"] != "done":
                return None
            nodes = c.execute(
                "SELECT id::text, level, parent_id::text, text, source_chunk_ids"
                " FROM summary_nodes WHERE job_id = %s ORDER BY ord",
                (job_id,),
            ).fetchall()
            kps = c.execute(
                "SELECT rank, text, cluster_size, ev_start, ev_end, similarity"
                " FROM key_points WHERE job_id = %s ORDER BY rank",
                (job_id,),
            ).fetchall()
            claims = c.execute(
                "SELECT claim, summary_node_id::text, sentence_idx, evidence, deberta_entail,"
                " deberta_contra, minicheck_support, label_deberta, label_minicheck"
                " FROM claims WHERE job_id = %s ORDER BY ord",
                (job_id,),
            ).fetchall()
        key_points = [
            {
                "rank": k["rank"],
                "text": k["text"],
                "cluster_size": k["cluster_size"],
                "evidence": (
                    None if k["ev_start"] is None else {"start": k["ev_start"], "end": k["ev_end"]}
                ),
                "similarity": k["similarity"],
            }
            for k in kps
        ]
        return render_result(Job(**row), row["stats"], nodes, key_points, claims)

    def document(self, doc_id: UUID) -> Document | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM documents WHERE doc_id = %s", (doc_id,)).fetchone()
        if row is None:
            return None
        return Document(
            doc_id=str(row["doc_id"]),
            source=row["source"],
            title=row["title"],
            full_text=row["full_text"],
            structure_ok=row["structure_ok"],
            token_count=row["token_count"],
            **row["structure"],
        )
