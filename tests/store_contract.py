"""Behaviour every `Store` must have (FR-6.1, FR-6.2). Subclassed once per backend:
`tests/unit/test_memory_store.py` (no docker) and `tests/integration/test_pg_store.py`.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.core.schemas import (
    ClaimLabel,
    ClaimResult,
    Document,
    KeyPoint,
    Route,
    Section,
    Span,
    SummaryNode,
)
from app.db.store import JobOutput, Store
from app.ingest.objects import PaperObject
from app.summarize.common import SummaryResult

TEXT = "Cats purr. Dogs bark loudly."


def make_doc(doc_id: str = "ignored") -> Document:
    return Document(
        doc_id=doc_id,
        source="pdf",
        title="Pets",
        full_text=TEXT,
        sections=[
            Section(
                id="s0",
                heading="Intro",
                level=1,
                span=Span(start=0, end=28),
                paragraphs=[Span(start=0, end=28)],
            )
        ],
        sentences=[Span(start=0, end=10), Span(start=11, end=28)],
        token_count=7,
    )


PAPER_OBJECT = PaperObject(
    kind="table", page=2, caption="Table 1: Pets.", content="| a |\n|---|\n| 1 |", image_png="AA=="
)


def make_output() -> JobOutput:
    nodes = [
        SummaryNode(id="c0", level=0, parent_id="m1-0", text="Cats purr.", source_chunk_ids=["c0"]),
        SummaryNode(id="c1", level=0, parent_id="m1-0", text="Dogs bark.", source_chunk_ids=["c1"]),
        SummaryNode(
            id="m1-0", level=1, text="Cats purr and dogs bark.", source_chunk_ids=["c0", "c1"]
        ),
    ]
    summary = SummaryResult(
        doc_id="d",
        summary="Cats purr and dogs bark.",
        route=Route.B,
        tokens=7,
        tau=5,
        prompt_hashes={"merge.v1.txt": "ab"},
        model="qwen3:8b",
        nodes=nodes,
        root_id="m1-0",
    )
    claim = ClaimResult(
        claim="Cats purr.",
        summary_sentence_idx=0,
        summary_node_id="m1-0",
        evidence=[Span(start=0, end=10)],
        deberta_entail=0.9,
        deberta_contra=0.0,
        minicheck_support=0.95,
        label_deberta=ClaimLabel.SUPPORTED,
        label_minicheck=ClaimLabel.SUPPORTED,
    )
    kps = [
        KeyPoint(
            text="Cats purr.",
            rank=1,
            cluster_size=2,
            evidence=Span(start=0, end=10),
            similarity=0.9,
        ),
        KeyPoint(text="Fish swim.", rank=2, cluster_size=1, evidence=None, similarity=0.2),
    ]
    return JobOutput(
        summary=summary,
        nodes=nodes,
        claims=[claim],
        key_points=kps,
        metrics={"faithfulness_minicheck": 1.0, "compression_ratio": 0.8},
        objects=[PAPER_OBJECT],
    )


class StoreContract:
    @pytest.fixture
    def store(self) -> Store:  # overridden per backend
        raise NotImplementedError

    def test_new_job_is_queued(self, store: Store) -> None:
        jid = store.create_job({"url": "https://x"}, {"profile": "p"})
        job = store.get_job(jid)
        assert job is not None
        assert (job.status, job.progress, job.attempts) == ("queued", 0.0, 0)
        assert job.input == {"url": "https://x"} and job.config == {"profile": "p"}

    def test_unknown_job_is_none(self, store: Store) -> None:
        assert store.get_job(uuid4()) is None
        assert store.result(uuid4()) is None

    def test_claim_takes_oldest_queued_once(self, store: Store) -> None:
        a = store.create_job({"n": 1}, {})
        b = store.create_job({"n": 2}, {})
        j1 = store.claim("w1", lease_s=60, max_attempts=3)
        j2 = store.claim("w2", lease_s=60, max_attempts=3)
        assert j1 is not None and j2 is not None
        assert (j1.job_id, j2.job_id) == (a, b)
        assert (j1.status, j1.locked_by, j1.attempts) == ("running", "w1", 1)
        assert store.claim("w3", lease_s=60, max_attempts=3) is None

    def test_progress_is_recorded_as_ordered_events(self, store: Store) -> None:
        jid = store.create_job({}, {})
        store.claim("w", lease_s=60, max_attempts=3)
        store.progress(jid, "ingest", 0.1, "parsing")
        store.progress(jid, "summarize", 0.4, "3 chunks")
        job = store.get_job(jid)
        assert job is not None and (job.stage, job.progress) == ("summarize", 0.4)
        ev = store.events(jid, after=0)
        assert [(e.stage, e.progress, e.message) for e in ev] == [
            ("ingest", 0.1, "parsing"),
            ("summarize", 0.4, "3 chunks"),
        ]
        assert ev[0].id < ev[1].id
        assert store.events(jid, after=ev[0].id) == ev[1:]

    def test_stale_running_job_is_reclaimed_and_counts_attempts(self, store: Store) -> None:
        """FR-6.2: a worker that dies stops heartbeating; after the lease another takes over."""
        jid = store.create_job({}, {})
        assert store.claim("dead", lease_s=0.3, max_attempts=3) is not None
        assert store.claim("w2", lease_s=0.3, max_attempts=3) is None  # lease still held
        time.sleep(0.5)
        job = store.claim("w2", lease_s=0.3, max_attempts=3)
        assert job is not None and (job.job_id, job.locked_by, job.attempts) == (jid, "w2", 2)

    def test_heartbeat_keeps_the_lease_and_reports_a_lost_one(self, store: Store) -> None:
        jid = store.create_job({}, {})
        store.claim("w1", lease_s=0.4, max_attempts=3)
        for _ in range(3):
            time.sleep(0.2)
            assert store.heartbeat(jid, "w1") is True
        assert store.claim("w2", lease_s=0.4, max_attempts=3) is None
        time.sleep(0.6)
        assert store.claim("w2", lease_s=0.4, max_attempts=3) is not None
        assert store.heartbeat(jid, "w1") is False  # w1 must stop: someone else owns it

    def test_job_that_kept_losing_its_worker_fails_cleanly(self, store: Store) -> None:
        jid = store.create_job({}, {})
        store.claim("a", lease_s=0.2, max_attempts=2)
        time.sleep(0.3)
        store.claim("b", lease_s=0.2, max_attempts=2)
        time.sleep(0.3)
        assert store.claim("c", lease_s=0.2, max_attempts=2) is None
        job = store.get_job(jid)
        assert job is not None and job.status == "failed"
        assert job.error is not None and "2 attempts" in job.error

    def test_fail_records_error(self, store: Store) -> None:
        jid = store.create_job({}, {})
        store.claim("w", lease_s=60, max_attempts=3)
        store.fail(jid, "GROBID unreachable")
        job = store.get_job(jid)
        assert job is not None and (job.status, job.error) == ("failed", "GROBID unreachable")
        assert store.claim("w", lease_s=0, max_attempts=3) is None  # failed is never retaken

    def test_document_round_trips_with_spans(self, store: Store) -> None:
        jid = store.create_job({}, {})
        store.claim("w", lease_s=60, max_attempts=3)
        doc_id = store.attach_document(jid, make_doc())
        job = store.get_job(jid)
        assert job is not None and job.doc_id == doc_id
        got = store.document(doc_id)
        assert got is not None
        assert got.model_dump(exclude={"doc_id"}) == make_doc().model_dump(exclude={"doc_id"})
        assert got.doc_id == str(doc_id)
        assert store.document(uuid4()) is None

    def test_finish_then_result_has_the_api_shape(self, store: Store) -> None:
        jid = store.create_job({}, {})
        store.claim("w", lease_s=60, max_attempts=3)
        store.attach_document(jid, make_doc())
        store.finish(jid, make_output())
        job = store.get_job(jid)
        assert job is not None and (job.status, job.progress, job.route) == ("done", 1.0, "B")
        r: dict[str, Any] = store.result(jid)  # type: ignore[assignment]
        assert r["job_id"] == str(jid) and r["doc_id"] == str(job.doc_id)
        assert (r["route"], r["tokens"], r["tau"]) == ("B", 7, 5)
        assert r["summary"]["text"] == "Cats purr and dogs bark."
        nodes = r["summary"]["nodes"]
        ids = {n["text"]: n["id"] for n in nodes}
        assert all(UUID(i) for i in ids.values())
        by_text = {n["text"]: n for n in nodes}
        assert by_text["Cats purr."]["parent_id"] == ids["Cats purr and dogs bark."]
        assert by_text["Cats purr and dogs bark."] | {"id": "x"} == {
            "id": "x",
            "level": 1,
            "parent_id": None,
            "text": "Cats purr and dogs bark.",
            "source_chunk_ids": ["c0", "c1"],
        }
        assert r["key_points"] == [
            {
                "rank": 1,
                "text": "Cats purr.",
                "cluster_size": 2,
                "evidence": {"start": 0, "end": 10},
                "similarity": pytest.approx(0.9),
            },
            {
                "rank": 2,
                "text": "Fish swim.",
                "cluster_size": 1,
                "evidence": None,
                "similarity": pytest.approx(0.2),
            },
        ]
        (c,) = r["claims"]
        assert (
            c["claim"] == "Cats purr." and c["summary_node_id"] == ids["Cats purr and dogs bark."]
        )
        assert c["evidence"] == [{"start": 0, "end": 10}]
        assert (c["label_deberta"], c["label_minicheck"]) == ("supported", "supported")
        assert c["minicheck_support"] == pytest.approx(0.95)
        assert r["metrics"] == {"faithfulness_minicheck": 1.0, "compression_ratio": 0.8}
        assert r["objects"] == [PAPER_OBJECT.model_dump()]

    def test_result_is_none_until_done(self, store: Store) -> None:
        jid = store.create_job({}, {})
        store.claim("w", lease_s=60, max_attempts=3)
        assert store.result(jid) is None
