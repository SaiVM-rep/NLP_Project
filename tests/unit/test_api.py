"""HTTP API (PRD §11.10, FR-6.1, FR-6.3) on the in-memory store. No docker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.db.memory import MemoryStore
from tests.store_contract import make_doc, make_output

PDF = b"%PDF-1.4\n" + b"x" * 100


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def client(store: MemoryStore, tmp_path: Path) -> TestClient:
    checks = {"db": store.ping, "grobid": lambda: None}
    return TestClient(
        create_app(store, upload_dir=tmp_path, checks=checks, max_upload_mb=0.001, poll_s=0.01)
    )


def _problem(r: Any, status: int, code: str) -> dict[str, Any]:
    assert r.status_code == status, r.text
    assert r.headers["content-type"].startswith("application/problem+json")
    body: dict[str, Any] = r.json()
    assert body["status"] == status and body["code"] == code and "doc_id" in body
    return body


# ------------------------------------------------------------------ create ---


def test_create_from_url_queues_a_job(client: TestClient, store: MemoryStore) -> None:
    r = client.post("/v1/jobs", json={"url": "https://example.org/a", "options": {"tau": 900}})
    assert r.status_code == 202
    jid = UUID(r.json()["job_id"])
    assert r.headers["location"] == f"/v1/jobs/{jid}"
    job = store.get_job(jid)
    assert job is not None and job.status == "queued"
    assert job.input == {"url": "https://example.org/a"}
    assert job.config == {
        "profile": None,
        "tau": 900,
        "keypoints": True,
        "verify": True,
        "verify_scope": "final",
        "max_summary_words": 500,
        "minicheck_top_k": 5,
    }


def test_create_from_pdf_upload_stores_the_file(
    client: TestClient, store: MemoryStore, tmp_path: Path
) -> None:
    r = client.post(
        "/v1/jobs",
        files={"file": ("paper.pdf", PDF, "application/pdf")},
        data={"options": json.dumps({"verify": False})},
    )
    assert r.status_code == 202
    job = store.get_job(UUID(r.json()["job_id"]))
    assert job is not None and job.config["verify"] is False
    saved = Path(job.input["path"])
    assert saved.parent == tmp_path and saved.read_bytes() == PDF
    assert job.input["filename"] == "paper.pdf"


def test_upload_over_the_limit_is_413(client: TestClient) -> None:
    big = b"%PDF" + b"x" * 2000  # limit is 0.001 MB = 1048 bytes here
    r = client.post("/v1/jobs", files={"file": ("big.pdf", big, "application/pdf")})
    _problem(r, 413, "too_large")


def test_non_pdf_upload_is_415(client: TestClient) -> None:
    r = client.post("/v1/jobs", files={"file": ("a.pdf", b"hello", "application/pdf")})
    _problem(r, 415, "not_pdf")


def test_other_content_type_is_415(client: TestClient) -> None:
    r = client.post("/v1/jobs", content=b"x", headers={"content-type": "text/plain"})
    _problem(r, 415, "unsupported_media_type")


@pytest.mark.parametrize(
    "body",
    [
        {"url": "ftp://x"},
        {"url": "https://x", "options": {"colour": "red"}},
        {"url": "https://x", "options": {"tau": -1}},
        {},
    ],
)
def test_bad_request_is_422(client: TestClient, body: dict[str, Any]) -> None:
    _problem(client.post("/v1/jobs", json=body), 422, "invalid_request")


def test_unknown_profile_is_422(client: TestClient) -> None:
    r = client.post("/v1/jobs", json={"url": "https://x", "options": {"profile": "nope"}})
    _problem(r, 422, "invalid_request")


# ------------------------------------------------------------ status/result ---


def test_status(client: TestClient, store: MemoryStore) -> None:
    jid = store.create_job({"url": "u"}, {})
    store.claim("w", lease_s=60, max_attempts=3)
    store.progress(jid, "summarize", 0.4, "")
    body = client.get(f"/v1/jobs/{jid}").json()
    assert body | {"job_id": "x"} == {
        "job_id": "x",
        "status": "running",
        "stage": "summarize",
        "progress": 0.4,
        "route": None,
        "doc_id": None,
        "error": None,
        "attempts": 1,
    }


def test_unknown_job_is_404(client: TestClient) -> None:
    for path in ("", "/result", "/events"):
        _problem(client.get(f"/v1/jobs/{uuid4()}{path}"), 404, "job_not_found")
    _problem(client.get("/v1/jobs/not-a-uuid"), 422, "invalid_request")


def test_result_before_done_is_409_then_200(client: TestClient, store: MemoryStore) -> None:
    jid = store.create_job({}, {})
    store.claim("w", lease_s=60, max_attempts=3)
    doc_id = store.attach_document(jid, make_doc())
    body = _problem(client.get(f"/v1/jobs/{jid}/result"), 409, "job_not_done")
    assert body["doc_id"] == str(doc_id)
    store.finish(jid, make_output())
    r = client.get(f"/v1/jobs/{jid}/result")
    assert r.status_code == 200 and r.json()["route"] == "B"


def test_failed_job_result_is_409_with_error(client: TestClient, store: MemoryStore) -> None:
    jid = store.create_job({}, {})
    store.claim("w", lease_s=60, max_attempts=3)
    store.fail(jid, "IngestError: not a PDF")
    body = _problem(client.get(f"/v1/jobs/{jid}/result"), 409, "job_failed")
    assert "not a PDF" in body["detail"]


# ------------------------------------------------------------------ events ---


def _sse(text: str) -> list[dict[str, Any]]:
    out, cur = [], {}
    for line in text.splitlines():
        if not line.strip():
            if cur:
                out.append(cur)
            cur = {}
        elif ":" in line and not line.startswith(":"):
            k, v = line.split(":", 1)
            cur[k] = v.strip()
    if cur:
        out.append(cur)
    return out


def test_events_stream_in_order_and_end_when_job_ends(
    client: TestClient, store: MemoryStore
) -> None:
    jid = store.create_job({}, {})
    store.claim("w", lease_s=60, max_attempts=3)
    store.progress(jid, "ingest", 0.05, "")
    store.progress(jid, "summarize", 0.15, "9 chars")
    store.fail(jid, "boom")
    with client.stream("GET", f"/v1/jobs/{jid}/events") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = _sse(r.read().decode())
    progress = [e for e in events if e.get("event") == "progress"]
    assert [json.loads(e["data"]) for e in progress] == [
        {"stage": "ingest", "progress": 0.05, "message": ""},
        {"stage": "summarize", "progress": 0.15, "message": "9 chars"},
    ]
    assert int(progress[0]["id"]) < int(progress[1]["id"])
    end = events[-1]
    assert end["event"] == "end" and json.loads(end["data"])["status"] == "failed"


def test_events_resume_after_last_event_id(client: TestClient, store: MemoryStore) -> None:
    jid = store.create_job({}, {})
    store.claim("w", lease_s=60, max_attempts=3)
    store.progress(jid, "ingest", 0.05, "")
    store.progress(jid, "summarize", 0.15, "")
    first = store.events(jid, 0)[0].id
    store.finish(jid, make_output())
    with client.stream("GET", f"/v1/jobs/{jid}/events", headers={"last-event-id": str(first)}) as r:
        events = _sse(r.read().decode())
    stages = [json.loads(e["data"])["stage"] for e in events if e.get("event") == "progress"]
    assert stages == ["summarize"]


# ---------------------------------------------------------------- document ---


def test_document_text_has_full_text_and_spans(client: TestClient, store: MemoryStore) -> None:
    jid = store.create_job({}, {})
    store.claim("w", lease_s=60, max_attempts=3)
    doc_id = store.attach_document(jid, make_doc())
    body = client.get(f"/v1/documents/{doc_id}/text").json()
    d = make_doc()
    assert body["full_text"] == d.full_text and body["title"] == "Pets"
    assert body["sections"] == [
        {"id": "s0", "heading": "Intro", "level": 1, "span": {"start": 0, "end": 28}}
    ]
    assert body["sentences"] == [{"start": 0, "end": 10}, {"start": 11, "end": 28}]
    _problem(client.get(f"/v1/documents/{uuid4()}/text"), 404, "document_not_found")


# ----------------------------------------------------------------- healthz ---


def test_healthz_ok_and_failing(store: MemoryStore, tmp_path: Path) -> None:
    ok = TestClient(create_app(store, upload_dir=tmp_path, checks={"db": lambda: None}))
    assert ok.get("/healthz").json() == {"status": "ok", "checks": {"db": "ok"}}

    def down() -> None:
        raise ConnectionError("refused")

    bad = TestClient(
        create_app(store, upload_dir=tmp_path, checks={"db": lambda: None, "ollama": down})
    )
    r = bad.get("/healthz")
    assert r.status_code == 503
    assert r.json() == {
        "status": "error",
        "checks": {"db": "ok", "ollama": "error: ConnectionError: refused"},
    }
