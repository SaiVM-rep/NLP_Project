"""HTTP API, PRD Â§11.10 (FR-6.1, FR-6.3). `uvicorn app.api.main:app`.

The API only queues and reads; the worker (`python -m app.worker`) does the processing.
Errors are RFC 9457 problem details carrying `code` and `doc_id` (NFR-5).
Store calls are blocking (psycopg), so async handlers run them in a thread.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sse_starlette.sse import EventSourceResponse
from starlette.datastructures import UploadFile

from app.db.store import Store
from app.worker.pipeline import JobOptions

ROOT = Path(__file__).resolve().parents[3]
TERMINAL = {"done", "failed"}


class ApiProblem(Exception):
    def __init__(
        self, status: int, code: str, title: str, detail: str | None = None, doc_id: Any = None
    ) -> None:
        super().__init__(title)
        self.status, self.code, self.title, self.detail = status, code, title, detail
        self.doc_id = None if doc_id is None else str(doc_id)


def _problem(p: ApiProblem) -> JSONResponse:
    body = {
        "type": "about:blank",
        "title": p.title,
        "status": p.status,
        "code": p.code,
        "detail": p.detail,
        "doc_id": p.doc_id,
    }
    return JSONResponse(body, status_code=p.status, media_type="application/problem+json")


class CreateJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    options: JobOptions = Field(default_factory=JobOptions)

    @field_validator("url")
    @classmethod
    def _http(cls, v: str) -> str:
        if not v.lower().startswith(("http://", "https://")):
            raise ValueError("url must be http(s)")
        return v


def _default_profiles() -> set[str]:
    from app.core.config import load_config

    return set(load_config(ROOT / "configs").profiles)


def create_app(
    store: Store,
    *,
    upload_dir: Path,
    checks: Mapping[str, Callable[[], None]],
    profiles: set[str] | None = None,
    max_upload_mb: float = 25,
    poll_s: float = 0.5,
) -> FastAPI:
    app = FastAPI(title="Semantic compression", version="0.1")
    known_profiles = profiles if profiles is not None else _default_profiles()
    max_bytes = int(max_upload_mb * 2**20)

    @app.exception_handler(ApiProblem)
    async def _api_problem(_: Request, e: ApiProblem) -> JSONResponse:
        return _problem(e)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, e: RequestValidationError) -> JSONResponse:
        return _problem(ApiProblem(422, "invalid_request", "Invalid request", str(e.errors())))

    def options_or_422(raw: Any) -> JobOptions:
        try:
            opts = raw if isinstance(raw, JobOptions) else JobOptions(**(raw or {}))
        except (ValidationError, TypeError) as e:
            raise ApiProblem(422, "invalid_request", "Invalid options", str(e)) from e
        if opts.profile is not None and opts.profile not in known_profiles:
            raise ApiProblem(
                422, "invalid_request", "Unknown profile", f"known: {sorted(known_profiles)}"
            )
        return opts

    async def job_or_404(job_id: UUID) -> Any:
        job = await asyncio.to_thread(store.get_job, job_id)
        if job is None:
            raise ApiProblem(404, "job_not_found", "No such job", str(job_id))
        return job

    @app.post("/v1/jobs", status_code=202)
    async def create_job(request: Request) -> JSONResponse:
        ctype = request.headers.get("content-type", "")
        if ctype.startswith("application/json"):
            try:
                body = CreateJob(**await request.json())
            except (ValidationError, TypeError, ValueError) as e:
                raise ApiProblem(422, "invalid_request", "Invalid request", str(e)) from e
            opts = options_or_422(body.options)
            job_input: dict[str, Any] = {"url": body.url}
        elif ctype.startswith("multipart/form-data"):
            async with request.form() as form:  # closes the spooled upload files
                upload = form.get("file")
                if not isinstance(upload, UploadFile):
                    raise ApiProblem(422, "invalid_request", "Missing multipart field 'file'")
                raw_opts = form.get("options")
                try:
                    parsed = json.loads(raw_opts) if isinstance(raw_opts, str) else {}
                except json.JSONDecodeError as e:
                    raise ApiProblem(422, "invalid_request", "options is not JSON", str(e)) from e
                opts = options_or_422(parsed)
                data = await upload.read(max_bytes + 1)
                if len(data) > max_bytes:
                    raise ApiProblem(
                        413, "too_large", "Upload too large", f"limit {max_upload_mb} MB"
                    )
                if not data.startswith(b"%PDF"):
                    raise ApiProblem(415, "not_pdf", "Not a PDF", upload.filename)
                upload_dir.mkdir(parents=True, exist_ok=True)
                path = upload_dir / f"{uuid4()}.pdf"
                path.write_bytes(data)
                job_input = {"path": str(path), "filename": upload.filename}
        else:
            raise ApiProblem(
                415, "unsupported_media_type", "Use multipart/form-data or application/json"
            )
        jid = await asyncio.to_thread(store.create_job, job_input, opts.model_dump())
        return JSONResponse(
            {"job_id": str(jid)}, status_code=202, headers={"location": f"/v1/jobs/{jid}"}
        )

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: UUID) -> dict[str, Any]:
        job = await job_or_404(job_id)
        return {
            "job_id": str(job.job_id),
            "status": job.status,
            "stage": job.stage,
            "progress": job.progress,
            "route": job.route,
            "doc_id": None if job.doc_id is None else str(job.doc_id),
            "error": job.error,
            "attempts": job.attempts,
        }

    @app.get("/v1/jobs/{job_id}/result")
    async def get_result(job_id: UUID) -> dict[str, Any]:
        job = await job_or_404(job_id)
        if job.status == "failed":
            raise ApiProblem(409, "job_failed", "Job failed", job.error, job.doc_id)
        result = await asyncio.to_thread(store.result, job_id)
        if result is None:
            raise ApiProblem(409, "job_not_done", f"Job is {job.status}", job.stage, job.doc_id)
        return result

    @app.get("/v1/jobs/{job_id}/events")
    async def events(job_id: UUID, request: Request) -> EventSourceResponse:
        await job_or_404(job_id)
        try:
            after = int(request.headers.get("last-event-id", "0"))
        except ValueError:
            after = 0

        async def stream() -> AsyncIterator[dict[str, str]]:
            last = after
            while True:
                # status before events: events are written before the status changes,
                # so a terminal status read first means every event is already there
                job = await asyncio.to_thread(store.get_job, job_id)
                for e in await asyncio.to_thread(store.events, job_id, last):
                    last = e.id
                    yield {
                        "event": "progress",
                        "id": str(e.id),
                        "data": json.dumps(
                            {"stage": e.stage, "progress": e.progress, "message": e.message}
                        ),
                    }
                if job is None or job.status in TERMINAL:
                    status = None if job is None else job.status
                    error = None if job is None else job.error
                    yield {"event": "end", "data": json.dumps({"status": status, "error": error})}
                    return
                await asyncio.sleep(poll_s)

        return EventSourceResponse(stream())

    @app.get("/v1/documents/{doc_id}/text")
    async def document_text(doc_id: UUID) -> dict[str, Any]:
        doc = await asyncio.to_thread(store.document, doc_id)
        if doc is None:
            raise ApiProblem(404, "document_not_found", "No such document", None, doc_id)
        return {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "full_text": doc.full_text,
            "structure_ok": doc.structure_ok,
            "sections": [
                s.model_dump(include={"id", "heading", "level", "span"}) for s in doc.sections
            ],
            "sentences": [s.model_dump() for s in doc.sentences],
        }

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        results: dict[str, str] = {}
        for name, check in checks.items():
            try:
                await asyncio.to_thread(check)
                results[name] = "ok"
            except Exception as e:  # report every dependency, never crash the probe
                results[name] = f"error: {type(e).__name__}: {e}"
        ok = all(v == "ok" for v in results.values())
        return JSONResponse(
            {"status": "ok" if ok else "error", "checks": results},
            status_code=200 if ok else 503,
        )

    return app


def _http_check(url: str) -> Callable[[], None]:
    def check() -> None:
        import httpx

        httpx.get(url, timeout=3).raise_for_status()

    return check


def app_from_env() -> FastAPI:
    from dotenv import load_dotenv

    from app.core.logging import configure_logging
    from app.db.pg import PgStore

    load_dotenv(ROOT / ".env")
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    store = PgStore(os.environ.get("DATABASE_URL", "postgresql://app:app@localhost:5432/app"))
    grobid = os.getenv("GROBID_URL", "http://localhost:8070")
    ollama = os.getenv("OLLAMA_URL", "http://localhost:11434")
    return create_app(
        store,
        upload_dir=Path(os.getenv("UPLOAD_DIR", ROOT / "data" / "uploads")),
        checks={
            "db": store.ping,
            "grobid": _http_check(f"{grobid}/api/isalive"),
            "ollama": _http_check(f"{ollama}/api/tags"),
        },
    )


def __getattr__(name: str) -> Any:
    """`uvicorn app.api.main:app` builds the real app on first access, so importing this
    module (tests) never reads .env or touches the database."""
    if name == "app":
        return app_from_env()
    raise AttributeError(name)
