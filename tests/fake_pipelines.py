"""Pipelines for worker-process tests (`--pipeline tests.fake_pipelines:slow`)."""

from __future__ import annotations

import time

from app.db.store import Job, JobOutput
from app.worker.runner import JobContext, Pipeline
from tests.store_contract import make_doc, make_output


def slow() -> Pipeline:
    async def run(job: Job, ctx: JobContext) -> JobOutput:
        ctx.progress("ingest", 0.05, "slow")
        ctx.attach_document(make_doc())
        time.sleep(120)  # killed long before this ends
        return make_output()

    return run


def fast() -> Pipeline:
    async def run(job: Job, ctx: JobContext) -> JobOutput:
        ctx.progress("ingest", 0.05, "fast")
        ctx.attach_document(make_doc())
        return make_output()

    return run
