"""FR-6.2 acceptance: kill the worker process mid-job; the job resumes or fails cleanly.

Real worker processes (`python -m app.worker`) against real Postgres, hard-killed
(TerminateProcess on Windows: no cleanup code runs, like a crash or power loss).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

import pytest

pytest.importorskip("psycopg")

from app.db.migrate import migrate
from app.db.pg import PgStore

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
LEASE = "1.5"


def worker(dsn: str, pipeline: str, *extra: str) -> subprocess.Popen[bytes]:
    env = {
        **os.environ,
        "DATABASE_URL": dsn,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
    }
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "app.worker",
            "--once",
            "--lease",
            LEASE,
            "--heartbeat",
            "0.3",
            "--pipeline",
            f"tests.fake_pipelines:{pipeline}",
            *extra,
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def wait_for(cond: Callable[[], bool], timeout: float = 30) -> None:
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            raise TimeoutError
        time.sleep(0.1)


def kill_mid_job(store: PgStore, jid: UUID, dsn: str, *extra: str) -> str:
    """Start a slow worker, wait until it is inside the job, hard-kill it."""

    def slow_events() -> int:
        return sum(e.message == "slow" for e in store.events(jid, 0))

    before = slow_events()
    with worker(dsn, "slow", *extra) as p:  # closes its pipes
        try:
            wait_for(lambda: slow_events() > before)
        finally:
            p.kill()
    job = store.get_job(jid)
    assert job is not None and job.status == "running" and job.locked_by
    return job.locked_by


def test_killed_job_is_taken_over_and_finishes(dsn: str) -> None:
    migrate(dsn)
    store = PgStore(dsn)
    jid = store.create_job({"url": "https://x"}, {})
    kill_mid_job(store, jid, dsn)

    with worker(dsn, "fast") as p:
        out, _ = p.communicate(timeout=60)
    assert p.returncode == 0, out.decode(errors="replace")
    job = store.get_job(jid)
    assert job is not None
    assert (job.status, job.attempts, job.locked_by) == ("done", 2, None)
    r = store.result(jid)
    assert r is not None and r["route"] == "B"
    assert [e.message for e in store.events(jid, 0)][-2:] == ["fast", ""]


def test_job_killed_every_time_fails_cleanly(dsn: str) -> None:
    migrate(dsn)
    store = PgStore(dsn)
    jid = store.create_job({"url": "https://x"}, {})
    kill_mid_job(store, jid, dsn, "--max-attempts", "2")
    kill_mid_job(store, jid, dsn, "--max-attempts", "2")

    # the next worker's claim fails the job instead of running it a third time;
    # with nothing left to do it keeps polling, so stop it once the job is failed
    with worker(dsn, "fast", "--max-attempts", "2") as p:
        try:
            wait_for(lambda: getattr(store.get_job(jid), "status", None) == "failed")
        finally:
            p.kill()
    job = store.get_job(jid)
    assert job is not None and job.status == "failed"
    assert job.error is not None and "2 attempts" in job.error
