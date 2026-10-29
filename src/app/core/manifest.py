"""Run manifest. PRD FR-7.3, §11 results layout.

Written once at start (so a crashed run still leaves a trace) and rewritten at
the end with status and end time.
"""

from __future__ import annotations

import hashlib
import secrets
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from app.core.config import AppConfig


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(t: datetime) -> str:
    return t.astimezone(UTC).isoformat().replace("+00:00", "Z")


class RunManifest(BaseModel):
    run_id: str
    experiment: str
    status: Literal["running", "ok", "error"]
    commit: str | None
    dirty: bool | None
    config_fingerprint: str
    config: dict[str, Any]
    prompt_hashes: dict[str, str]
    model: dict[str, Any]
    seeds: dict[str, int]
    started_at: str
    ended_at: str | None = None


def prompt_hashes(prompt_dir: str | Path) -> dict[str, str]:
    """sha256 of every frozen prompt file (`*.txt`), keyed by filename."""
    directory = Path(prompt_dir)
    if not directory.is_dir():
        return {}
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob("*.txt"))
    }


def git_state(cwd: str | Path = ".") -> dict[str, Any]:
    """HEAD commit and dirty flag; both None outside a repo or before the first commit."""

    def git(*args: str) -> str | None:
        try:
            res = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
            )
        except OSError:
            return None
        return res.stdout.strip() if res.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(git("status", "--porcelain"))}


@contextmanager
def run_manifest(
    run_dir: str | Path,
    cfg: AppConfig,
    experiment: str,
    prompt_dir: str | Path = "prompts",
    clock: Callable[[], datetime] = _utcnow,
) -> Iterator[RunManifest]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = clock()
    profile = cfg.active_profile
    manifest = RunManifest(
        run_id=f"{started:%Y%m%dT%H%M%S}-{secrets.token_hex(3)}",
        experiment=experiment,
        status="running",
        **git_state(),
        config_fingerprint=cfg.fingerprint(),
        config=cfg.model_dump(mode="json"),
        prompt_hashes=prompt_hashes(prompt_dir),
        model={
            "provider": profile.provider,
            "model": profile.model,
            "effective_context": profile.effective_context,
        },
        seeds={"decoding": cfg.decoding.seed},
        started_at=_iso(started),
    )
    path = run_dir / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    try:
        yield manifest
    except BaseException:
        manifest.status = "error"
        raise
    else:
        manifest.status = "ok"
    finally:
        manifest.ended_at = _iso(clock())
        path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
