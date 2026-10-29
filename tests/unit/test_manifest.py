"""Run manifest. PRD FR-7.3: commit, config, prompt hashes, models, seeds, times.

Without it no result in chapter 5 can be traced back to the code and prompts
that produced it (SC5).
"""

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from app.core.config import load_config
from app.core.manifest import git_state, prompt_hashes, run_manifest


@pytest.fixture
def cfg():
    return load_config("configs")


class FakeClock:
    def __init__(self) -> None:
        self.t = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.t += timedelta(seconds=5)
        return self.t


def test_prompt_hashes_are_sha256_of_file_bytes(tmp_path):
    (tmp_path / "single_pass.v1.txt").write_bytes(b"Summarise:\n{doc}")
    (tmp_path / "notes.md").write_text("not a prompt")
    assert prompt_hashes(tmp_path) == {
        "single_pass.v1.txt": hashlib.sha256(b"Summarise:\n{doc}").hexdigest()
    }


def test_prompt_hashes_of_missing_dir_is_empty(tmp_path):
    assert prompt_hashes(tmp_path / "nope") == {}


def test_git_state_outside_a_repo_is_none(tmp_path):
    assert git_state(tmp_path) == {"commit": None, "dirty": None}


def test_manifest_written_at_start_so_a_crash_still_leaves_one(tmp_path, cfg):
    run_dir = tmp_path / "run"
    with pytest.raises(RuntimeError), run_manifest(run_dir, cfg, "E1", prompt_dir=tmp_path):
        assert json.loads((run_dir / "manifest.json").read_text())["ended_at"] is None
        raise RuntimeError("mid-run crash")
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["status"] == "error"
    assert m["ended_at"] is not None


def test_manifest_records_every_fr_7_3_field(tmp_path, cfg):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "p.v1.txt").write_text("x")
    clock = FakeClock()
    with run_manifest(tmp_path / "r", cfg, "E2", prompt_dir=prompts, clock=clock) as m:
        assert m.run_id
    saved = json.loads((tmp_path / "r" / "manifest.json").read_text())

    assert saved["experiment"] == "E2"
    assert saved["status"] == "ok"
    assert {"commit", "dirty"} <= saved.keys()
    assert saved["config_fingerprint"] == cfg.fingerprint()
    assert saved["config"]["profile"] == cfg.profile
    assert saved["prompt_hashes"] == {"p.v1.txt": hashlib.sha256(b"x").hexdigest()}
    assert saved["model"] == {
        "provider": cfg.active_profile.provider,
        "model": cfg.active_profile.model,
        "effective_context": cfg.active_profile.effective_context,
    }
    assert saved["seeds"] == {"decoding": cfg.decoding.seed}
    assert saved["started_at"] == "2026-10-29T12:00:05Z"
    assert saved["ended_at"] == "2026-10-29T12:00:10Z"
