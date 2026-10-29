"""`make models`: the only step allowed to download weights. Verification itself is
local-files-only (FR-5.6), so a fresh machine must run this first."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.config import load_config
from app.verify.models import check, fetch, ignore_patterns, pinned

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "configs")


def test_every_model_the_pipeline_loads_is_pinned() -> None:
    names = {m.name: (m.checkpoint, m.revision) for m in pinned(CFG)}
    assert names == {
        "embedder": (CFG.verify.embedder.checkpoint, CFG.verify.embedder.revision),
        "nli": (CFG.verify.nli.checkpoint, CFG.verify.nli.revision),
        "minicheck": (CFG.verify.minicheck.checkpoint, CFG.verify.minicheck.revision),
        "bertscore": (CFG.metrics.bertscore.checkpoint, CFG.metrics.bertscore.revision),
    }


def test_skips_bin_and_other_formats_for_safetensors_models() -> None:
    pats = ignore_patterns("safetensors")
    assert "*.bin" in pats and "onnx/*" in pats and "*.msgpack" in pats


def test_keeps_bin_when_it_is_the_only_weights() -> None:
    assert "*.bin" not in ignore_patterns("bin")


def test_weight_formats_match_what_the_repos_ship() -> None:
    """Measured 2026-10-29: MiniCheck-Flan-T5-Large has pytorch_model.bin only."""
    assert {m.name: m.weights for m in pinned(CFG)} == {
        "embedder": "safetensors",
        "nli": "safetensors",
        "minicheck": "bin",
        "bertscore": "safetensors",
    }


def test_fetch_downloads_each_model_at_its_pinned_revision() -> None:
    calls: list[dict[str, Any]] = []

    def download(repo_id: str, **kw: Any) -> str:
        calls.append({"repo": repo_id, **kw})
        return f"/cache/{repo_id}"

    fetch(CFG, download=download)
    assert [(c["repo"], c["revision"]) for c in calls] == [
        (m.checkpoint, m.revision) for m in pinned(CFG)
    ]
    assert all(c["local_files_only"] is False for c in calls)


def test_check_filters_exactly_like_fetch() -> None:
    """huggingface_hub judges an offline snapshot complete only against the same filter
    it was downloaded with; a mismatch reads as 'incomplete' (seen on roberta-large)."""
    got: dict[str, dict[str, list[str]]] = {"fetch": {}, "check": {}}

    def recorder(kind: str) -> Any:
        def download(repo_id: str, **kw: Any) -> str:
            got[kind][repo_id] = kw["ignore_patterns"]
            return "/x"

        return download

    fetch(CFG, download=recorder("fetch"))
    check(CFG, download=recorder("check"))
    assert got["fetch"] == got["check"]


def test_check_reports_missing_without_network() -> None:
    def download(repo_id: str, **kw: Any) -> str:
        assert kw["local_files_only"] is True  # --check must never download
        if repo_id == CFG.verify.minicheck.checkpoint:
            raise FileNotFoundError("not cached")
        return f"/cache/{repo_id}"

    status = check(CFG, download=download)
    assert status == {"embedder": True, "nli": True, "minicheck": False, "bertscore": True}
