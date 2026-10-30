"""`make models`: the only step allowed to download weights. Verification itself is
local-files-only (FR-5.6), so a fresh machine must run this first."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.core.config import load_config
from app.verify.models import bf16_dir, check, convert_bf16, fetch, ignore_patterns, pinned

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


def _no_convert(src: str, dst: Path, **kw: Any) -> None:
    dst.mkdir(parents=True)


def test_fetch_downloads_each_model_at_its_pinned_revision(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def download(repo_id: str, **kw: Any) -> str:
        calls.append({"repo": repo_id, **kw})
        return f"/cache/{repo_id}"

    fetch(CFG, download=download, convert=_no_convert, hf_home=tmp_path)
    assert [(c["repo"], c["revision"]) for c in calls] == [
        (m.checkpoint, m.revision) for m in pinned(CFG)
    ]
    assert all(c["local_files_only"] is False for c in calls)


def test_check_filters_exactly_like_fetch(tmp_path: Path) -> None:
    """huggingface_hub judges an offline snapshot complete only against the same filter
    it was downloaded with; a mismatch reads as 'incomplete' (seen on roberta-large)."""
    got: dict[str, dict[str, list[str]]] = {"fetch": {}, "check": {}}

    def recorder(kind: str) -> Any:
        def download(repo_id: str, **kw: Any) -> str:
            got[kind][repo_id] = kw["ignore_patterns"]
            return "/x"

        return download

    fetch(CFG, download=recorder("fetch"), convert=_no_convert, hf_home=tmp_path)
    check(CFG, download=recorder("check"), hf_home=tmp_path)
    assert got["fetch"] == got["check"]


def test_check_reports_missing_without_network(tmp_path: Path) -> None:
    def download(repo_id: str, **kw: Any) -> str:
        assert kw["local_files_only"] is True  # --check must never download
        if repo_id == CFG.verify.minicheck.checkpoint:
            raise FileNotFoundError("not cached")
        return f"/cache/{repo_id}"

    status = check(CFG, download=download, hf_home=tmp_path)
    assert status == {
        "embedder": True,
        "nli": True,
        "minicheck": False,
        "bertscore": True,
        "minicheck-bf16": False,
    }


# ------------------------------------- MiniCheck bf16 safetensors local copy ---
# The pinned checkpoint ships one 3.1 GB fp32 pytorch_model.bin, read whole into RAM
# before it reaches the GPU; the live Route B run hit 0.09 GB available there. A bf16
# safetensors copy is half the bytes and memory-mapped.


def _tiny_t5_dir(path: Path) -> Any:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.T5Config(
        vocab_size=256, d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2
    )
    model = transformers.T5ForConditionalGeneration(cfg)
    path.mkdir(parents=True)
    model.config.save_pretrained(path)  # transformers 5 always writes safetensors, so the
    torch.save(model.state_dict(), path / "pytorch_model.bin")  # .bin is written by hand
    (path / "spiece.model").write_bytes(b"tokenizer bytes")
    (path / "tokenizer_config.json").write_text("{}")
    return model


def test_bf16_dir_is_keyed_by_the_pinned_revision(tmp_path: Path) -> None:
    d = bf16_dir(CFG.verify.minicheck, tmp_path)
    assert d.parent == tmp_path / "converted"
    assert CFG.verify.minicheck.revision[:12] in d.name and d.name.endswith("bf16")


def test_convert_bf16_writes_safetensors_with_the_same_weights(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    src, dst = tmp_path / "src", tmp_path / "out" / "mc-bf16"
    model = _tiny_t5_dir(src)
    assert (src / "pytorch_model.bin").exists()

    convert_bf16(src, dst, source={"checkpoint": "x/y", "revision": "abc"})

    names = {p.name for p in dst.iterdir()}
    assert "model.safetensors" in names and "pytorch_model.bin" not in names
    assert {"spiece.model", "tokenizer_config.json", "SOURCE.json"} <= names
    assert json.loads((dst / "SOURCE.json").read_text()) == {
        "checkpoint": "x/y",
        "revision": "abc",
        "dtype": "bfloat16",
    }
    assert not list(dst.parent.glob("*.tmp*"))  # written atomically
    got = transformers.T5ForConditionalGeneration.from_pretrained(dst)
    assert got.dtype == torch.bfloat16
    want = model.state_dict()
    for k, v in got.state_dict().items():
        assert torch.equal(v, want[k].to(torch.bfloat16)), k


def test_fetch_converts_minicheck_once(tmp_path: Path) -> None:
    made: list[tuple[str, Path]] = []

    def convert(src: str, dst: Path, **kw: Any) -> None:
        made.append((src, dst))
        dst.mkdir(parents=True)

    def download(repo_id: str, **kw: Any) -> str:
        return f"/cache/{repo_id}"

    fetch(CFG, download=download, convert=convert, hf_home=tmp_path)
    fetch(CFG, download=download, convert=convert, hf_home=tmp_path)
    mc = CFG.verify.minicheck
    assert made == [(f"/cache/{mc.checkpoint}", bf16_dir(mc, tmp_path))]


def test_check_reports_the_bf16_copy(tmp_path: Path) -> None:
    def download(repo_id: str, **kw: Any) -> str:
        return f"/cache/{repo_id}"

    assert check(CFG, download=download, hf_home=tmp_path)["minicheck-bf16"] is False
    bf16_dir(CFG.verify.minicheck, tmp_path).mkdir(parents=True)
    assert check(CFG, download=download, hf_home=tmp_path)["minicheck-bf16"] is True
