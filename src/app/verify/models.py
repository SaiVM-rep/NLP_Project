"""`python -m app.verify.models --fetch|--check` (`make models`).

The only place model weights are downloaded. Verification loads with
`local_files_only=True` (FR-5.6), so a fresh machine runs `--fetch` once.
Downloads go to HF_HOME, outside the repo.

`--fetch` also writes a bf16 safetensors copy of MiniCheck (DECISION-22): the pinned
checkpoint is one 3.1 GB fp32 `.bin`, read whole into RAM on every load.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, NamedTuple

from app.core.config import AppConfig, ModelRef, load_config

Download = Callable[..., str]  # huggingface_hub.snapshot_download
Weights = Literal["safetensors", "bin"]

# Formats nothing here loads: torch reads safetensors (or .bin when it is all there is).
ALWAYS_SKIP = ["onnx/*", "openvino/*", "*.onnx", "*.h5", "*.msgpack", "*.ot", "tf_model*", "flax_*"]


class Pinned(NamedTuple):
    name: str
    checkpoint: str
    revision: str
    weights: Weights  # what the repo ships at that revision (measured, CLAUDE.md §8)


def pinned(cfg: AppConfig) -> list[Pinned]:
    v, b = cfg.verify, cfg.metrics.bertscore
    return [
        Pinned("embedder", v.embedder.checkpoint, v.embedder.revision, "safetensors"),
        Pinned("nli", v.nli.checkpoint, v.nli.revision, "safetensors"),
        Pinned("minicheck", v.minicheck.checkpoint, v.minicheck.revision, "bin"),
        Pinned("bertscore", b.checkpoint, b.revision, "safetensors"),
    ]


def ignore_patterns(weights: Weights) -> list[str]:
    """Static, so --fetch and the offline --check filter identically: huggingface_hub
    judges a local snapshot complete only against the filter it was fetched with."""
    return [*ALWAYS_SKIP, *(["*.bin"] if weights == "safetensors" else [])]


def _hf_download(repo_id: str, **kw: Any) -> str:
    from huggingface_hub import snapshot_download

    path: str = snapshot_download(repo_id, **kw)
    return path


def default_hf_home() -> Path:
    from huggingface_hub import constants

    return Path(os.environ.get("HF_HOME") or constants.HF_HOME)


def bf16_dir(m: ModelRef, root: Path) -> Path:
    return root / "converted" / f"{m.checkpoint.split('/')[-1]}-{m.revision[:12]}-bf16"


WEIGHT_FILES = ("pytorch_model.bin", "pytorch_model.bin.index.json")


def convert_bf16(src: str | Path, dst: Path, *, source: dict[str, str]) -> None:
    """Seq2seq checkpoint -> bf16 safetensors at `dst`, with its tokenizer files and a
    SOURCE.json provenance note. Written to a temp dir, then renamed into place."""
    import torch
    from transformers import AutoModelForSeq2SeqLM

    src = Path(src)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file() and f.name not in WEIGHT_FILES and f.suffix != ".safetensors":
            shutil.copyfile(f, tmp / f.name)
    model = AutoModelForSeq2SeqLM.from_pretrained(src, dtype=torch.bfloat16)
    model.save_pretrained(tmp, safe_serialization=True)
    del model
    (tmp / "SOURCE.json").write_text(json.dumps({**source, "dtype": "bfloat16"}, indent=2))
    tmp.rename(dst)


def fetch(
    cfg: AppConfig,
    *,
    download: Download = _hf_download,
    convert: Callable[..., None] = convert_bf16,
    hf_home: Path | None = None,
) -> dict[str, str]:
    out = {}
    for m in pinned(cfg):
        out[m.name] = download(
            m.checkpoint,
            revision=m.revision,
            ignore_patterns=ignore_patterns(m.weights),
            local_files_only=False,
        )
    mc = cfg.verify.minicheck
    dst = bf16_dir(mc, hf_home or default_hf_home())
    if not dst.exists():
        convert(
            out["minicheck"], dst, source={"checkpoint": mc.checkpoint, "revision": mc.revision}
        )
    out["minicheck-bf16"] = str(dst)
    return out


def check(
    cfg: AppConfig, *, download: Download = _hf_download, hf_home: Path | None = None
) -> dict[str, bool]:
    """Which pinned snapshots are in the local cache. Never touches the network."""
    status = {}
    for m in pinned(cfg):
        try:
            download(
                m.checkpoint,
                revision=m.revision,
                ignore_patterns=ignore_patterns(m.weights),
                local_files_only=True,
            )
            status[m.name] = True
        except (OSError, ValueError):  # LocalEntryNotFoundError is both
            status[m.name] = False
    status["minicheck-bf16"] = bf16_dir(cfg.verify.minicheck, hf_home or default_hf_home()).is_dir()
    return status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.verify.models")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fetch", action="store_true", help="download every pinned checkpoint")
    g.add_argument("--check", action="store_true", help="report what is cached, offline")
    args = ap.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv()  # HF_HOME, HF_HUB_DISABLE_XET
    cfg = load_config("configs")
    if args.fetch:
        for name, path in fetch(cfg).items():
            print(f"fetched  {name:10} {path}")
    status = check(cfg)
    for m in pinned(cfg):
        state = "OK" if status[m.name] else "MISSING"
        print(f"{state:8} {m.name:10} {m.checkpoint}@{m.revision[:7]}")
    bf16 = "OK" if status["minicheck-bf16"] else "MISSING"
    print(f"{bf16:8} {'minicheck-bf16':10} {bf16_dir(cfg.verify.minicheck, default_hf_home())}")
    return 0 if all(status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
