"""`python -m app.verify.models --fetch|--check` (`make models`).

The only place model weights are downloaded. Verification loads with
`local_files_only=True` (FR-5.6), so a fresh machine runs `--fetch` once.
Downloads go to HF_HOME, outside the repo.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any, Literal, NamedTuple

from app.core.config import AppConfig, load_config

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


def fetch(cfg: AppConfig, *, download: Download = _hf_download) -> dict[str, str]:
    out = {}
    for m in pinned(cfg):
        out[m.name] = download(
            m.checkpoint,
            revision=m.revision,
            ignore_patterns=ignore_patterns(m.weights),
            local_files_only=False,
        )
    return out


def check(cfg: AppConfig, *, download: Download = _hf_download) -> dict[str, bool]:
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
    return 0 if all(status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
