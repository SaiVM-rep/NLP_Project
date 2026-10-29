"""Disk cache for LLM responses. PRD §11.3.

Protects free-tier quota (NFR-1, NFR-3) and makes reruns deterministic (NFR-6).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def cache_key(
    provider: str,
    model: str,
    prompt: str,
    params: dict[str, Any],
    seed: int | None,
) -> str:
    """sha256 over everything that can change the response.

    `sort_keys` makes the key independent of dict ordering, so the same request
    built two different ways still hits.
    """
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "params": params,
            "seed": seed,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiskCache:
    """Sharded JSON files on disk. No eviction: experiment reruns must stay free."""

    def __init__(self, directory: str | Path, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = enabled

    def _path(self, key: str) -> Path:
        # Shard by the first two hex chars; one flat dir would hit 100k+ files.
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as fh:
                result: dict[str, Any] = json.load(fh)
        except (json.JSONDecodeError, OSError):
            # A truncated write (interrupted run) is a miss, not a crash.
            return None
        return result

    def set(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so an interrupted run cannot leave a half-file
        # that a later run would read as valid.
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False)
        tmp.replace(path)
