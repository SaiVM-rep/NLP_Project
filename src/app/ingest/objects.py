"""Tables and formulas of a PDF, for display next to a crop of the page (DECISION-40).

Docling runs as a subprocess in its own env (`make docling-env`), so its dependency pins
never touch the verified pipeline and its GPU/RAM is freed when it exits. The output is
display-only: summaries and claim checks never read it. Any failure gives an empty list,
never a failed job. Results are cached by the PDF's sha256.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "docling_extract.py"
DEFAULT_PYTHON = ROOT / ".venv-docling" / "Scripts" / "python.exe"
TIMEOUT_S = 300.0

log = logging.getLogger(__name__)

# (argv, env, timeout_s) -> None; raises on failure. Writes argv[-1].
Runner = Callable[[list[str], dict[str, str], float], None]


class PaperObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["table", "formula"]
    page: int = Field(ge=1)
    caption: str | None = None
    content: str  # markdown table, or LaTeX ("" when Docling could not read it)
    image_png: str  # base64 PNG crop of the region on the page


_LIST = TypeAdapter(list[PaperObject])


def _subprocess(args: list[str], env: dict[str, str], timeout: float) -> None:
    subprocess.run(args, env=env, timeout=timeout, check=True, capture_output=True)


def extract_objects(
    pdf: Path,
    *,
    cache_dir: Path,
    python: Path | None = None,
    runner: Runner = _subprocess,
    timeout: float = TIMEOUT_S,
) -> list[PaperObject]:
    key = hashlib.sha256(pdf.read_bytes()).hexdigest()
    done = cache_dir / f"{key}.json"
    if done.exists():
        return _LIST.validate_json(done.read_text(encoding="utf-8"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f"{key}.tmp.json"
    env = {k: v for k, v in os.environ.items() if k != "HF_HOME"}
    env["HF_HUB_OFFLINE"] = "1"  # weights are fetched by `make docling-env`, never mid-job
    exe = python or Path(os.getenv("DOCLING_PYTHON", DEFAULT_PYTHON))
    try:
        runner([str(exe), str(SCRIPT), str(pdf), str(tmp)], env, timeout)
        objects = _LIST.validate_json(tmp.read_text(encoding="utf-8"))
    except (OSError, subprocess.SubprocessError, ValidationError) as e:
        log.warning("tables and formulas skipped for %s: %s", pdf.name, e)
        tmp.unlink(missing_ok=True)
        return []
    tmp.replace(done)
    return objects
