"""First live Route B run + per-level verification (P4 exit). Not a unit test: real Qwen3,
real verifiers, ~20-30 min on the 4070.

Runs from stored TEI so GROBID/Docker is not needed (CLAUDE.md §0). Forces Route B with
`tau_override=1000`, then decomposes every merge-tree node, unloads Qwen, runs embedder ->
DeBERTa -> MiniCheck, and prints faithfulness per merge level for both verifiers.

Samples system RAM and GPU memory once a second, so CLAUDE.md §2.3 estimates become numbers.
Refuses to start below --min-free-gb available RAM (run 3 was killed for low memory).

Usage:
    .venv/Scripts/python.exe scripts/live_route_b.py [--doc sbert] [--min-free-gb 8]
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.cli import make_provider  # noqa: E402
from app.core.config import load_config  # noqa: E402
from app.ingest.spans import build_document  # noqa: E402
from app.ingest.tei import parse_tei  # noqa: E402
from app.llm.cache import DiskCache  # noqa: E402
from app.llm.client import LLMClient  # noqa: E402
from app.summarize.single_pass import summarize  # noqa: E402
from app.verify.pipeline import openers_from_config, verify_summary  # noqa: E402
from eval.metrics import faithfulness_by_level  # noqa: E402


class _MemStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_gb() -> float:
    s = _MemStatus()
    s.dwLength = ctypes.sizeof(s)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))  # type: ignore[attr-defined]
    return s.ullAvailPhys / 2**30


def gpu_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.split()[0])


class Sampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.min_avail_gb = available_gb()
        self.peak_gpu_mib = gpu_used_mib()
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.wait(1.0):
            self.min_avail_gb = min(self.min_avail_gb, available_gb())
            self.peak_gpu_mib = max(self.peak_gpu_mib, gpu_used_mib())


async def main(doc_name: str, out_dir: Path) -> dict[str, object]:
    cfg = load_config(ROOT / "configs", {"profile": "qwen3_8b_local"})
    profile = cfg.active_profile
    tei = parse_tei((ROOT / f"tests/fixtures/tei/{doc_name}.tei.xml").read_bytes())
    doc = build_document(doc_name, "pdf", tei.title, tei.sections, tei.references_text)
    provider = make_provider(profile)
    client = LLMClient(provider, profile, DiskCache(ROOT / cfg.cache.dir))

    async def release() -> None:
        await provider.unload(profile.model)  # type: ignore[attr-defined]

    t0 = time.monotonic()
    try:
        result = await summarize(doc, client, cfg, prompt_dir=ROOT / "prompts", tau_override=1000)
        t_sum = time.monotonic() - t0
        print(f"summarized: route {result.route.value}, {len(result.nodes)} nodes, {t_sum:.0f} s")
        v = await verify_summary(
            doc,
            result,
            client,
            openers_from_config(cfg.verify),
            cfg.verify,
            release_llm=release,
            prompt_dir=ROOT / "prompts",
        )
    finally:
        await provider.aclose()
    by_level = {
        name: faithfulness_by_level(v.claims, v.nodes, name)  # type: ignore[arg-type]
        for name in ("deberta", "minicheck")
    }
    report = {
        "doc_id": doc.doc_id,
        "route": result.route.value,
        "nodes": len(v.nodes),
        "levels": max(n.level for n in v.nodes),
        "claims": len(v.claims),
        "faithfulness_by_level": by_level,
        "seconds_summarize": round(t_sum),
        "seconds_total": round(time.monotonic() - t0),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{doc_name}.route_b.json").write_text(
        json.dumps(
            {
                **report,
                "summary": result.model_dump(mode="json"),
                "claims_detail": [c.model_dump(mode="json") for c in v.claims],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="sbert")
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    ap.add_argument("--out", type=Path, default=ROOT / "results/live_route_b")
    a = ap.parse_args()
    free = available_gb()
    if free < a.min_free_gb:
        sys.exit(
            f"only {free:.1f} GB RAM available, need {a.min_free_gb:g}: quit Docker from the tray, "
            "run `wsl --shutdown`, close games/browsers (CLAUDE.md M8)"
        )
    sampler = Sampler()
    sampler.start()
    try:
        rep = asyncio.run(main(a.doc, a.out))
    finally:
        sampler.stop.set()
    rep |= {"min_ram_available_gb": round(sampler.min_avail_gb, 2)}
    rep |= {"peak_gpu_mib": sampler.peak_gpu_mib}
    print(json.dumps(rep, indent=2))
