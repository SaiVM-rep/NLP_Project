"""First live P5 run: key points for one paper with real Qwen3 + MiniLM. Not a unit test.

Runs from stored TEI (no GROBID/Docker). Candidates per section from Qwen3, Qwen unloaded,
then MiniLM for salience, clustering and attribution (PRD §11.7). Prints each key point
next to the source sentence it was attributed to, for reading by hand, and samples RAM/GPU
like `live_route_b.py`.

Usage:
    .venv/Scripts/python.exe scripts/live_keypoints.py [--doc sbert] [--min-free-gb 6]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from live_route_b import Sampler, available_gb  # noqa: E402

from app.cli import make_provider  # noqa: E402
from app.core.config import load_config  # noqa: E402
from app.ingest.spans import build_document  # noqa: E402
from app.ingest.tei import parse_tei  # noqa: E402
from app.keypoints.pipeline import extract_keypoints  # noqa: E402
from app.llm.cache import DiskCache  # noqa: E402
from app.llm.client import LLMClient  # noqa: E402
from app.verify.pipeline import openers_from_config  # noqa: E402


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
        kps = await extract_keypoints(
            doc,
            client,
            openers_from_config(cfg.verify).embedder,
            cfg,
            release_llm=release,
            prompt_dir=ROOT / "prompts",
        )
    finally:
        await provider.aclose()
    seconds = time.monotonic() - t0
    rows = [
        {
            **k.model_dump(mode="json"),
            "evidence_text": doc.text_of(k.evidence) if k.evidence else None,
        }
        for k in kps
    ]
    for r in rows:
        src = r["evidence_text"] or "(unattributed)"
        sim = r["similarity"] or 0.0
        print(f"#{r['rank']} [cluster {r['cluster_size']}, sim {sim:.2f}] {r['text']}")
        print(f"    <- {src}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{doc_name}.keypoints.json").write_text(
        json.dumps({"doc_id": doc.doc_id, "keypoints": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "doc_id": doc.doc_id,
        "sections": len(doc.sections),
        "keypoints": len(kps),
        "unattributed": sum(k.evidence is None for k in kps),
        "seconds": round(seconds),
    }


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="sbert")
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument("--out", type=Path, default=ROOT / "results/live_keypoints")
    a = ap.parse_args()
    free = available_gb()
    if free < a.min_free_gb:
        sys.exit(f"only {free:.1f} GB RAM available, need {a.min_free_gb:g} (CLAUDE.md M8)")
    sampler = Sampler()
    sampler.start()
    try:
        rep = asyncio.run(main(a.doc, a.out))
    finally:
        sampler.stop.set()
    rep |= {"min_ram_available_gb": round(sampler.min_avail_gb, 2)}
    rep |= {"peak_gpu_mib": sampler.peak_gpu_mib}
    print(json.dumps(rep, indent=2))
