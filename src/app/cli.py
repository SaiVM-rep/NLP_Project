"""Command line. P2 exit: `python -m app.cli summarize paper.pdf`.

Prints the summary and writes `<out>/<doc_id>.json` with the summary, routing
metadata (FR-2.4) and the parsed document, so offsets can be inspected later.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

from app.core.config import ModelProfile, load_config
from app.core.logging import configure_logging, doc_context
from app.ingest.grobid import GrobidClient, ingest_pdf
from app.ingest.tei import IngestError
from app.ingest.web import ingest_url
from app.llm.base import LLMError, Provider
from app.llm.cache import DiskCache
from app.llm.client import LLMClient
from app.llm.limiter import TokenBucket
from app.llm.providers.gemini import GeminiProvider
from app.llm.providers.ollama import OllamaProvider
from app.summarize.single_pass import SummaryError, summarize

ROOT = Path(__file__).resolve().parents[2]


def make_provider(profile: ModelProfile) -> Provider:
    if profile.provider == "ollama":
        # keep_alive=0: DECISION-3, the 8 GB GPU must be free for the verifiers afterwards.
        # num_ctx: Ollama's default (4096) silently truncates the prompt from the front.
        return OllamaProvider(
            os.getenv("OLLAMA_URL", "http://localhost:11434"),
            keep_alive=0,
            num_ctx=profile.effective_context,
        )
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise LLMError("GEMINI_API_KEY is empty; set it in .env or use --profile qwen3_8b_local")
    return GeminiProvider(api_key=key)


async def _summarize(args: argparse.Namespace) -> int:
    is_url = args.file.lower().startswith(("http://", "https://"))
    path = Path(args.file)
    if not is_url and not path.is_file():
        print(f"error: {path} not found", file=sys.stderr)
        return 1

    overrides = {"profile": args.profile} if args.profile else None
    cfg = load_config(ROOT / "configs", overrides)
    profile = cfg.active_profile
    client = LLMClient(
        make_provider(profile),
        profile,
        DiskCache(Path(os.getenv("LLM_CACHE_DIR", ROOT / cfg.cache.dir))),
        rpm_bucket=TokenBucket(profile.limits.rpm / 60, profile.limits.rpm),
        fallback=cfg.fallback,
    )

    if is_url:
        with httpx.Client(timeout=30, headers={"user-agent": "semantic-compression/0.1"}) as http:
            doc = ingest_url(args.file, http)
    else:
        doc = ingest_pdf(path, GrobidClient(os.getenv("GROBID_URL", "http://localhost:8070")))
    with doc_context(doc.doc_id):
        if not doc.structure_ok:
            print("warning: no section structure found; summarizing flat text", file=sys.stderr)
        result = await summarize(
            doc, client, cfg, prompt_dir=ROOT / "prompts", tau_override=args.tau
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {**result.model_dump(mode="json"), "document": doc.model_dump(mode="json")}
    target = out / f"{doc.doc_id}.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"# {doc.title}\n\n{result.summary}\n")
    print(
        f"[{result.model} · route {result.route.value} · {result.tokens} tokens · saved {target}]"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize", help="summarize a PDF or a web article URL")
    s.add_argument("file", help="PDF path or http(s) URL")
    s.add_argument("--profile", help="model profile from configs/models.yaml")
    s.add_argument("--tau", type=int, help="override the Route A/B threshold")
    s.add_argument("--out", default=str(ROOT / "results" / "summaries"))
    args = parser.parse_args(argv)

    load_dotenv(ROOT / ".env")
    configure_logging("WARNING")
    try:
        return asyncio.run(_summarize(args))
    except (IngestError, LLMError, SummaryError) as e:
        logging.getLogger("app.cli").error(str(e), extra={"source": args.file})
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
