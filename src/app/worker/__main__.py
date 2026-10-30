"""`python -m app.worker`: poll the job queue and process jobs one at a time (FR-6.2).

One job at a time on purpose: the 8 GB GPU holds one model (DECISION-3). Run a second
worker only on a second machine. `--pipeline module:factory` swaps the real pipeline for a
test one (worker-restart test); the factory returns a `Pipeline`.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
import socket
import sys
from pathlib import Path

from dotenv import load_dotenv

from app.core.logging import configure_logging
from app.worker.runner import Pipeline, run_once

ROOT = Path(__file__).resolve().parents[3]
log = logging.getLogger("app.worker")


def live_pipeline() -> Pipeline:
    from app.core.config import load_config
    from app.worker.pipeline import LiveStages, make_pipeline

    return make_pipeline(lambda o: load_config(ROOT / "configs", o), LiveStages)


def load_pipeline(spec: str) -> Pipeline:
    module, _, attr = spec.partition(":")
    factory = getattr(importlib.import_module(module), attr)
    pipeline: Pipeline = factory()
    return pipeline


async def serve(args: argparse.Namespace) -> None:
    from app.db.pg import PgStore

    store = PgStore(os.environ.get("DATABASE_URL", "postgresql://app:app@localhost:5432/app"))
    pipeline = load_pipeline(args.pipeline) if args.pipeline else live_pipeline()
    worker = f"{socket.gethostname()}-{os.getpid()}"
    log.info("worker started", extra={"worker": worker, "lease_s": args.lease})
    while True:
        busy = await run_once(
            store,
            worker,
            pipeline,
            lease_s=args.lease,
            max_attempts=args.max_attempts,
            heartbeat_s=args.heartbeat,
        )
        if args.once and busy:
            return
        if not busy:
            await asyncio.sleep(args.poll)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.worker")
    ap.add_argument("--lease", type=float, default=120.0, help="seconds without heartbeat")
    ap.add_argument("--heartbeat", type=float, default=None, help="default: lease / 3")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--poll", type=float, default=1.0, help="seconds between empty polls")
    ap.add_argument("--once", action="store_true", help="exit after processing one job")
    ap.add_argument("--pipeline", help="module:factory returning a Pipeline (tests)")
    args = ap.parse_args(argv)
    load_dotenv(ROOT / ".env", override=False)  # DATABASE_URL from the caller wins
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        log.info("worker stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
