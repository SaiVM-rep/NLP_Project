"""Structured JSON logging. PRD NFR-5.

One JSON object per line. `doc_id` rides a contextvar, so code deep inside a
document's processing never has to thread it through: every line logged
inside `doc_context(...)` carries it, and every line outside carries `null`.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import IO, Any

_doc_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("doc_id", default=None)

# Attributes every LogRecord has; anything else on a record came from `extra=`.
_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "doc_id": _doc_id.get(),
        }
        out.update({k: v for k, v in vars(record).items() if k not in _RESERVED})
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        # default=str: a stray non-JSON extra must not take the log line down with it.
        return json.dumps(out, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO", stream: IO[str] | None = None) -> None:
    """Replace root handlers with one JSON handler. Safe to call twice."""
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


@contextmanager
def doc_context(doc_id: str) -> Iterator[None]:
    token = _doc_id.set(doc_id)
    try:
        yield
    finally:
        _doc_id.reset(token)


@contextmanager
def stage(name: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Log one line per pipeline stage with its duration; failures logged then re-raised."""
    log = logger or logging.getLogger("app.stage")
    start = time.perf_counter()
    try:
        yield
    except Exception:
        log.exception(
            "stage failed",
            extra={"stage": name, "status": "error", "duration_ms": _ms(start)},
        )
        raise
    log.info("stage done", extra={"stage": name, "status": "ok", "duration_ms": _ms(start)})


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)
