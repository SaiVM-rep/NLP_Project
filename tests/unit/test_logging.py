"""Structured JSON logging. PRD NFR-5.

The property that matters: every log line is one parseable JSON object, and a
failure anywhere inside a document's processing carries that document's id
without every call site having to pass it.
"""

import io
import json
import logging

import pytest

from app.core.logging import configure_logging, doc_context, stage


@pytest.fixture
def stream():
    buf = io.StringIO()
    configure_logging(level="DEBUG", stream=buf)
    yield buf
    logging.getLogger().handlers.clear()


def lines(buf: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in buf.getvalue().splitlines()]


def test_every_line_is_one_json_object(stream):
    logging.getLogger("x").info("hello %s", "world")
    [rec] = lines(stream)
    assert rec["msg"] == "hello world"
    assert rec["level"] == "INFO"
    assert rec["logger"] == "x"
    assert "ts" in rec


def test_doc_id_key_always_present_even_outside_a_document(stream):
    logging.getLogger("x").info("no doc")
    assert lines(stream)[0]["doc_id"] is None


def test_doc_context_stamps_doc_id_on_nested_logs(stream):
    with doc_context("doc-42"):
        logging.getLogger("deep.module").warning("inside")
    logging.getLogger("x").info("outside")
    inside, outside = lines(stream)
    assert inside["doc_id"] == "doc-42"
    assert outside["doc_id"] is None


def test_extra_fields_are_emitted(stream):
    logging.getLogger("x").info("call", extra={"provider": "gemini", "tokens_in": 34})
    rec = lines(stream)[0]
    assert rec["provider"] == "gemini"
    assert rec["tokens_in"] == 34


def test_stage_logs_duration_on_success(stream):
    with doc_context("d1"), stage("ingest"):
        pass
    [rec] = lines(stream)
    assert rec["stage"] == "ingest"
    assert rec["status"] == "ok"
    assert rec["doc_id"] == "d1"
    assert rec["duration_ms"] >= 0


def test_stage_failure_carries_doc_id_and_reraises(stream):
    with pytest.raises(ValueError), doc_context("d9"), stage("route"):
        raise ValueError("boom")
    [rec] = lines(stream)
    assert rec["level"] == "ERROR"
    assert rec["status"] == "error"
    assert rec["doc_id"] == "d9"
    assert rec["stage"] == "route"
    assert "ValueError: boom" in rec["exc"]


def test_non_json_extra_does_not_crash_logging(stream):
    logging.getLogger("x").info("obj", extra={"path": object()})
    assert "path" in lines(stream)[0]


def test_configure_is_idempotent(stream):
    configure_logging(level="DEBUG", stream=stream)
    logging.getLogger("x").info("once")
    assert len(lines(stream)) == 1
