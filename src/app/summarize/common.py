"""What both routes share, so they can't drift apart (FR-3.5): output schema, JSON repair,
decoding parameters, prompt loading + hashing, and the result type."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.config import AppConfig
from app.core.schemas import Route, SummaryNode
from app.llm.client import LLMClient
from app.route.chunker import Chunk

SCHEMA = {"type": "OBJECT", "properties": {"summary": {"type": "STRING"}}, "required": ["summary"]}
REPAIR = '\n\nYour previous reply was not valid JSON. Reply with only: {"summary": "<text>"}'
# DECISION-38: with a word cap, a final summary over LENGTH_SLACK x cap is compressed once.
LENGTH_SLACK = 1.25
COMPRESS_PROMPT = "compress.v1.txt"
WORDS_PER_SENTENCE = 27  # measured: Qwen summary sentences run ~20-28 words


class SummaryError(RuntimeError):
    pass


class SummaryResult(BaseModel):
    doc_id: str
    summary: str
    route: Route
    tokens: int
    tau: int
    prompt_hashes: dict[str, str]  # file -> sha256, same as manifest.prompt_hashes (FR-3.5)
    model: str
    # Route B only: every node of the merge tree is kept so E3 can verify each level.
    nodes: list[SummaryNode] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    root_id: str | None = None


def load_prompt(prompt_dir: str | Path, name: str) -> tuple[str, str]:
    """(template, sha256 of the file bytes)."""
    raw = (Path(prompt_dir) / name).read_bytes()
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()


def fill(template: str, **values: str) -> str:
    # str.replace, not .format: the templates contain literal JSON braces.
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def _parse(text: str) -> str | None:
    body = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        out = json.loads(body)
    except json.JSONDecodeError:
        return None
    summary = out.get("summary") if isinstance(out, dict) else None
    return summary.strip() if isinstance(summary, str) and summary.strip() else None


def capped(target: int, cfg: AppConfig) -> int:
    """`summary.max_words` bounds every target word count (DECISION-37)."""
    cap = cfg.summary.max_words
    return target if cap is None else min(target, cap)


def _parse_sentences(text: str) -> list[str] | None:
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        return None
    items = out.get("sentences") if isinstance(out, dict) else None
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        return None
    kept = [x.strip() for x in items if x.strip()]
    return kept or None


async def compress_if_long(
    text: str, client: LLMClient, cfg: AppConfig, *, prompt_dir: str | Path, what: str
) -> tuple[str, str] | None:
    """(shorter summary, compress-prompt sha256), or None when no cap applies or `text` is
    within LENGTH_SLACK of it. DECISION-38: Qwen copies its input when asked for fewer words
    (T5 root 2,100 words for 500, unchanged when re-asked or re-summarized), so the limit is a
    sentence count enforced by the output schema (`maxItems`) while decoding."""
    cap = cfg.summary.max_words
    if cap is None or len(text.split()) <= LENGTH_SLACK * cap:
        return None
    template, digest = load_prompt(prompt_dir, COMPRESS_PROMPT)
    k = max(3, round(cap / WORDS_PER_SENTENCE))
    schema = {
        "type": "OBJECT",
        "properties": {"sentences": {"type": "ARRAY", "items": {"type": "STRING"}, "maxItems": k}},
        "required": ["sentences"],
    }
    prompt = fill(template, max_sentences=str(k), target_words=str(cap), summary=text)
    dec = cfg.decoding
    for attempt in (prompt, prompt + '\n\nReply with only: {"sentences": ["...", "..."]}'):
        res = await client.generate(
            attempt,
            purpose="compress",
            json_schema=schema,
            temperature=dec.temperature,
            top_p=dec.top_p,
            seed=dec.seed,
            think=cfg.summary.think,
            accept=lambda t: _parse_sentences(t) is not None,
        )
        if (sentences := _parse_sentences(res.text)) is not None:
            return " ".join(sentences[:k]), digest
    raise SummaryError(f"model returned invalid JSON twice compressing {what}")


async def ask_summary(
    client: LLMClient, prompt: str, *, purpose: str, cfg: AppConfig, what: str
) -> tuple[str, str]:
    """(summary, model). One repair retry on invalid JSON, then SummaryError."""
    dec = cfg.decoding
    for attempt in (prompt, prompt + REPAIR):
        res = await client.generate(
            attempt,
            purpose=purpose,
            json_schema=SCHEMA,
            temperature=dec.temperature,
            top_p=dec.top_p,
            seed=dec.seed,
            think=cfg.summary.think,
            accept=lambda t: _parse(t) is not None,
        )
        if (summary := _parse(res.text)) is not None:
            return summary, res.model
    raise SummaryError(f"model returned invalid JSON twice for {what}")
