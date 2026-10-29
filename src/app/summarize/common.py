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
            accept=lambda t: _parse(t) is not None,
        )
        if (summary := _parse(res.text)) is not None:
            return summary, res.model
    raise SummaryError(f"model returned invalid JSON twice for {what}")
