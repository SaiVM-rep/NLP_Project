"""FR-4.1: candidate key points, one LLM call per section. PRD §11.7 step 1.

Generation, not measurement: decoding comes from `cfg.decoding`, the same as the summaries.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from app.core.config import AppConfig
from app.core.schemas import Document, Span
from app.llm.client import LLMClient
from app.summarize.common import fill, load_prompt

PROMPT = "keypoints.v1.txt"
SCHEMA = {
    "type": "OBJECT",
    "properties": {"points": {"type": "ARRAY", "items": {"type": "STRING"}}},
    "required": ["points"],
}
REPAIR = '\n\nYour previous reply was not valid JSON. Reply with only: {"points": ["...", "..."]}'


class CandidateError(RuntimeError):
    pass


class Candidate(BaseModel):
    text: str
    section_id: str
    position: int  # index in document order over all candidates (FR-4.4 position weight)


def _parse(text: str) -> list[str] | None:
    body = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        out = json.loads(body)
    except json.JSONDecodeError:
        return None
    points = out.get("points") if isinstance(out, dict) else None
    if not isinstance(points, list) or not all(isinstance(p, str) for p in points):
        return None
    return [p.strip() for p in points if p.strip()]


async def generate_candidates(
    doc: Document, client: LLMClient, cfg: AppConfig, prompt_dir: str | Path = "prompts"
) -> list[Candidate]:
    template, _ = load_prompt(prompt_dir, PROMPT)
    n, dec = cfg.keypoints.candidates_per_section, cfg.decoding
    out: list[Candidate] = []
    for section in doc.sections:
        if not section.paragraphs:
            continue
        # body only: the heading is already in the prompt
        body = doc.text_of(Span(start=section.paragraphs[0].start, end=section.paragraphs[-1].end))
        prompt = fill(template, n=str(n), heading=section.heading, section_text=body)
        points = None
        for attempt in (prompt, prompt + REPAIR):
            res = await client.generate(
                attempt,
                purpose="keypoints",
                json_schema=SCHEMA,
                temperature=dec.temperature,
                top_p=dec.top_p,
                seed=dec.seed,
                accept=lambda t: _parse(t) is not None,
            )
            if (points := _parse(res.text)) is not None:
                break
        if points is None:
            raise CandidateError(f"{doc.doc_id}/{section.id}: invalid JSON twice")
        base = len(out)  # not inside the generator: extend() grows `out` while consuming it
        out.extend(
            Candidate(text=t, section_id=section.id, position=base + i)
            for i, t in enumerate(points[:n])
        )
    return out
