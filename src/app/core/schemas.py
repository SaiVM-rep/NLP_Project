"""Core data model. PRD §11.1.

Offsets are the backbone: `full_text` is stored once and everything else is a
span over it. The validators below exist so SC2 ("every span slices back to its
source text") is impossible to violate silently.

PRD §16: this file changes only by PR.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class Span(BaseModel):
    """Character offsets into `Document.full_text`. `end` is exclusive."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _end_not_before_start(self) -> Span:
        if self.end < self.start:
            raise ValueError(f"span end {self.end} is before start {self.start}")
        return self

    @property
    def length(self) -> int:
        return self.end - self.start


class Section(BaseModel):
    id: str
    heading: str
    level: int = Field(ge=0)
    span: Span
    paragraphs: list[Span] = Field(default_factory=list)

    @model_validator(mode="after")
    def _paragraphs_inside_section(self) -> Section:
        for i, para in enumerate(self.paragraphs):
            if para.start < self.span.start or para.end > self.span.end:
                raise ValueError(
                    f"paragraph {i} ({para.start},{para.end}) escapes "
                    f"section {self.id} ({self.span.start},{self.span.end})"
                )
        return self


class Document(BaseModel):
    doc_id: str
    source: str  # "pdf" | "web"
    title: str
    full_text: str
    sections: list[Section] = Field(default_factory=list)
    sentences: list[Span] = Field(default_factory=list)
    references_text: str | None = None
    structure_ok: bool = True
    token_count: int | None = None

    @model_validator(mode="after")
    def _spans_within_full_text(self) -> Document:
        limit = len(self.full_text)
        for span, label in self._all_spans():
            if span.end > limit:
                raise ValueError(
                    f"{label} span ({span.start},{span.end}) is out of bounds "
                    f"for full_text of length {limit}"
                )
        return self

    def _all_spans(self) -> list[tuple[Span, str]]:
        spans: list[tuple[Span, str]] = [(s, "sentence") for s in self.sentences]
        for section in self.sections:
            spans.append((section.span, f"section {section.id}"))
            spans.extend((p, f"paragraph of section {section.id}") for p in section.paragraphs)
        return spans

    def text_of(self, span: Span) -> str:
        """The exact source text a span refers to. SC2 depends on this."""
        return self.full_text[span.start : span.end]


class Route(str, Enum):
    A = "A"
    B = "B"


class SummaryNode(BaseModel):
    """A node in the reduce tree. level 0 = chunk summary. PRD §11.5, FR-3.4."""

    id: str
    level: int = Field(ge=0)
    parent_id: str | None = None
    text: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class KeyPoint(BaseModel):
    text: str
    rank: int
    cluster_size: int
    evidence: Span | None = None  # None = unattributed (FR-4.5)
    similarity: float | None = None

    @property
    def is_attributed(self) -> bool:
        return self.evidence is not None


class ClaimLabel(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


class ClaimResult(BaseModel):
    """One atomic claim, judged independently by both verifiers (FR-5.3)."""

    claim: str
    summary_sentence_idx: int = Field(ge=0)
    summary_node_id: str  # so per-merge-level faithfulness works (G5, E3)
    evidence: list[Span] = Field(default_factory=list)
    deberta_entail: float = Field(ge=0.0, le=1.0)
    deberta_contra: float = Field(ge=0.0, le=1.0)
    minicheck_support: float = Field(ge=0.0, le=1.0)
    label_deberta: ClaimLabel
    label_minicheck: ClaimLabel
