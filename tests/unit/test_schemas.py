"""Core data model. PRD §11.1.

SC2 is the requirement under test: every stored span must slice back out of
`full_text` to exactly the text it claims to cover, for 100% of spans. If this
is not enforced at the schema level it will be violated silently somewhere in
ingestion and discovered at demo time.
"""

import pytest
from pydantic import ValidationError

from app.core.schemas import (
    ClaimLabel,
    ClaimResult,
    Document,
    KeyPoint,
    Route,
    Section,
    Span,
    SummaryNode,
)

TEXT = "Introduction\n\nTransformers are large.\n\nMethod\n\nWe used a GPU."


def _doc(**overrides):
    kwargs = {
        "doc_id": "d1",
        "source": "pdf",
        "title": "A Paper",
        "full_text": TEXT,
        "sections": [],
        "sentences": [],
    }
    kwargs.update(overrides)
    return Document(**kwargs)


# --------------------------------------------------------------- Span ---


def test_span_rejects_end_before_start():
    with pytest.raises(ValidationError):
        Span(start=10, end=4)


def test_span_rejects_negative_start():
    with pytest.raises(ValidationError):
        Span(start=-1, end=4)


def test_empty_span_is_allowed():
    assert Span(start=5, end=5).length == 0


def test_span_reports_its_length():
    assert Span(start=3, end=10).length == 7


# ----------------------------------------------------------- Document ---


def test_text_of_returns_the_exact_slice():
    doc = _doc()
    span = Span(start=0, end=12)
    assert doc.text_of(span) == "Introduction"


def test_document_rejects_span_past_end_of_full_text():
    with pytest.raises(ValidationError, match="out of bounds"):
        _doc(sentences=[Span(start=0, end=len(TEXT) + 1)])


def test_document_accepts_span_ending_exactly_at_end_of_text():
    doc = _doc(sentences=[Span(start=0, end=len(TEXT))])
    assert doc.text_of(doc.sentences[0]) == TEXT


def test_section_spans_are_bounds_checked_too():
    with pytest.raises(ValidationError, match="out of bounds"):
        _doc(
            sections=[
                Section(
                    id="s1",
                    heading="Introduction",
                    level=1,
                    span=Span(start=0, end=9999),
                    paragraphs=[],
                )
            ]
        )


def test_paragraph_must_lie_inside_its_section():
    with pytest.raises(ValidationError, match="paragraph"):
        _doc(
            sections=[
                Section(
                    id="s1",
                    heading="Introduction",
                    level=1,
                    span=Span(start=0, end=12),
                    paragraphs=[Span(start=0, end=40)],
                )
            ]
        )


def test_round_trip_every_span_slices_to_its_text():
    """SC2 on a realistic document."""
    sections = [
        Section(
            id="s1",
            heading="Introduction",
            level=1,
            span=Span(start=0, end=37),
            paragraphs=[Span(start=14, end=37)],
        ),
        Section(
            id="s2",
            heading="Method",
            level=1,
            span=Span(start=39, end=len(TEXT)),
            paragraphs=[Span(start=47, end=len(TEXT))],
        ),
    ]
    doc = _doc(sections=sections)

    assert doc.text_of(sections[0].paragraphs[0]) == "Transformers are large."
    assert doc.text_of(sections[1].paragraphs[0]) == "We used a GPU."
    for section in doc.sections:
        assert doc.text_of(section.span).startswith(section.heading)


def test_structure_ok_defaults_true_and_can_be_set_false():
    assert _doc().structure_ok is True
    assert _doc(structure_ok=False).structure_ok is False


def test_references_are_not_part_of_full_text():
    """FR-1.6: references live in metadata, never in the summarization input."""
    doc = _doc(references_text="[1] Vaswani et al.")
    assert "Vaswani" not in doc.full_text
    assert doc.references_text == "[1] Vaswani et al."


# -------------------------------------------------------------- enums ---


def test_route_values():
    assert (Route.A.value, Route.B.value) == ("A", "B")


def test_claim_label_values():
    assert {c.value for c in ClaimLabel} == {"supported", "contradicted", "unsupported"}


# ------------------------------------------------- summary / keypoint ---


def test_summary_node_level_zero_is_a_chunk_summary():
    node = SummaryNode(id="n1", level=0, parent_id=None, text="t", source_chunk_ids=["c1"])
    assert node.level == 0 and node.parent_id is None


def test_summary_node_rejects_negative_level():
    with pytest.raises(ValidationError):
        SummaryNode(id="n1", level=-1, parent_id=None, text="t", source_chunk_ids=[])


def test_keypoint_without_evidence_is_unattributed():
    """FR-4.5: below theta_attr the point carries no span."""
    kp = KeyPoint(text="p", rank=1, cluster_size=2, evidence=None, similarity=0.1)
    assert kp.is_attributed is False


def test_keypoint_with_evidence_is_attributed():
    kp = KeyPoint(text="p", rank=1, cluster_size=2, evidence=Span(start=0, end=5), similarity=0.9)
    assert kp.is_attributed is True


def test_claim_result_carries_both_verifier_labels_separately():
    """FR-5.3: DeBERTa and MiniCheck results are stored separately, never merged."""
    claim = ClaimResult(
        claim="c",
        summary_sentence_idx=0,
        summary_node_id="n1",
        evidence=[Span(start=0, end=5)],
        deberta_entail=0.9,
        deberta_contra=0.01,
        minicheck_support=0.4,
        label_deberta=ClaimLabel.SUPPORTED,
        label_minicheck=ClaimLabel.UNSUPPORTED,
    )
    assert claim.label_deberta != claim.label_minicheck


def test_claim_result_rejects_probability_outside_zero_one():
    with pytest.raises(ValidationError):
        ClaimResult(
            claim="c",
            summary_sentence_idx=0,
            summary_node_id="n1",
            evidence=[],
            deberta_entail=1.4,
            deberta_contra=0.0,
            minicheck_support=0.5,
            label_deberta=ClaimLabel.SUPPORTED,
            label_minicheck=ClaimLabel.SUPPORTED,
        )
