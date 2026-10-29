"""Verify stage orchestration. PRD §11.8 steps 1-6, FR-5.1..5.4, DECISION-3.

Fakes stand in for every model; what is under test is ordering, wiring and labels.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from app.core.config import VerifyCfg, load_config
from app.core.schemas import ClaimLabel, Document, Span, SummaryNode
from app.llm.base import LLMResult
from app.llm.cache import DiskCache
from app.llm.client import LLMClient
from app.verify.decompose import Claim
from app.verify.nli import LabelIdx
from app.verify.pipeline import Openers, verify_claims, verify_nodes

ROOT = Path(__file__).resolve().parents[2]

SENTS = ["Cats purr softly.", "Dogs bark loudly.", "Birds sing at dawn."]
TEXT = " ".join(SENTS)


def _doc() -> Document:
    spans, pos = [], 0
    for s in SENTS:
        spans.append(Span(start=pos, end=pos + len(s)))
        pos += len(s) + 1
    return Document(doc_id="d", source="pdf", title="t", full_text=TEXT, sentences=spans)


def _cfg(**kw: Any) -> VerifyCfg:
    base = load_config(ROOT / "configs").verify.model_dump()
    base.update(top_n_evidence=1, theta_e=0.5, theta_c=0.5)
    base["minicheck"]["chunk_words"] = 3  # one sentence per chunk in this toy doc
    base.update(kw)
    return VerifyCfg(**base)


AXES = ["cat", "dog", "bird"]


def fake_embed(texts: list[str]) -> NDArray[np.float32]:
    return np.array(
        [[1.0 if a in t.lower() else 0.0 for a in AXES] for t in texts], dtype=np.float32
    )


# (premise, claim) -> [entail, neutral, contra]
NLI = {
    ("Cats purr softly.", "A cat purrs."): [0.9, 0.05, 0.05],
    ("Dogs bark loudly.", "The dog is silent."): [0.05, 0.05, 0.9],
    ("Birds sing at dawn.", "The bird is blue."): [0.1, 0.8, 0.1],
}
# claim -> support prob per chunk, in chunk order
MC = {"A cat purrs.": [0.95, 0.1, 0.1], "The dog is silent.": [0.1, 0.2, 0.1]}


class FakeNli:
    label_idx = LabelIdx(entail=0, contra=2)

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str]]] = []

    def __call__(self, premises: list[str], hyps: list[str]) -> NDArray[np.float32]:
        self.calls.append((premises, hyps))
        return np.array(
            [NLI.get((p, h), [0.0, 1.0, 0.0]) for p, h in zip(premises, hyps, strict=True)],
            dtype=np.float32,
        )


class FakeMc:
    def __init__(self) -> None:
        self.chunks_seen: list[list[str]] = []

    def __call__(self, chunks: list[str], claims: list[str]) -> NDArray[np.float32]:
        self.chunks_seen.append(chunks)
        scores = MC.get(claims[0], [0.0] * len(chunks))
        return np.array(scores[: len(chunks)], dtype=np.float32)


class Rig:
    """Openers that log open/close so residency can be asserted."""

    def __init__(self, fail_in: str | None = None) -> None:
        self.events: list[str] = []
        self.nli = FakeNli()
        self.mc = FakeMc()
        self.fail_in = fail_in

    def _open(self, name: str, obj: Any) -> Any:
        @contextmanager
        def cm() -> Iterator[Any]:
            self.events.append(f"open {name}")
            try:
                if self.fail_in == name:
                    raise RuntimeError(f"boom in {name}")
                yield obj
            finally:
                self.events.append(f"close {name}")

        return cm

    @property
    def openers(self) -> Openers:
        return Openers(
            embedder=self._open("embedder", fake_embed),
            nli=self._open("nli", self.nli),
            minicheck=self._open("minicheck", self.mc),
        )


CLAIMS = [
    ("n0", Claim(text="A cat purrs.", sentence_idx=0)),
    ("n0", Claim(text="The dog is silent.", sentence_idx=1)),
    ("n1", Claim(text="The bird is blue.", sentence_idx=0)),
]


def test_one_result_per_claim_with_provenance() -> None:
    out = verify_claims(_doc(), CLAIMS, Rig().openers, _cfg())
    assert [(r.summary_node_id, r.summary_sentence_idx, r.claim) for r in out] == [
        ("n0", 0, "A cat purrs."),
        ("n0", 1, "The dog is silent."),
        ("n1", 0, "The bird is blue."),
    ]


def test_evidence_spans_come_from_the_embedder() -> None:
    d = _doc()
    out = verify_claims(d, CLAIMS, Rig().openers, _cfg())
    assert [[d.text_of(s) for s in r.evidence] for r in out] == [
        ["Cats purr softly."],
        ["Dogs bark loudly."],
        ["Birds sing at dawn."],
    ]


def test_scores_and_labels_per_verifier() -> None:
    out = verify_claims(_doc(), CLAIMS, Rig().openers, _cfg())
    cat, dog, bird = out
    assert (cat.deberta_entail, cat.minicheck_support) == pytest.approx((0.9, 0.95))
    assert (cat.label_deberta, cat.label_minicheck) == (ClaimLabel.SUPPORTED,) * 2
    assert dog.deberta_contra == pytest.approx(0.9)
    assert dog.label_deberta == ClaimLabel.CONTRADICTED
    # MiniCheck is single-score: it can say unsupported, never contradicted
    assert dog.label_minicheck == ClaimLabel.UNSUPPORTED
    assert (bird.label_deberta, bird.label_minicheck) == (ClaimLabel.UNSUPPORTED,) * 2


def test_nli_premises_are_evidence_sentences() -> None:
    rig = Rig()
    verify_claims(_doc(), CLAIMS[:1], rig.openers, _cfg())
    assert rig.nli.calls == [(["Cats purr softly."], ["A cat purrs."])]


def test_minicheck_sees_whole_document_chunks_not_just_evidence() -> None:
    rig = Rig()
    verify_claims(_doc(), CLAIMS[:1], rig.openers, _cfg())
    assert rig.mc.chunks_seen == [SENTS]


def test_one_model_resident_at_a_time() -> None:
    rig = Rig()
    verify_claims(_doc(), CLAIMS, rig.openers, _cfg())
    assert rig.events == [
        "open embedder",
        "close embedder",
        "open nli",
        "close nli",
        "open minicheck",
        "close minicheck",
    ]


def test_no_claims_loads_no_model() -> None:
    rig = Rig()
    assert verify_claims(_doc(), [], rig.openers, _cfg()) == []
    assert rig.events == []


def test_failure_still_closes_the_model_and_stops() -> None:
    rig = Rig(fail_in="nli")
    with pytest.raises(RuntimeError, match="boom in nli"):
        verify_claims(_doc(), CLAIMS, rig.openers, _cfg())
    assert rig.events == ["open embedder", "close embedder", "open nli", "close nli"]


# ------------------------------------------------------- with decomposition ---


class ScriptedProvider:
    name = "ollama"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)

    async def generate(self, prompt, *, model, params, seed):  # type: ignore[no-untyped-def]
        return LLMResult(
            text=self.replies.pop(0), tokens_in=1, tokens_out=1, provider="ollama", model=model
        )


async def test_verify_nodes_decomposes_each_node_then_releases_llm_before_gpu(
    tmp_path: Path,
) -> None:
    cfg = load_config(ROOT / "configs")
    client = LLMClient(
        ScriptedProvider('{"claims": ["A cat purrs."]}', '{"claims": ["The bird is blue."]}'),
        cfg.profiles["qwen3_8b_local"],
        DiskCache(tmp_path),
    )
    nodes = [
        SummaryNode(id="n0", level=0, text="Cats purr."),
        SummaryNode(id="n1", level=1, text="Birds are blue."),
    ]
    rig = Rig()

    async def release() -> None:
        rig.events.append("release llm")

    out = await verify_nodes(
        _doc(), nodes, client, rig.openers, _cfg(), release_llm=release, prompt_dir=ROOT / "prompts"
    )
    assert [(r.summary_node_id, r.claim) for r in out] == [
        ("n0", "A cat purrs."),
        ("n1", "The bird is blue."),
    ]
    assert rig.events[:2] == ["release llm", "open embedder"]
