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
    base.update(top_n_evidence=1)
    base["nli"].update(theta_e=0.5, theta_c=0.5)
    base["minicheck"]["theta_e"] = 0.5
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


def test_each_verifier_labels_with_its_own_threshold() -> None:
    """AggreFact-dev tuning yields one threshold pair per verifier; scores are not on one scale."""
    cfg = _cfg()
    cfg.nli.theta_e = 0.95  # cat entail 0.9 -> no longer supported by DeBERTa
    cfg.minicheck.theta_e = 0.9  # cat support 0.95 -> still supported by MiniCheck
    cat = verify_claims(_doc(), CLAIMS[:1], Rig().openers, cfg)[0]
    assert (cat.label_deberta, cat.label_minicheck) == (
        ClaimLabel.UNSUPPORTED,
        ClaimLabel.SUPPORTED,
    )


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


def test_real_openers_load_from_local_cache_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-5.6: HF loaders phone huggingface.co even for a pinned sha (46 DNS lookups seen
    2026-10-29). Verification must never touch the network; fetching is `make models`."""
    from app.verify import embedder, minicheck, nli
    from app.verify.pipeline import openers_from_config

    seen: dict[str, dict[str, Any]] = {}

    def recorder(name: str) -> Any:
        def from_pretrained(cls: Any, checkpoint: str, **kw: Any) -> str:
            seen[name] = kw
            return name

        return classmethod(from_pretrained)

    monkeypatch.setattr(embedder.SentenceEmbedder, "from_pretrained", recorder("embedder"))
    monkeypatch.setattr(nli.DebertaNLI, "from_pretrained", recorder("nli"))
    monkeypatch.setattr(minicheck.MiniCheckT5, "from_pretrained", recorder("minicheck"))

    o = openers_from_config(_cfg(), device="cpu")
    for open_ in (o.embedder, o.nli, o.minicheck):
        open_()
    assert {k: v.get("local_files_only") for k, v in seen.items()} == {
        "embedder": True,
        "nli": True,
        "minicheck": True,
    }


# ------------------------------------------------- whole summary, per level ---


def _result(**kw: Any) -> Any:
    from app.core.schemas import Route
    from app.summarize.common import SummaryResult

    base: dict[str, Any] = dict(
        doc_id="d",
        summary="Cats purr.",
        route=Route.A,
        tokens=10,
        tau=100,
        prompt_hashes={},
        model="m",
    )
    base.update(kw)
    return SummaryResult(**base)


def test_route_a_summary_is_one_level_zero_node() -> None:
    from app.verify.pipeline import summary_nodes

    assert summary_nodes(_result()) == [SummaryNode(id="root", level=0, text="Cats purr.")]


def test_route_b_verifies_every_node_of_the_merge_tree() -> None:
    from app.core.schemas import Route
    from app.verify.pipeline import summary_nodes

    tree = [
        SummaryNode(id="n0-0", level=0, parent_id="n1-0", text="Cats purr."),
        SummaryNode(id="n0-1", level=0, parent_id="n1-0", text="Birds are blue."),
        SummaryNode(id="n1-0", level=1, text="Cats purr; birds are blue."),
    ]
    r = _result(route=Route.B, summary=tree[-1].text, nodes=tree, root_id="n1-0")
    assert summary_nodes(r) == tree


async def test_verify_summary_gives_faithfulness_per_level_per_verifier(tmp_path: Path) -> None:
    """G5/E3: one call verifies a Route B result at every merge level."""
    from app.core.schemas import Route
    from app.verify.pipeline import verify_summary

    tree = [
        SummaryNode(id="n0-0", level=0, parent_id="n1-0", text="Cats purr."),
        SummaryNode(id="n0-1", level=0, parent_id="n1-0", text="Dogs are quiet."),
        SummaryNode(id="n1-0", level=1, text="Cats purr; dogs are quiet."),
    ]
    r = _result(route=Route.B, summary=tree[-1].text, nodes=tree, root_id="n1-0")
    cfg = load_config(ROOT / "configs")
    client = LLMClient(
        ScriptedProvider(
            '{"claims": ["A cat purrs."]}',
            '{"claims": ["The dog is silent."]}',
            '{"claims": ["A cat purrs."]}',
        ),
        cfg.profiles["qwen3_8b_local"],
        DiskCache(tmp_path),
    )
    v = await verify_summary(_doc(), r, client, Rig().openers, _cfg(), prompt_dir=ROOT / "prompts")
    from eval.metrics import faithfulness_by_level

    assert [c.summary_node_id for c in v.claims] == ["n0-0", "n0-1", "n1-0"]
    assert v.nodes == tree
    # cat supported by both; dog contradicted (DeBERTa) / unsupported (MiniCheck)
    for verifier in ("deberta", "minicheck"):
        assert faithfulness_by_level(v.claims, v.nodes, verifier) == {0: 0.5, 1: 1.0}


def test_minicheck_opener_prefers_the_local_bf16_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """3.1 GB fp32 .bin -> 1.5 GB mmap'd bf16 safetensors (live RAM low point 0.09 GB)."""
    from app.verify import minicheck
    from app.verify.pipeline import openers_from_config

    seen: list[tuple[str, dict[str, Any]]] = []

    def from_pretrained(cls: Any, checkpoint: str, **kw: Any) -> str:
        seen.append((checkpoint, kw))
        return "mc"

    monkeypatch.setattr(minicheck.MiniCheckT5, "from_pretrained", classmethod(from_pretrained))
    local = tmp_path / "mc-bf16"
    openers_from_config(_cfg(), device="cpu", minicheck_local=local).minicheck()
    local.mkdir()
    openers_from_config(_cfg(), device="cpu", minicheck_local=local).minicheck()

    (pinned_ckpt, kw0), (local_ckpt, kw1) = seen
    assert pinned_ckpt == _cfg().minicheck.checkpoint and kw0["revision"]  # fallback
    assert local_ckpt == str(local) and kw1["revision"] is None
    assert kw0["local_files_only"] is kw1["local_files_only"] is True


async def test_verify_summary_final_only_checks_just_the_root(tmp_path: Path) -> None:
    """Uploads (DECISION-37): the user reads the root; E3 keeps every level."""
    from app.core.schemas import Route
    from app.verify.pipeline import verify_summary

    tree = [
        SummaryNode(id="n0-0", level=0, parent_id="n1-0", text="Cats purr."),
        SummaryNode(id="n0-1", level=0, parent_id="n1-0", text="Dogs are quiet."),
        SummaryNode(id="n1-0", level=1, text="Cats purr; dogs are quiet."),
    ]
    r = _result(route=Route.B, summary=tree[-1].text, nodes=tree, root_id="n1-0")
    cfg = load_config(ROOT / "configs")
    p = ScriptedProvider('{"claims": ["A cat purrs."]}')
    client = LLMClient(p, cfg.profiles["qwen3_8b_local"], DiskCache(tmp_path))
    v = await verify_summary(
        _doc(), r, client, Rig().openers, _cfg(), prompt_dir=ROOT / "prompts", final_only=True
    )
    assert [c.summary_node_id for c in v.claims] == ["n1-0"]
    assert v.nodes == tree  # the tree is still stored and shown
    assert p.replies == []  # one scripted reply: a second decomposition would raise
