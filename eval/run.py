"""P8 pilot runner (FR-7.2): `make exp E=E1|E5` -> results/<E>/<run_id>/.

E1: Route A vs forced Route B on the same papers (abstract removed; it is the ROUGE
reference, DECISION-24). Every merge level is verified, so E3 per-level numbers come from
the same runs. E5: perturb claims both verifiers supported in E1, re-verify, and count how
often each verifier gets the perturbed claim right (FR-7.5, PRD §11.9).

Rows are appended to per_doc.jsonl as they finish; a rerun into the same folder skips
what is done (with the LLM cache, a crash costs only the unfinished arm).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import sys
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from app.core.schemas import ClaimLabel, Document, Route
from app.ingest.spans import RawSection, build_document
from app.summarize.common import SummaryResult
from app.verify.pipeline import VerifiedSummary
from eval.metrics import faithfulness, faithfulness_by_level, rouge
from eval.perturb import ASSERTION, corrupt, negate

ROOT = Path(__file__).resolve().parents[1]
TEI = ROOT / "tests" / "fixtures" / "tei"
PILOT_DOCS = ("sbert", "frank", "true", "debertav3", "chatgpt_factuality")


@dataclass(frozen=True)
class ExpDoc:
    doc: Document
    reference: str  # the paper's own abstract, removed from the input


class ArmResult(NamedTuple):
    result: SummaryResult
    verified: VerifiedSummary
    seconds: float


RunArm = Callable[[Document, int | None], Awaitable[ArmResult]]


def exp_doc(doc_id: str, title: str, sections: Sequence[RawSection]) -> ExpDoc:
    abstract = [s for s in sections if s.heading.strip().lower() == "abstract"]
    if not abstract:
        raise ValueError(f"{doc_id}: no abstract section to use as the reference")
    body = [s for s in sections if s not in abstract]
    reference = " ".join(p for s in abstract for p in s.paragraphs)
    return ExpDoc(build_document(doc_id, "pdf", title, body, None), reference)


# ---------------------------------------------------------------------- E1 ---


def e1_row(doc_id: str, arm: str, a: ArmResult, *, reference: str) -> dict[str, Any]:
    """Headline faithfulness = the final summary (top level); every level kept for E3."""
    nodes, claims = a.verified.nodes, a.verified.claims
    level = {n.id: n.level for n in nodes}
    top = max(level.values(), default=0)
    final = [c for c in claims if level.get(c.summary_node_id) == top]
    return {
        "doc_id": doc_id,
        "arm": arm,
        "route": a.result.route.value,
        "tokens": a.result.tokens,
        "tau": a.result.tau,
        "n_nodes": len(nodes),
        "n_claims": len(final),
        "faith_deberta": faithfulness([c.label_deberta for c in final]),
        "faith_minicheck": faithfulness([c.label_minicheck for c in final]),
        "levels_deberta": {
            str(k): v for k, v in faithfulness_by_level(claims, nodes, "deberta").items()
        },
        "levels_minicheck": {
            str(k): v for k, v in faithfulness_by_level(claims, nodes, "minicheck").items()
        },
        **rouge(a.result.summary, reference),
        "summary_words": len(a.result.summary.split()),
        "seconds": a.seconds,
        "summary": a.result.summary,
        "claims": [
            {
                "claim": c.claim,
                "level": level[c.summary_node_id],
                "deberta": c.label_deberta.value,
                "minicheck": c.label_minicheck.value,
            }
            for c in claims
        ],
    }


def _done(out: Path) -> list[dict[str, Any]]:
    if not out.exists():
        return []
    return [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]


async def run_e1(
    docs: Sequence[ExpDoc], run_arm: RunArm, out: Path, *, tau_b: int
) -> list[dict[str, Any]]:
    rows = _done(out)
    have = {(r["doc_id"], r["arm"]) for r in rows}
    arms: tuple[tuple[str, int | None, Route], ...] = (
        ("A", None, Route.A),
        ("B", tau_b, Route.B),
    )
    for d in docs:
        for arm, tau, want in arms:
            if (d.doc.doc_id, arm) in have:
                continue
            a = await run_arm(d.doc, tau)
            if a.result.route is not want:
                raise RuntimeError(
                    f"{d.doc.doc_id} arm {arm}: took route {a.result.route.value}, "
                    f"expected {want.value} (tau {tau}, {a.result.tokens} tokens)"
                )
            row = e1_row(d.doc.doc_id, arm, a, reference=d.reference)
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            rows.append(row)
    return rows


E1_METRICS = (
    "faith_deberta",
    "faith_minicheck",
    "rouge1",
    "rouge2",
    "rougeL",
    "seconds",
    "summary_words",
)


def _mean(xs: Sequence[float]) -> float:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(vals) / len(vals) if vals else math.nan


def aggregate_e1(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    return [
        {
            "arm": arm,
            "n_docs": len(rs),
            **{m: _mean([r[m] for r in rs if m in r]) for m in E1_METRICS},
        }
        for arm, rs in sorted(by_arm.items())
    ]


# ---------------------------------------------------------------------- E5 ---


def e5_items(doc_id: str, claims: Sequence[str], *, per_type: int) -> list[dict[str, Any]]:
    """Rule-based perturbations; paraphrase is added live (it needs the LLM).

    `should_support` is what a correct verifier says about the perturbed claim.
    """
    items: list[dict[str, Any]] = []
    count: dict[str, int] = defaultdict(int)
    for c in claims:
        for kind, fn in (("negation", negate), ("corruption", corrupt)):
            out = fn(c)
            if out is not None and out != c and count[kind] < per_type:
                count[kind] += 1
                items.append(
                    {
                        "doc_id": doc_id,
                        "type": kind,
                        "original": c,
                        "perturbed": out,
                        "should_support": False,
                    }
                )
    items.append(
        {
            "doc_id": doc_id,
            "type": "assertion",
            "original": "",
            "perturbed": ASSERTION,
            "should_support": False,
        }
    )
    return items


def aggregate_e5(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    sup = ClaimLabel.SUPPORTED.value
    return [
        {
            "type": t,
            "n": len(rs),
            **{
                f"{v}_correct": sum((r[v] == sup) == r["should_support"] for r in rs) / len(rs)
                for v in ("deberta", "minicheck")
            },
        }
        for t, rs in sorted(by_type.items())
    ]


# -------------------------------------------------------------------- live ---


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _load_docs(names: Sequence[str]) -> list[ExpDoc]:
    from app.ingest.tei import parse_tei

    out = []
    for n in names:
        tei = parse_tei((TEI / f"{n}.tei.xml").read_bytes())
        out.append(exp_doc(n, tei.title, tei.sections))
    return out


async def _live_e1(cfg: Any, docs: list[ExpDoc], run_dir: Path, tau_b: int) -> None:
    from app.summarize.single_pass import summarize
    from app.worker.pipeline import LiveStages

    async def run_arm(doc: Document, tau: int | None) -> ArmResult:
        st = LiveStages(cfg)
        try:
            t0 = time.perf_counter()
            res = await summarize(doc, st.client, cfg, ROOT / "prompts", tau_override=tau)
            ver = await st.verify(doc, res)
            secs = time.perf_counter() - t0
        finally:
            await st.aclose()
        print(
            f"  {doc.doc_id} route {res.route.value}: {secs:.0f} s, {len(ver.claims)} claims",
            flush=True,
        )
        return ArmResult(res, ver, secs)

    rows = await run_e1(docs, run_arm, run_dir / "per_doc.jsonl", tau_b=tau_b)
    _write_csv(run_dir / "aggregate.csv", aggregate_e1(rows))


async def _live_e5(
    cfg: Any, docs: list[ExpDoc], run_dir: Path, e1_dir: Path, per_type: int
) -> None:
    from app.verify.decompose import Claim
    from app.verify.pipeline import openers_from_config, verify_claims
    from app.worker.pipeline import LiveStages
    from eval.perturb import paraphrase

    by_doc = {r["doc_id"]: r for r in _done(e1_dir / "per_doc.jsonl") if r["arm"] == "A"}
    out = run_dir / "per_item.jsonl"
    openers = openers_from_config(cfg.verify)
    sup = ClaimLabel.SUPPORTED.value
    rows: list[dict[str, Any]] = []
    for d in docs:
        r = by_doc.get(d.doc.doc_id)
        if r is None:
            print(f"  {d.doc.doc_id}: no E1 arm A row, skipped", flush=True)
            continue
        good = [c["claim"] for c in r["claims"] if c["deberta"] == sup and c["minicheck"] == sup]
        items = e5_items(d.doc.doc_id, good, per_type=per_type)
        st = LiveStages(cfg)
        try:
            for c in good[:per_type]:
                p = await paraphrase(c, st.client, ROOT / "prompts")
                if p is not None:
                    items.append(
                        {
                            "doc_id": d.doc.doc_id,
                            "type": "paraphrase",
                            "original": c,
                            "perturbed": p,
                            "should_support": True,
                        }
                    )
            await st._release()  # Qwen off the GPU before the verifiers load (DECISION-3)
        finally:
            await st.aclose()
        claims = [("e5", Claim(text=i["perturbed"], sentence_idx=0)) for i in items]
        results = verify_claims(d.doc, claims, openers, cfg.verify)
        with out.open("a", encoding="utf-8") as f:
            for i, res in zip(items, results, strict=True):
                row = {
                    **i,
                    "deberta": res.label_deberta.value,
                    "minicheck": res.label_minicheck.value,
                }
                f.write(json.dumps(row) + "\n")
                rows.append(row)
        print(f"  {d.doc.doc_id}: {len(items)} perturbed claims", flush=True)
    _write_csv(run_dir / "aggregate.csv", aggregate_e5(rows))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=["E1", "E5"])
    ap.add_argument("--docs", nargs="+", default=list(PILOT_DOCS))
    ap.add_argument("--tau-b", type=int, default=4000, help="E1 arm B budget (forces Route B)")
    ap.add_argument("--per-type", type=int, default=10, help="E5 perturbations per type per doc")
    ap.add_argument("--e1-dir", type=Path, help="E5: the E1 run folder to take claims from")
    a = ap.parse_args(argv)
    if a.experiment == "E5" and a.e1_dir is None:
        ap.error("E5 needs --e1-dir results/E1/<run_id>")

    from dotenv import load_dotenv

    from app.core.config import load_config
    from app.core.manifest import run_manifest

    load_dotenv(ROOT / ".env")
    cfg = load_config(ROOT / "configs")
    docs = _load_docs(a.docs)
    base = ROOT / "results" / a.experiment
    with run_manifest(base / "_running", cfg, a.experiment, ROOT / "prompts") as m:
        print(f"{a.experiment} run {m.run_id}", flush=True)
        run_dir = base / "_running"
        if a.experiment == "E1":
            asyncio.run(_live_e1(cfg, docs, run_dir, a.tau_b))
        else:
            asyncio.run(_live_e5(cfg, docs, run_dir, a.e1_dir, a.per_type))
    final = base / m.run_id
    run_dir.rename(final)
    print(f"done -> {final}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
