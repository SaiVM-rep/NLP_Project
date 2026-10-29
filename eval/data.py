"""FR-7.1 dataset loaders + `python -m eval.data --check`. PRD §12.1.

Every source is a parquet file at a pinned commit (FR-7.3): rows never change under a run.
IDs confirmed 2026-10-29, see CLAUDE.md §8. Downloads go to HF_HOME (outside the repo).

arXiv/PubMed (ccdv) are the Cohan et al. preprocessed text: lowercased, tokenised,
`@xmath`/`@xcite` placeholders, and **no section names** (the `section` config only
breaks lines). They become unstructured Documents (FR-1.5).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from pydantic import BaseModel, Field

from app.core.schemas import Document
from app.ingest.spans import RawSection, build_document

Fetch = Callable[[str, str, str], Path]  # (repo, filename, revision) -> local path


@dataclass(frozen=True)
class Source:
    repo: str
    revision: str
    files: dict[str, str]  # split -> parquet path in the repo
    license: str
    gated: bool = False
    expected_rows: dict[str, int] = field(default_factory=dict)


SOURCES: dict[str, Source] = {
    "arxiv": Source(
        repo="ccdv/arxiv-summarization",
        revision="240aaf1a969b3f8cd0ade6986bfad0cd730ee288",
        files={"test": "section/test-00000-of-00001.parquet"},
        license="unspecified (Cohan et al. 2018)",
        expected_rows={"test": 6440},
    ),
    "pubmed": Source(
        repo="ccdv/pubmed-summarization",
        revision="6b30a2cae59b11ed77cb19959bffccbbd18e1106",
        files={"test": "section/test-00000-of-00001.parquet"},
        license="unspecified (Cohan et al. 2018)",
        expected_rows={"test": 6658},
    ),
    "summeval": Source(
        repo="mteb/summeval",  # parquet mirror of the Yale-LILY release
        revision="bfc121155064afa2d81b5505682ffc0d96f4334c",
        files={"test": "data/test-00000-of-00001-35901af5f6649399.parquet"},
        license="mit",
        expected_rows={"test": 100},
    ),
    "aggrefact": Source(
        repo="lytang/LLM-AggreFact",
        revision="981dfd0bd8e58e7238a9ab92b2e6ea44bce918e4",
        files={
            "dev": "data/dev-00000-of-00001.parquet",
            "test": "data/test-00000-of-00001.parquet",
        },
        license="cc-by-nd-4.0",
        gated=True,
    ),
}

GOLD_DIR = Path("data/gold")
GOLD_SIZE = 50  # PRD §12.1


def hf_fetch(repo: str, filename: str, revision: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def _rows(name: str, split: str, fetch: Fetch) -> list[dict[str, Any]]:
    src = SOURCES[name]
    path = fetch(src.repo, src.files[split], src.revision)
    rows: list[dict[str, Any]] = pq.read_table(path).to_pylist()
    return rows


def _need(rows: list[dict[str, Any]], cols: list[str], name: str) -> None:
    have = list(rows[0]) if rows else []
    missing = [c for c in cols if c not in have]
    if rows and missing:
        raise ValueError(f"{name}: missing columns {missing}; found {have}")


# -------------------------------------------------------- arxiv / pubmed ---


class LongDoc(BaseModel):
    id: str
    article: str
    abstract: str  # the reference summary


def load_long_docs(
    name: str, split: str = "test", *, fetch: Fetch = hf_fetch, limit: int | None = None
) -> list[LongDoc]:
    rows = _rows(name, split, fetch)
    _need(rows, ["article", "abstract"], name)
    rows = rows[:limit] if limit is not None else rows
    return [
        LongDoc(id=f"{name}-{split}-{i:05d}", article=r["article"], abstract=r["abstract"])
        for i, r in enumerate(rows)
    ]


def to_document(doc: LongDoc, name: str) -> Document:
    lines = [ln.strip() for ln in doc.article.split("\n")]
    section = RawSection(heading="", level=1, paragraphs=[ln for ln in lines if ln])
    built = build_document(doc.id, f"dataset:{name}", "", [section], None)
    # Known property of the source, not a parse failure: flag it without the per-doc
    # FR-1.5 warning, which would otherwise fire once for every dataset document.
    return built.model_copy(update={"structure_ok": False})


# -------------------------------------------------------------- summeval ---

SUMMEVAL_DIMS = ("relevance", "coherence", "fluency", "consistency")


class SummEvalItem(BaseModel):
    doc_id: str
    summary_idx: int
    text: str
    summary: str
    references: list[str]  # human summaries, for reference-based metrics
    relevance: float
    coherence: float
    fluency: float
    consistency: float


def load_summeval(*, fetch: Fetch = hf_fetch) -> list[SummEvalItem]:
    rows = _rows("summeval", "test", fetch)
    _need(rows, ["id", "text", "machine_summaries", "human_summaries", *SUMMEVAL_DIMS], "summeval")
    out = []
    for r in rows:
        n = len(r["machine_summaries"])
        for dim in SUMMEVAL_DIMS:
            if len(r[dim]) != n:
                raise ValueError(
                    f"summeval {r['id']}: {len(r[dim])} {dim} scores for {n} summaries"
                )
        for i, summary in enumerate(r["machine_summaries"]):
            scores = {dim: float(r[dim][i]) for dim in SUMMEVAL_DIMS}
            out.append(
                SummEvalItem(
                    doc_id=r["id"],
                    summary_idx=i,
                    text=r["text"],
                    summary=summary,
                    references=list(r["human_summaries"]),
                    **scores,
                )
            )
    return out


# ------------------------------------------------------------- aggrefact ---


class AggreFactItem(BaseModel):
    dataset: str  # sub-benchmark, e.g. AggreFact-CNN; results are reported per dataset
    doc: str
    claim: str
    label: int = Field(ge=0, le=1)  # 1 = supported


def load_aggrefact(split: str, *, fetch: Fetch = hf_fetch) -> list[AggreFactItem]:
    rows = _rows("aggrefact", split, fetch)
    _need(rows, ["dataset", "doc", "claim", "label"], "aggrefact")
    return [
        AggreFactItem(dataset=r["dataset"], doc=r["doc"], claim=r["claim"], label=r["label"])
        for r in rows
    ]


# ----------------------------------------------------------------- check ---


class Status(str, Enum):
    OK = "OK"
    BLOCKED = "BLOCKED"  # needs a human action (e.g. accept gated terms)
    PENDING = "PENDING"  # not built yet (gold set before P8)
    FAIL = "FAIL"


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str


def _count(name: str, split: str, fetch: Fetch) -> int:
    if name in ("arxiv", "pubmed"):
        return len(load_long_docs(name, split, fetch=fetch))
    if name == "summeval":
        load_summeval(fetch=fetch)  # parses and validates every row
        return len(_rows(name, split, fetch))
    return len(load_aggrefact(split, fetch=fetch))


def check(
    name: str,
    *,
    fetch: Fetch = hf_fetch,
    expected_rows: dict[str, int] | None = None,
    gold_dir: Path = GOLD_DIR,
) -> CheckResult:
    if name == "gold":
        n = len(list(gold_dir.glob("*.pdf"))) if gold_dir.is_dir() else 0
        status = Status.OK if n == GOLD_SIZE else Status.PENDING
        return CheckResult(name, status, f"{n}/{GOLD_SIZE} PDFs in {gold_dir} (curated in P8)")

    from huggingface_hub.errors import GatedRepoError

    src = SOURCES[name]
    expected = src.expected_rows if expected_rows is None else expected_rows
    parts = []
    try:
        for split in src.files:
            n = _count(name, split, fetch)
            want = expected.get(split)
            if want is not None and n != want:
                return CheckResult(name, Status.FAIL, f"{split}: {n} rows, expected {want}")
            parts.append(f"{split}={n}")
    except GatedRepoError:
        return CheckResult(
            name,
            Status.BLOCKED,
            f"gated: log in, accept the terms at https://huggingface.co/datasets/{src.repo}, "
            "then put a read token in .env as HF_TOKEN",
        )
    except Exception as exc:  # report every dataset, don't stop at the first
        return CheckResult(name, Status.FAIL, f"{type(exc).__name__}: {exc}")
    return CheckResult(
        name, Status.OK, f"{', '.join(parts)} rows @ {src.revision[:7]} ({src.license})"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.data")
    ap.add_argument("--check", action="store_true", help="fetch, parse and count every dataset")
    ap.add_argument("--only", nargs="*", default=None, help="subset of dataset names")
    args = ap.parse_args(argv)
    if not args.check:
        ap.print_help()
        return 2
    from dotenv import load_dotenv

    load_dotenv()  # HF_HOME, HF_TOKEN
    names = args.only or [*SOURCES, "gold"]
    results = [check(n) for n in names]
    for r in results:
        print(f"{r.status.value:8} {r.name:10} {r.detail}")
    return 0 if all(r.status is Status.OK for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
