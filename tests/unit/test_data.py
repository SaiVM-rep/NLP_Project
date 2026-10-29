"""FR-7.1 dataset loaders. Synthetic parquet + injected fetch: no network (DECISION-9)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eval.data import (
    SOURCES,
    AggreFactItem,
    Status,
    check,
    load_aggrefact,
    load_long_docs,
    load_summeval,
    to_document,
)


def _parquet(tmp_path: Path, name: str, rows: dict[str, list[Any]]) -> Path:
    p = tmp_path / name
    pq.write_table(pa.table(rows), p)
    return p


class FakeFetch:
    def __init__(self, files: dict[str, Path], error: Exception | None = None) -> None:
        self.files = files
        self.error = error
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, repo: str, filename: str, revision: str) -> Path:
        self.calls.append((repo, filename, revision))
        if self.error:
            raise self.error
        return self.files[repo]


ARXIV = {
    "article": [
        "first line of paper one . \n second line @xcite . \n\n  \n third line .",
        "paper two only line .",
        "paper three .",
    ],
    "abstract": ["abs one .", "abs two .", "abs three ."],
}


# --------------------------------------------------------------- sources ---


def test_every_source_is_pinned_to_a_commit_sha() -> None:
    for name, src in SOURCES.items():
        assert len(src.revision) == 40 and int(src.revision, 16) >= 0, name
        assert src.revision != "main"


def test_fetch_uses_the_pinned_repo_revision_and_file(tmp_path: Path) -> None:
    fetch = FakeFetch({SOURCES["arxiv"].repo: _parquet(tmp_path, "a.parquet", ARXIV)})
    load_long_docs("arxiv", "test", fetch=fetch)
    src = SOURCES["arxiv"]
    assert fetch.calls == [(src.repo, src.files["test"], src.revision)]


# -------------------------------------------------------- arxiv / pubmed ---


def test_long_docs_get_stable_ids_and_keep_text(tmp_path: Path) -> None:
    fetch = FakeFetch({SOURCES["arxiv"].repo: _parquet(tmp_path, "a.parquet", ARXIV)})
    docs = load_long_docs("arxiv", "test", fetch=fetch)
    assert [d.id for d in docs] == ["arxiv-test-00000", "arxiv-test-00001", "arxiv-test-00002"]
    assert docs[1].abstract == "abs two ."


def test_limit_takes_the_first_rows_deterministically(tmp_path: Path) -> None:
    fetch = FakeFetch({SOURCES["arxiv"].repo: _parquet(tmp_path, "a.parquet", ARXIV)})
    assert [d.id for d in load_long_docs("arxiv", "test", fetch=fetch, limit=2)] == [
        "arxiv-test-00000",
        "arxiv-test-00001",
    ]


def test_to_document_lines_become_paragraphs_with_exact_spans(tmp_path: Path) -> None:
    fetch = FakeFetch({SOURCES["arxiv"].repo: _parquet(tmp_path, "a.parquet", ARXIV)})
    doc = to_document(load_long_docs("arxiv", "test", fetch=fetch)[0], "arxiv")
    paras = [doc.text_of(p) for s in doc.sections for p in s.paragraphs]
    assert paras == ["first line of paper one .", "second line @xcite .", "third line ."]
    assert doc.source == "dataset:arxiv"
    assert doc.doc_id == "arxiv-test-00000"


def test_to_document_is_flagged_unstructured() -> None:
    """ccdv data has no section names (measured 2026-10-29): never pretend it does (FR-1.5)."""
    from eval.data import LongDoc

    doc = to_document(LongDoc(id="x", article="a .\nb .", abstract="s"), "pubmed")
    assert doc.structure_ok is False
    assert [s.heading for s in doc.sections] == [""]


def test_to_document_does_not_warn_per_doc(caplog: pytest.LogCaptureFixture) -> None:
    """No headings is a known property of the dataset, not a parse failure: FR-1.5's
    warning is for parsers that lose structure. 1000 docs logged 1000 warnings."""
    import logging

    from eval.data import LongDoc

    with caplog.at_level(logging.WARNING):
        to_document(LongDoc(id="x", article="a .\nb .", abstract="s"), "arxiv")
    assert caplog.records == []


# -------------------------------------------------------------- summeval ---


def test_summeval_flattens_to_one_item_per_machine_summary(tmp_path: Path) -> None:
    rows = {
        "id": ["d1", "d2"],
        "text": ["source one", "source two"],
        "machine_summaries": [["s1a", "s1b"], ["s2a", "s2b"]],
        "human_summaries": [["h"], ["h"]],
        "relevance": [[1.0, 2.0], [3.0, 4.0]],
        "coherence": [[1.5, 2.5], [3.5, 4.5]],
        "fluency": [[5.0, 5.0], [4.0, 4.0]],
        "consistency": [[1.0, 5.0], [2.0, 3.0]],
    }
    fetch = FakeFetch({SOURCES["summeval"].repo: _parquet(tmp_path, "s.parquet", rows)})
    items = load_summeval(fetch=fetch)
    assert len(items) == 4
    it = items[3]
    assert (it.doc_id, it.summary_idx, it.summary, it.text) == ("d2", 1, "s2b", "source two")
    assert (it.relevance, it.coherence, it.fluency, it.consistency) == (4.0, 4.5, 4.0, 3.0)


def test_summeval_items_keep_human_reference_summaries(tmp_path: Path) -> None:
    """ROUGE/BERTScore on SummEval are reference-based: the 11 human summaries are needed."""
    rows = {
        "id": ["d1"],
        "text": ["t"],
        "machine_summaries": [["a"]],
        "human_summaries": [["ref one", "ref two"]],
        "relevance": [[1.0]],
        "coherence": [[1.0]],
        "fluency": [[1.0]],
        "consistency": [[1.0]],
    }
    fetch = FakeFetch({SOURCES["summeval"].repo: _parquet(tmp_path, "s.parquet", rows)})
    assert load_summeval(fetch=fetch)[0].references == ["ref one", "ref two"]


def test_summeval_misaligned_scores_fail_loudly(tmp_path: Path) -> None:
    rows = {
        "id": ["d1"],
        "text": ["t"],
        "machine_summaries": [["a", "b"]],
        "human_summaries": [["h"]],
        "relevance": [[1.0]],  # one score for two summaries
        "coherence": [[1.0, 1.0]],
        "fluency": [[1.0, 1.0]],
        "consistency": [[1.0, 1.0]],
    }
    fetch = FakeFetch({SOURCES["summeval"].repo: _parquet(tmp_path, "s.parquet", rows)})
    with pytest.raises(ValueError, match="relevance"):
        load_summeval(fetch=fetch)


# ------------------------------------------------------------- aggrefact ---


def _aggrefact(tmp_path: Path, **over: list[Any]) -> FakeFetch:
    rows: dict[str, list[Any]] = {
        "dataset": ["AggreFact-CNN", "ExpertQA"],
        "doc": ["doc one", "doc two"],
        "claim": ["claim one", "claim two"],
        "label": [1, 0],
    }
    rows.update(over)
    return FakeFetch({SOURCES["aggrefact"].repo: _parquet(tmp_path, "g.parquet", rows)})


def test_aggrefact_items(tmp_path: Path) -> None:
    items = load_aggrefact("dev", fetch=_aggrefact(tmp_path))
    assert items == [
        AggreFactItem(dataset="AggreFact-CNN", doc="doc one", claim="claim one", label=1),
        AggreFactItem(dataset="ExpertQA", doc="doc two", claim="claim two", label=0),
    ]


def test_aggrefact_rejects_non_binary_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_aggrefact("dev", fetch=_aggrefact(tmp_path, label=[1, 2]))


def test_aggrefact_missing_column_names_what_it_found(tmp_path: Path) -> None:
    rows = {"dataset": ["x"], "document": ["d"], "claim": ["c"], "label": [1]}
    fetch = FakeFetch({SOURCES["aggrefact"].repo: _parquet(tmp_path, "g.parquet", rows)})
    with pytest.raises(ValueError, match=r"\['doc'\].*'document'"):
        load_aggrefact("dev", fetch=fetch)


def test_aggrefact_has_no_train_split() -> None:
    """Thresholds tune on dev, report on test (PRD §11.8); there is nothing else."""
    assert set(SOURCES["aggrefact"].files) == {"dev", "test"}


# ----------------------------------------------------------------- check ---


def test_check_ok_reports_rows(tmp_path: Path) -> None:
    fetch = FakeFetch({SOURCES["arxiv"].repo: _parquet(tmp_path, "a.parquet", ARXIV)})
    r = check("arxiv", fetch=fetch, expected_rows={"test": 3})
    assert r.status is Status.OK
    assert "3" in r.detail


def test_check_row_count_mismatch_fails(tmp_path: Path) -> None:
    fetch = FakeFetch({SOURCES["arxiv"].repo: _parquet(tmp_path, "a.parquet", ARXIV)})
    r = check("arxiv", fetch=fetch, expected_rows={"test": 6440})
    assert r.status is Status.FAIL
    assert "6440" in r.detail


def test_check_gated_without_access_is_blocked_with_instructions() -> None:
    import httpx
    from huggingface_hub.errors import GatedRepoError

    resp = httpx.Response(401, request=httpx.Request("GET", "https://huggingface.co/x"))
    fetch = FakeFetch({}, error=GatedRepoError("401 gated", response=resp))
    r = check("aggrefact", fetch=fetch)
    assert r.status is Status.BLOCKED
    assert "huggingface.co/datasets/lytang/LLM-AggreFact" in r.detail
    assert "HF_TOKEN" in r.detail


def test_check_gold_set_pending_until_curated(tmp_path: Path) -> None:
    r = check("gold", gold_dir=tmp_path / "gold")
    assert r.status is Status.PENDING
