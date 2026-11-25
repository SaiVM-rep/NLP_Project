"""P7 view model: a finished job (+ document text) -> what the page shows."""

import json
from pathlib import Path

import httpx
import pytest

from app.ui.view import Bundle, bundle_from_api, load_bundles, make_view

TEXT = "SBERT uses triplet loss. It is fast. Training took 20 minutes."

RESULT = {
    "job_id": "j1",
    "doc_id": "d1",
    "route": "A",
    "tokens": 900,
    "tau": 13107,
    "model": "qwen3:8b",
    "summary": {"text": "SBERT is fast.", "nodes": []},
    "key_points": [
        {
            "rank": 1,
            "text": "Fast.",
            "cluster_size": 3,
            "evidence": {"start": 25, "end": 36},
            "similarity": 0.9,
        },
        {"rank": 2, "text": "Lost one.", "cluster_size": 1, "evidence": None, "similarity": 0.3},
    ],
    "claims": [
        {
            "claim": "SBERT uses triplet loss.",
            "summary_node_id": "n",
            "sentence_idx": 0,
            "evidence": [{"start": 0, "end": 24}],
            "deberta_entail": 0.9,
            "deberta_contra": 0.0,
            "minicheck_support": 0.95,
            "label_deberta": "supported",
            "label_minicheck": "supported",
        },
        {
            "claim": "Training took a day.",
            "summary_node_id": "n",
            "sentence_idx": 0,
            "evidence": [{"start": 37, "end": 62}],
            "deberta_entail": 0.1,
            "deberta_contra": 0.8,
            "minicheck_support": 0.2,
            "label_deberta": "contradicted",
            "label_minicheck": "unsupported",
        },
    ],
    "metrics": {
        "compression_ratio": 0.06,
        "faithfulness_deberta": 0.5,
        "faithfulness_minicheck": 0.5,
    },
}
DOC = {
    "doc_id": "d1",
    "title": "SBERT",
    "full_text": TEXT,
    "structure_ok": True,
    "sections": [],
    "sentences": [],
}


def _bundle() -> Bundle:
    return Bundle(title="SBERT", document=DOC, result=RESULT)


def test_key_points_become_entries_with_their_evidence_span() -> None:
    v = make_view(_bundle())
    assert v.key_points[0].label.startswith("1.")
    assert v.key_points[0].spans == [(25, 36)]


def test_unattributed_key_point_has_no_span_and_says_so() -> None:
    v = make_view(_bundle())
    assert v.key_points[1].spans == []
    assert "unattributed" in v.key_points[1].label.lower()


def test_claims_carry_icons_for_both_verifiers_and_evidence_spans() -> None:
    v = make_view(_bundle())
    assert v.claims[1].spans == [(37, 62)]
    assert v.claims[1].label.endswith("Training took a day.")
    assert v.claims[0].label[:2] != v.claims[1].label[:2]  # DeBERTa icon differs


def test_header_shows_route_model_and_both_faithfulness_numbers() -> None:
    h = make_view(_bundle()).header
    assert "Route A" in h and "qwen3:8b" in h and "50%" in h


def test_view_has_the_summary_and_source_text() -> None:
    v = make_view(_bundle())
    assert v.summary == "SBERT is fast." and v.full_text == TEXT


def test_load_bundles_reads_every_json_in_the_folder(tmp_path: Path) -> None:
    (tmp_path / "b.json").write_text(_bundle().model_dump_json(), encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")
    assert list(load_bundles(tmp_path)) == ["SBERT"]


def test_load_bundles_on_missing_folder_is_empty(tmp_path: Path) -> None:
    assert load_bundles(tmp_path / "nope") == {}


def test_bundle_from_api_fetches_result_then_document_text() -> None:
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        if req.url.path.endswith("/result"):
            return httpx.Response(200, json=RESULT)
        return httpx.Response(200, json=DOC)

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t") as c:
        b = bundle_from_api(c, "j1")
    assert seen == ["/v1/jobs/j1/result", "/v1/documents/d1/text"]
    assert b.title == "SBERT"


def test_bundle_from_api_raises_on_a_failed_job() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"code": "job_failed", "detail": "boom"})

    with (
        httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t") as c,
        pytest.raises(RuntimeError, match="job_failed"),
    ):
        bundle_from_api(c, "j1")


def test_bundle_round_trips_through_json() -> None:
    b = _bundle()
    assert Bundle.model_validate(json.loads(b.model_dump_json())) == b


@pytest.mark.parametrize("verify", [True, False])
@pytest.mark.parametrize("tables", [True, False])
def test_submit_job_posts_the_pdf_with_the_verify_and_tables_options(
    tmp_path: Path, verify: bool, tables: bool
) -> None:
    # verify=False is the low-RAM live-demo path: no MiniCheck/DeBERTa load in the worker.
    from app.ui.view import submit_job

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 x")
    seen: dict[str, bytes] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = req.read()
        return httpx.Response(202, json={"job_id": "j9"})

    with httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler)) as c:
        assert submit_job(c, str(pdf), verify=verify, tables=tables) == "j9"
    body = seen["body"]
    assert b'filename="paper.pdf"' in body and b"%PDF-1.4 x" in body
    assert json.dumps({"verify": verify, "tables": tables}).encode() in body


def test_submit_job_raises_the_problem_detail_on_rejection(tmp_path: Path) -> None:
    from app.ui.view import submit_job

    pdf = tmp_path / "big.pdf"
    pdf.write_bytes(b"%PDF")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(413, json={"title": "Too large", "detail": "over 20 MB"})

    with (
        httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler)) as c,
        pytest.raises(RuntimeError, match="Too large: over 20 MB"),
    ):
        submit_job(c, str(pdf), verify=True)


def test_view_carries_the_tables_and_formulas_tab() -> None:
    obj = {"kind": "table", "page": 2, "caption": "Table 1", "content": "| a |", "image_png": "AA"}
    v = make_view(Bundle(title="SBERT", document=DOC, result={**RESULT, "objects": [obj]}))
    assert "data:image/png;base64,AA" in v.objects
    assert "No tables or formulas" in _bundle_view_without_objects().objects


def _bundle_view_without_objects():  # type: ignore[no-untyped-def]
    return make_view(_bundle())  # bundles made before DECISION-40 have no "objects"
