"""P7 view model: finished job + document text -> the lists and strings the page shows.

A `Bundle` is the demo-mode file format (one JSON per example) and also what a live job is
turned into, so both modes render through the same code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from app.core.schemas import ClaimLabel
from app.ui.render import claim_choice


class Bundle(BaseModel):
    title: str
    document: dict[str, Any]  # GET /v1/documents/{id}/text
    result: dict[str, Any]  # GET /v1/jobs/{id}/result


@dataclass(frozen=True)
class Item:
    label: str  # the list entry shown to the user
    spans: list[tuple[int, int]]  # what to highlight in the source when picked


@dataclass(frozen=True)
class View:
    title: str
    header: str
    summary: str
    full_text: str
    key_points: list[Item]
    claims: list[Item]


def _spans(raw: Any) -> list[tuple[int, int]]:
    if not raw:
        return []
    items = raw if isinstance(raw, list) else [raw]
    return [(int(s["start"]), int(s["end"])) for s in items]


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def make_view(b: Bundle) -> View:
    r = b.result
    m = r.get("metrics", {})
    header = (
        f"Route {r['route']} · {r['model']} · {r['tokens']} tokens · "
        f"faithfulness DeBERTa {_pct(m.get('faithfulness_deberta'))} / "
        f"MiniCheck {_pct(m.get('faithfulness_minicheck'))} · "
        f"compression {_pct(m.get('compression_ratio'))}"
    )
    kps = [
        Item(
            label=(
                f"{k['rank']}. {k['text']}"
                if k.get("evidence")
                else f"{k['rank']}. {k['text']} (unattributed)"
            ),
            spans=_spans(k.get("evidence")),
        )
        for k in r["key_points"]
    ]
    claims = [
        Item(
            label=claim_choice(
                ClaimLabel(c["label_deberta"]), ClaimLabel(c["label_minicheck"]), c["claim"]
            ),
            spans=_spans(c.get("evidence")),
        )
        for c in r["claims"]
    ]
    return View(
        title=b.title,
        header=header,
        summary=r["summary"]["text"],
        full_text=b.document["full_text"],
        key_points=kps,
        claims=claims,
    )


def load_bundles(folder: Path) -> dict[str, Bundle]:
    """Demo mode: every `*.json` bundle in `folder`, keyed by title. Missing folder = none."""
    if not folder.is_dir():
        return {}
    out: dict[str, Bundle] = {}
    for p in sorted(folder.glob("*.json")):
        b = Bundle.model_validate_json(p.read_text(encoding="utf-8"))
        out[b.title] = b
    return out


def bundle_from_api(client: httpx.Client, job_id: str) -> Bundle:
    """Live mode / demo export: result, then the document text it points at."""
    res = client.get(f"/v1/jobs/{job_id}/result")
    if res.status_code != 200:
        body = res.json()
        raise RuntimeError(f"{body.get('code', res.status_code)}: {body.get('detail')}")
    result = res.json()
    doc = client.get(f"/v1/documents/{result['doc_id']}/text")
    doc.raise_for_status()
    document = doc.json()
    return Bundle(title=document["title"], document=document, result=result)


def submit_job(client: httpx.Client, path: str, *, verify: bool) -> str:
    """POST /v1/jobs with the PDF; `verify=False` skips claim checking (low-RAM live demo)."""
    with open(path, "rb") as f:
        r = client.post(
            "/v1/jobs",
            files={"file": (Path(path).name, f, "application/pdf")},
            data={"options": json.dumps({"verify": verify})},
        )
    if r.status_code != 202:
        body = r.json()
        raise RuntimeError(f"{body.get('title', 'Upload failed')}: {body.get('detail')}")
    return str(r.json()["job_id"])
