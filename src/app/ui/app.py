"""P7 page (FR-6.4, FR-6.5): two panes, offset highlighting, claim colours, demo mode.

Left: the source text. Right: header, summary, key points, claims. Picking a key point or
claim highlights its evidence span in the source and scrolls to it. Demo mode reads saved
bundles from disk, so it needs no GPU, database or network. Upload mode drives the P6 HTTP API.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import gradio as gr
import httpx

from app.ui.render import render_source
from app.ui.view import View, bundle_from_api, load_bundles, make_view

CSS = """
.src { white-space: pre-wrap; line-height: 1.5; font-size: 14px; max-height: 78vh;
       overflow-y: auto; padding: 12px; border: 1px solid var(--border-color-primary); }
.src mark { background: #ffe066; color: #111; padding: 0 1px; }
.pick label { align-items: flex-start; }
"""
SCROLL_JS = """() => { setTimeout(() => {
  const el = document.getElementById('active-hl');
  if (el) el.scrollIntoView({block: 'center', behavior: 'smooth'}); }, 80); }"""
LEGEND = (
    "Claims: first icon = DeBERTa, second = MiniCheck. "
    "\U0001f7e2 supported · \U0001f7e1 unsupported · \U0001f534 contradicted"
)
POLL_S = 2.0


def _show(v: View) -> tuple[Any, ...]:
    kp = [k.label for k in v.key_points]
    cl = [c.label for c in v.claims]
    return (
        v,
        f"### {v.title}\n{v.header}",
        v.summary,
        render_source(v.full_text, []),
        gr.update(choices=kp, value=None),
        gr.update(choices=cl, value=None),
    )


def _highlight(v: View | None, items: str, picked: str | None) -> str:
    if v is None:
        return ""
    pool = v.key_points if items == "kp" else v.claims
    spans = next((i.spans for i in pool if i.label == picked), [])
    return render_source(v.full_text, spans)


def build_app(demo_dir: Path, api_url: str) -> gr.Blocks:
    bundles = load_bundles(demo_dir)

    def open_demo(title: str | None) -> tuple[Any, ...]:
        if not title or title not in bundles:
            raise gr.Error("Pick an example first.")
        return _show(make_view(bundles[title]))

    def upload(path: str | None) -> Iterator[tuple[Any, ...]]:
        if not path:
            raise gr.Error("Choose a PDF first.")
        blank = (gr.update(),) * 6
        with httpx.Client(base_url=api_url, timeout=30) as c:
            try:
                with open(path, "rb") as f:
                    r = c.post("/v1/jobs", files={"file": (Path(path).name, f, "application/pdf")})
            except httpx.ConnectError as e:
                raise gr.Error(f"API not reachable at {api_url}: run `make api`.") from e
            if r.status_code != 202:
                raise gr.Error(
                    f"{r.json().get('title', 'Upload failed')}: {r.json().get('detail')}"
                )
            jid = r.json()["job_id"]
            while True:
                j = c.get(f"/v1/jobs/{jid}").json()
                pct = f"{(j.get('progress') or 0):.0%}"
                if j["status"] == "failed":
                    raise gr.Error(f"Job failed: {j.get('error')}")
                if j["status"] == "done":
                    break
                yield (*blank[:1], f"Running — {j.get('stage')} ({pct})", *blank[2:])
                time.sleep(POLL_S)
            yield _show(make_view(bundle_from_api(c, jid)))

    app = gr.Blocks(title="Semantic Compression")
    with app:
        state = gr.State(None)
        gr.Markdown("# Semantic compression & key points")
        with gr.Row(), gr.Column(scale=1, min_width=260):
            with gr.Tab("Demo examples"):
                pick = gr.Dropdown(sorted(bundles), label="Example", value=None)
                open_btn = gr.Button("Open", variant="primary")
            with gr.Tab("Upload a PDF"):
                pdf = gr.File(file_types=[".pdf"], type="filepath", label="PDF")
                run_btn = gr.Button("Summarize", variant="primary")
        head = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=3):
                source = gr.HTML(label="Source")
            with gr.Column(scale=2):
                summary = gr.Textbox(label="Summary", lines=6, interactive=False)
                kps = gr.Radio([], label="Key points (click to locate)", elem_classes="pick")
                gr.Markdown(LEGEND)
                claims = gr.Radio([], label="Claims (click to locate)", elem_classes="pick")

        outs = [state, head, summary, source, kps, claims]
        open_btn.click(open_demo, pick, outs)
        run_btn.click(upload, pdf, outs)
        kps.input(lambda v, p: _highlight(v, "kp", p), [state, kps], source).then(
            None, js=SCROLL_JS
        )
        claims.input(lambda v, p: _highlight(v, "claim", p), [state, claims], source).then(
            None, js=SCROLL_JS
        )
    return app
