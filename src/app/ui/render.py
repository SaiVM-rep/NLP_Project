"""P7 rendering (FR-6.4, FR-6.5). Pure functions: no Gradio, no network.

The source pane is `full_text` and nothing else; highlights are character spans over it
(SC2), so what is marked is exactly `full_text[start:end]`.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from typing import Any

from app.core.schemas import ClaimLabel

_ICONS = {
    ClaimLabel.SUPPORTED: "\U0001f7e2",  # green
    ClaimLabel.UNSUPPORTED: "\U0001f7e1",  # yellow
    ClaimLabel.CONTRADICTED: "\U0001f534",  # red
}


def label_icon(label: ClaimLabel) -> str:
    return _ICONS[label]


def claim_choice(deberta: ClaimLabel, minicheck: ClaimLabel, claim: str) -> str:
    """One list entry: DeBERTa icon, MiniCheck icon, claim text."""
    return f"{label_icon(deberta)}{label_icon(minicheck)} {claim}"


def _merged(spans: Sequence[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    """Valid spans only (inside the text, non-empty), sorted, overlaps joined."""
    ok = sorted((s, e) for s, e in spans if 0 <= s < e <= n)
    out: list[tuple[int, int]] = []
    for s, e in ok:
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def render_source(full_text: str, spans: Sequence[tuple[int, int]]) -> str:
    """HTML of the whole document with `spans` marked; the first mark is the scroll target."""
    parts: list[str] = []
    pos = 0
    for i, (s, e) in enumerate(_merged(spans, len(full_text))):
        parts.append(html.escape(full_text[pos:s]))
        tag = '<mark id="active-hl">' if i == 0 else "<mark>"
        parts.append(f"{tag}{html.escape(full_text[s:e])}</mark>")
        pos = e
    parts.append(html.escape(full_text[pos:]))
    return f'<div class="src">{"".join(parts)}</div>'


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def render_objects(objects: Sequence[dict[str, Any]]) -> str:
    """Markdown for the "Tables & formulas" tab (DECISION-40): each item shows a crop of the
    printed page (always correct) beside Docling's extraction (may contain errors)."""
    if not objects:
        return "No tables or formulas were extracted for this document."
    tables = sum(o["kind"] == "table" for o in objects)
    head = (
        f"**{_plural(tables, 'table')} · {_plural(len(objects) - tables, 'formula')}** — "
        "left: the original page (exact); right: extracted by Docling "
        "(dense tables can come out scrambled; trust the left side)."
    )
    parts = [head]
    for o in objects:
        name = "Table" if o["kind"] == "table" else "Formula"
        title = f"{name} · page {o['page']}"
        if o.get("caption"):
            title += f" · {html.escape(o['caption'])}"
        body = o["content"].strip()
        if o["kind"] == "formula":
            body = f"$${body}$$" if body else "_Docling could not read this formula._"
        elif not body:
            body = "_Docling could not read this table._"
        parts.append(
            f'<div class="obj obj-{o["kind"]}"><div class="obj-title">{title}</div>'
            '<div class="obj-row">'
            f'<div class="obj-crop"><div class="obj-tag">Original</div>'
            f'<img src="data:image/png;base64,{o["image_png"]}" alt="{name} as printed"></div>'
            f'<div class="obj-ext"><div class="obj-tag">Extracted</div>\n\n{body}\n\n</div>'
            "</div></div>"
        )
    return "\n\n".join(parts)
