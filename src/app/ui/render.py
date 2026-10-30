"""P7 rendering (FR-6.4, FR-6.5). Pure functions: no Gradio, no network.

The source pane is `full_text` and nothing else; highlights are character spans over it
(SC2), so what is marked is exactly `full_text[start:end]`.
"""

from __future__ import annotations

import html
from collections.abc import Sequence

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
