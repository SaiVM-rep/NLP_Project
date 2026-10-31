"""Token estimate. PRD FR-2.1, §11.5.

shortcut: chars/4 heuristic, not a real tokenizer. Measured against the Qwen3 tokenizer
(scripts/token_check.py, DECISION-33): FR-2.1's ±5% is NOT met. GROBID papers: always an
overestimate (+3..+16%, safe: errs toward Route B); ccdv arXiv/PubMed: -25..+36%.
Future work: the provider's own tokenizer for docs near tau.
"""

from __future__ import annotations

import math


def estimate_tokens(text: str, margin: float) -> int:
    # Round up: undercounting could push an over-budget doc into Route A.
    # round(…, 6) first so float noise (100 * 1.1 = 110.00000000000001) can't add a token.
    return math.ceil(round(len(text) / 4 * (1 + margin), 6))
