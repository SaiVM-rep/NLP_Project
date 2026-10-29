"""Token estimate. PRD FR-2.1, §11.5.

shortcut: chars/4 heuristic, not a real tokenizer; FR-2.1's ±5% is unverified.
Upgrade to the provider's countTokens for docs near tau before E1-E4 (P8).
"""

from __future__ import annotations

import math


def estimate_tokens(text: str, margin: float) -> int:
    # Round up: undercounting could push an over-budget doc into Route A.
    # round(…, 6) first so float noise (100 * 1.1 = 110.00000000000001) can't add a token.
    return math.ceil(round(len(text) / 4 * (1 + margin), 6))
