"""Token estimate. PRD FR-2.1, §11.5.

shortcut: chars/4 heuristic, not a real tokenizer. Measured against the Qwen3 tokenizer
(scripts/token_check.py, DECISION-33): FR-2.1's ±5% is NOT met. GROBID papers: always an
overestimate (+3..+16%, safe: errs toward Route B); ccdv arXiv/PubMed: -25..+36%.
Future work: the provider's own tokenizer for docs near tau.
"""

from __future__ import annotations

import math

# Room for the reply (+ the one-line JSON repair suffix) on top of the largest prompt.
REPLY_TOKENS = 1024


def small_context(
    prompts: list[str], choices: list[int], *, ceiling: int, margin: float = 0.10
) -> int | None:
    """Smallest window in `choices` (below `ceiling`) that holds every prompt plus a reply;
    None = use the profile's own window. One value per stage: changing the window between
    calls makes Ollama reload the model (DECISION-35)."""
    if not prompts:
        return None
    need = max(estimate_tokens(p, margin) for p in prompts) + REPLY_TOKENS
    return next((c for c in sorted(choices) if need <= c < ceiling), None)


def estimate_tokens(text: str, margin: float) -> int:
    # Round up: undercounting could push an over-budget doc into Route A.
    # round(…, 6) first so float noise (100 * 1.1 = 110.00000000000001) can't add a token.
    return math.ceil(round(len(text) / 4 * (1 + margin), 6))
