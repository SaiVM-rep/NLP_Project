"""FR-7.5 / PRD §11.9: adversarial perturbations applied to summaries before verification (E5).

Each returns None when it cannot apply, so the runner skips the item instead of
scoring an unchanged text as perturbed. Applicability rate is itself reported.
Negation is rule-based; PRD wants it LLM-checked afterwards, which is the runner's job.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.llm.client import LLMClient

ASSERTION = "The summary is fully supported by the document."

_AUX = re.compile(
    r"\b(is|are|was|were|can|could|will|would|should|has|have|had|does|do|did)\b(\s+not\b)?",
    re.IGNORECASE,
)

_SWAP = {
    "increase": "decrease",
    "increases": "decreases",
    "increased": "decreased",
    "higher": "lower",
    "better": "worse",
    "more": "less",
    "larger": "smaller",
    "faster": "slower",
    "improves": "degrades",
    "improved": "degraded",
    "outperforms": "underperforms",
    "outperformed": "underperformed",
}
_SWAP |= {v: k for k, v in _SWAP.items()}
_DIRECTION = re.compile(r"\b(" + "|".join(sorted(_SWAP, key=len, reverse=True)) + r")\b", re.I)
# Not part of an identifier: "ROUGE-1", "T5", "v2" stay as they are.
_NUMBER = re.compile(r"(?<![\w.-])\d+(?:\.\d+)?(?!\w)")


def append_assertion(text: str) -> str:
    return f"{text.rstrip()} {ASSERTION}"


def negate(claim: str) -> str | None:
    m = _AUX.search(claim)
    if m is None:
        return None
    aux, has_not = m.group(1), m.group(2)
    return claim[: m.start()] + (aux if has_not else f"{aux} not") + claim[m.end() :]


def _swap_word(m: re.Match[str]) -> str:
    word = m.group(0)
    out = _SWAP[word.lower()]
    return out.capitalize() if word[0].isupper() else out


def _double(m: re.Match[str]) -> str:
    s = m.group(0)
    decimals = len(s.split(".")[1]) if "." in s else 0
    value = float(s) * 2 or 1
    return f"{value:.{decimals}f}"


def corrupt(claim: str) -> str | None:
    """Swap the first direction word; failing that, double the first number."""
    for pattern, repl in ((_DIRECTION, _swap_word), (_NUMBER, _double)):
        out, n = pattern.subn(repl, claim, count=1)
        if n:
            return out
    return None


async def paraphrase(
    text: str, client: LLMClient, prompt_dir: str | Path = "prompts"
) -> str | None:
    template = (Path(prompt_dir) / "paraphrase.v1.txt").read_text(encoding="utf-8")
    prompt = template.replace("{text}", text)
    for attempt in (prompt, prompt + '\n\nReply with only valid JSON: {"text": "..."}'):
        res = await client.generate(attempt, purpose="paraphrase", temperature=0.0)
        body = res.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        try:
            out = json.loads(body).get("text")
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(out, str) and out.strip():
            return out.strip() if out.strip() != text.strip() else None
    return None
