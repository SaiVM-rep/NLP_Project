"""FR-2.1 check: chars/4 estimate vs the Qwen3 tokenizer (pilot papers, 50 arXiv, 50 PubMed).

    .venv/Scripts/python.exe scripts/token_check.py   -> results/token_check.json

Downloads only the Qwen/Qwen3-8B tokenizer files (public). Signed error = (estimate - real) / real.
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from transformers import AutoTokenizer  # noqa: E402

from app.route.tokens import estimate_tokens  # noqa: E402
from eval.data import load_long_docs  # noqa: E402
from eval.run import PILOT_DOCS, _load_docs  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
sets = {"pilot (GROBID)": [d.doc.full_text for d in _load_docs(PILOT_DOCS)]}
for name in ("arxiv", "pubmed"):
    sets[name] = [d.article for d in load_long_docs(name, limit=50)]

out = {}
for name, texts in sets.items():
    errs = []
    for t in texts:
        real = len(tok(t, add_special_tokens=False)["input_ids"])
        errs.append((estimate_tokens(t, 0.0) - real) / real)
    within = sum(abs(e) <= 0.05 for e in errs)
    out[name] = {
        "n": len(errs),
        "median_err": statistics.median(errs),
        "min_err": min(errs),
        "max_err": max(errs),
        "within_5pct": within,
    }
    print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[name].items()})
(ROOT / "results" / "token_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
