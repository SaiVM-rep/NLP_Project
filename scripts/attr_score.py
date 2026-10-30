"""Score key-point attribution against eval/gold/attribution_labels.csv.

    .venv/Scripts/python.exe scripts/attr_score.py

Reads results/attribution/<paper>.system.json (written by attr_sheet.py) and writes
results/attribution/summary.json: per paper + overall precision@1 and accuracy, plus each
miss (system pick vs reference) for error analysis.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.attribution import score_attribution  # noqa: E402

LABELS = ROOT / "eval" / "gold" / "attribution_labels.csv"
OUT = ROOT / "results" / "attribution"


def main() -> int:
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with LABELS.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            system = json.loads((OUT / f"{row['paper']}.system.json").read_text(encoding="utf-8"))
            by_paper[row["paper"]].append(
                {
                    "kp": int(row["kp"]),
                    "system": system[row["kp"]],
                    "gold": [int(x) for x in row["gold_ids"].split(";") if x],
                    "unsupported": row["unsupported"] == "1",
                    "notes": row["notes"],
                }
            )
    every = [it for items in by_paper.values() for it in items]
    summary = {
        "labels": str(LABELS.relative_to(ROOT)),
        "annotator": "claude (LLM), not human",
        "overall": score_attribution(every),
        "per_paper": {p: score_attribution(items) for p, items in sorted(by_paper.items())},
        "misses": [
            {"paper": p, **it}
            for p, items in sorted(by_paper.items())
            for it in items
            if not (it["system"] in it["gold"] or (it["system"] is None and it["unsupported"]))
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    o = summary["overall"]
    print(f"overall: n={o['n']} precision@1={o['precision_at_1']:.3f} accuracy={o['accuracy']:.3f}")
    for p, m in summary["per_paper"].items():
        print(f"  {p:6s} precision@1={m['precision_at_1']:.3f} ({m['n']} key points)")
    for m in summary["misses"]:
        miss = f"{m['paper']} KP{m['kp']}: system {m['system']} vs gold {m['gold']}"
        print(f"  MISS {miss}  {m['notes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
