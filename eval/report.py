"""`make report`: chapter-5 tables and figures from results/ logs (FR-7.6, SC5).

Reads the newest finished run of each experiment and the tuning summary, writes
results/report/tables.md and results/report/*.png. Nothing in chapter 5 is typed by hand.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.run import aggregate_e1, aggregate_e5

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _cell(x: Any) -> str:
    if isinstance(x, float):
        return "n/a" if math.isnan(x) else f"{x:.3f}"
    return str(x)


def md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def latest_run(base: Path) -> Path | None:
    """Newest finished run folder (run ids start with a UTC timestamp); `_running` is not."""
    if not base.is_dir():
        return None
    runs = sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_"))
    return runs[-1] if runs else None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def e1_table(rows: Sequence[dict[str, Any]]) -> str:
    return md_table(
        [
            "Arm",
            "Docs",
            "Faith. DeBERTa",
            "Faith. MiniCheck",
            "ROUGE-1",
            "ROUGE-2",
            "ROUGE-L",
            "Seconds/doc",
        ],
        [
            [
                a["arm"],
                a["n_docs"],
                a["faith_deberta"],
                a["faith_minicheck"],
                a["rouge1"],
                a["rouge2"],
                a["rougeL"],
                a["seconds"],
            ]
            for a in aggregate_e1(rows)
        ],
    )


def e3_levels(rows: Sequence[dict[str, Any]]) -> dict[str, dict[int, tuple[float, int]]]:
    """Route B docs only: verifier -> level -> (mean faithfulness, docs reaching that level)."""
    out: dict[str, dict[int, tuple[float, int]]] = {}
    for v in ("deberta", "minicheck"):
        acc: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            if r["arm"] != "B":
                continue
            for lvl, f in r[f"levels_{v}"].items():
                if f is not None and not math.isnan(f):
                    acc[int(lvl)].append(f)
        out[v] = {lvl: (round(sum(fs) / len(fs), 10), len(fs)) for lvl, fs in sorted(acc.items())}
    return out


def e3_table(rows: Sequence[dict[str, Any]]) -> str:
    lv = e3_levels(rows)
    levels = sorted(set(lv["deberta"]) | set(lv["minicheck"]))
    return md_table(
        ["Merge level", "Docs", "Faith. DeBERTa", "Faith. MiniCheck"],
        [
            [
                lvl,
                lv["deberta"].get(lvl, (math.nan, 0))[1],
                lv["deberta"].get(lvl, (math.nan, 0))[0],
                lv["minicheck"].get(lvl, (math.nan, 0))[0],
            ]
            for lvl in levels
        ],
    )


def e5_table(rows: Sequence[dict[str, Any]]) -> str:
    return md_table(
        ["Perturbation", "Items", "DeBERTa correct", "MiniCheck correct"],
        [
            [a["type"], a["n"], a["deberta_correct"], a["minicheck_correct"]]
            for a in aggregate_e5(rows)
        ],
    )


def verifier_table(summary: dict[str, Any]) -> str:
    return md_table(
        ["Verifier", "Tuned theta_e", "Bal. acc. at 0.5", "Bal. acc. tuned"],
        [
            [
                name,
                summary[key]["theta_e"],
                summary[key]["balanced_accuracy_at_0.5"],
                summary[key]["balanced_accuracy"],
            ]
            for name, key in (("DeBERTa", "nli"), ("MiniCheck", "minicheck"))
        ],
    )


def sc4_table(summary: dict[str, Any]) -> str:
    """SC4: AggreFact TEST at the frozen thetas ("supported" is the positive class)."""
    return md_table(
        ["Verifier", "theta_e", "Precision", "Recall", "F1", "Bal. acc."],
        [
            [
                name,
                summary[key]["theta_e"],
                summary[key]["precision"],
                summary[key]["recall"],
                summary[key]["f1"],
                summary[key]["balanced_accuracy"],
            ]
            for name, key in (("DeBERTa", "nli"), ("MiniCheck", "minicheck"))
        ],
    )


def _figures(
    e1: list[dict[str, Any]] | None, e5: list[dict[str, Any]] | None, out: Path
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    if e1:
        lv = e3_levels(e1)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for v, name in (("deberta", "DeBERTa"), ("minicheck", "MiniCheck")):
            xs = sorted(lv[v])
            ax.plot(xs, [lv[v][x][0] for x in xs], marker="o", label=name)
        ax.set_xlabel("Merge level (0 = chunk summaries)")
        ax.set_ylabel("Faithfulness")
        ax.set_ylim(0, 1)
        ax.set_title("E3: faithfulness per merge level")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "e3_levels.png", dpi=150)
        plt.close(fig)
        made.append("e3_levels.png")
    if e5:
        agg = aggregate_e5(e5)
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        pos = list(range(len(agg)))
        ax.bar([x - 0.2 for x in pos], [a["deberta_correct"] for a in agg], 0.4, label="DeBERTa")
        ax.bar(
            [x + 0.2 for x in pos], [a["minicheck_correct"] for a in agg], 0.4, label="MiniCheck"
        )
        ax.set_xticks(pos, [a["type"] for a in agg])
        ax.set_ylabel("Share judged correctly")
        ax.set_ylim(0, 1)
        ax.set_title("E5: verifier behaviour under perturbation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "e5_perturbations.png", dpi=150)
        plt.close(fig)
        made.append("e5_perturbations.png")
    return made


def main() -> int:
    out = RESULTS / "report"
    out.mkdir(parents=True, exist_ok=True)
    parts = ["# Chapter 5 tables (generated by `make report`; do not edit)\n"]

    tuning = RESULTS / "tuning" / "aggrefact_dev.summary.json"
    if tuning.exists():
        s = json.loads(tuning.read_text(encoding="utf-8"))
        parts += [
            f"## Verifier thresholds (AggreFact dev, {s['n_rows']} rows)\n",
            verifier_table(s),
            "",
        ]
    sc4 = RESULTS / "tuning" / "aggrefact_test.summary.json"
    if sc4.exists():
        t = json.loads(sc4.read_text(encoding="utf-8"))
        parts += [
            f"## SC4: verifiers on AggreFact test ({t['n_rows']} rows, frozen thetas)\n",
            sc4_table(t),
            "",
        ]

    e1: list[dict[str, Any]] | None = None
    e5: list[dict[str, Any]] | None = None
    if (e1_dir := latest_run(RESULTS / "E1")) is not None:
        e1 = _jsonl(e1_dir / "per_doc.jsonl")
        parts += [f"## E1: Route A vs forced Route B (run `{e1_dir.name}`)\n", e1_table(e1), ""]
        parts += ["## E3: faithfulness per merge level (Route B arm of E1)\n", e3_table(e1), ""]
    if (e5_dir := latest_run(RESULTS / "E5")) is not None:
        e5 = _jsonl(e5_dir / "per_item.jsonl")
        parts += [f"## E5: perturbations (run `{e5_dir.name}`)\n", e5_table(e5), ""]

    (out / "tables.md").write_text("\n".join(parts), encoding="utf-8")
    figs = _figures(e1, e5, out)
    print(f"wrote {out / 'tables.md'}" + (f" + {', '.join(figs)}" if figs else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
