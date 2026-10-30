"""Tune theta_e for DeBERTa and MiniCheck on AggreFact DEV (M7, SC4). Live: GPU + local weights.

    .venv/Scripts/python.exe scripts/tune_aggrefact.py [--per-dataset 400] [--seed 0]
    .venv/Scripts/python.exe scripts/tune_aggrefact.py --from-scores   # sweep only, no GPU

Scores a stratified dev sample once (results/tuning/aggrefact_dev.scores.jsonl), then sweeps
theta on the stored scores. Prints the proposed thetas; it does NOT edit configs/ (a human
freezes them, then SC4 is computed once on the TEST split). Test rows are never loaded here.
Refuses to start below --min-free-gb available RAM.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "tuning"
SCORES = OUT / "aggrefact_dev.scores.jsonl"
SUMMARY = OUT / "aggrefact_dev.summary.json"


def available_gb() -> float:
    class MemStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("load", ctypes.c_ulong),
            ("total_phys", ctypes.c_ulonglong),
            ("avail_phys", ctypes.c_ulonglong),
            ("total_page", ctypes.c_ulonglong),
            ("avail_page", ctypes.c_ulonglong),
            ("total_virt", ctypes.c_ulonglong),
            ("avail_virt", ctypes.c_ulonglong),
            ("avail_ext", ctypes.c_ulonglong),
        ]

    s = MemStatus()
    s.length = ctypes.sizeof(MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))  # type: ignore[attr-defined]
    return s.avail_phys / 2**30


def score(per_dataset: int, seed: int) -> None:
    from app.core.config import load_config
    from app.verify.pipeline import openers_from_config
    from eval.data import load_aggrefact
    from eval.tune import score_rows, stratified_sample

    cfg = load_config(ROOT / "configs").verify
    items = stratified_sample(load_aggrefact("dev"), per_dataset=per_dataset, seed=seed)
    print(f"scoring {len(items)} dev rows (per_dataset={per_dataset}, seed={seed})", flush=True)
    t0 = time.perf_counter()
    rows = score_rows(items, openers_from_config(cfg), cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    with SCORES.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r._asdict()) + "\n")
    print(f"scored in {time.perf_counter() - t0:.0f} s -> {SCORES}", flush=True)


def sweep() -> dict[str, object]:
    from eval.tune import balanced_accuracy, best_theta

    rows = [json.loads(ln) for ln in SCORES.read_text(encoding="utf-8").splitlines() if ln]
    truth = [r["label"] for r in rows]
    report: dict[str, object] = {"n_rows": len(rows)}
    for name, key in (("nli", "entail"), ("minicheck", "minicheck")):
        sc = [r[key] for r in rows]
        fit = best_theta(sc, truth)
        at_half = balanced_accuracy(truth, [s >= 0.5 for s in sc])
        per_ds: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            per_ds[r["dataset"]].append(i)
        by_ds = {
            ds: balanced_accuracy([truth[i] for i in ix], [sc[i] >= fit.theta for i in ix])
            for ds, ix in sorted(per_ds.items())
        }
        report[name] = {
            "theta_e": fit.theta,
            "balanced_accuracy": fit.balanced_accuracy,
            "balanced_accuracy_at_0.5": at_half,
            "per_dataset": by_ds,
        }
        print(
            f"{name:9s} theta_e={fit.theta:.2f}  BA={fit.balanced_accuracy:.3f}  "
            f"(at 0.5: {at_half:.3f})",
            flush=True,
        )
        for ds, ba in by_ds.items():
            print(f"    {ds:18s} {ba:.3f}", flush=True)
    SUMMARY.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"summary -> {SUMMARY}", flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-free-gb", type=float, default=4.0)
    ap.add_argument("--from-scores", action="store_true")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")  # HF_HOME, HF_TOKEN
    if not a.from_scores:
        free = available_gb()
        if free < a.min_free_gb:
            print(f"only {free:.1f} GB RAM available, need {a.min_free_gb:g}: close apps first")
            return 2
        score(a.per_dataset, a.seed)
    sweep()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
