"""Unattended GPU chain for the 2026-11-02 deadline: demo bundles -> E1 -> E5 -> report.

    Start-Process -WindowStyle Hidden .venv\\Scripts\\python.exe scripts\\overnight.py

Runs detached from Claude Code so it survives the session ending. Every step is
resumable and skipped when already done, so starting it again is always safe:
  1. demo/<name>.json for each demo paper that is missing
  2. E1 unless a finished results/E1/<run_id> exists (an interrupted run resumes from
     results/E1/_running via the LLM cache and per_doc.jsonl)
  3. E5 on the newest finished E1 run, unless a finished E5 run exists
  3b. SC4: AggreFact test at the frozen thetas, unless its summary exists
  4. make report
Waits first while another GPU job (make_demo / eval.run / tune_aggrefact) is running.
Log: results/overnight.log (one line per step, plus each step's own output).
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOG = ROOT / "results" / "overnight.log"
DEMO = ("sbert", "frank", "true")
GPU_JOBS = ("make_demo.py", "eval.run", "tune_aggrefact.py")


def log(msg: str) -> None:
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def other_gpu_job() -> str | None:
    """Command line of another running GPU job, if any (not this process)."""
    out = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "ForEach-Object { $_.ProcessId.ToString() + ' ' + $_.CommandLine }",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    for line in out.splitlines():
        pid, _, cmd = line.partition(" ")
        mine = not pid.isdigit() or int(pid) == os.getpid() or "overnight.py" in cmd
        if not mine and any(j in cmd for j in GPU_JOBS):
            return cmd.strip()
    return None


def finished(exp: str) -> Path | None:
    base = ROOT / "results" / exp
    if not base.is_dir():
        return None
    runs = sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_"))
    return runs[-1] if runs else None


def run(step: str, args: list[str]) -> bool:
    log(f"START {step}: {' '.join(args)}")
    t0 = time.perf_counter()
    with LOG.open("a", encoding="utf-8") as f:
        rc = subprocess.run([PY, *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"END   {step}: exit {rc} after {time.perf_counter() - t0:.0f} s")
    return rc == 0


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log("overnight chain started")
    while (job := other_gpu_job()) is not None:
        log(f"waiting for running GPU job: {job[:120]}")
        time.sleep(60)

    missing = [n for n in DEMO if not (ROOT / "demo" / f"{n}.json").exists()]
    if missing:
        run("demo", ["scripts/make_demo.py", *missing, "--min-free-gb", "4"])
    else:
        log("demo: all bundles present, skipped")

    if finished("E1") is None:
        if not run("E1", ["-m", "eval.run", "--experiment", "E1"]):
            log("E1 failed; E5 needs it, stopping before E5 (report still runs)")
    else:
        log(f"E1: finished run {finished('E1')} present, skipped")

    e1 = finished("E1")
    if e1 is not None and finished("E5") is None:
        run("E5", ["-m", "eval.run", "--experiment", "E5", "--e1-dir", str(e1)])
    elif finished("E5") is not None:
        log(f"E5: finished run {finished('E5')} present, skipped")

    if not (ROOT / "results" / "tuning" / "aggrefact_test.summary.json").exists():
        run("SC4", ["scripts/tune_aggrefact.py", "--split", "test", "--min-free-gb", "3"])
    else:
        log("SC4: test summary present, skipped")

    run("report", ["-m", "eval.report"])
    log("overnight chain finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
