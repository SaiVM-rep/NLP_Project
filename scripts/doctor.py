"""Environment readiness check. Read-only, makes no paid API calls.

Run this first, and any time something stops working:

    make doctor

Every line tells you the exact fix. Nothing here costs quota.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

rows: list[tuple[str, str, str, str]] = []  # status, name, detail, fix


def good(name: str, detail: str) -> None:
    rows.append(("OK", name, detail, ""))


def warn(name: str, detail: str, fix: str) -> None:
    rows.append(("WARN", name, detail, fix))


def bad(name: str, detail: str, fix: str) -> None:
    rows.append(("MISS", name, detail, fix))


def check_python() -> None:
    v = sys.version_info
    text = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) == (3, 11):
        good("python", text)
    else:
        bad("python", text, "run through .venv: uv venv --python 3.11 && uv sync --group dev")


def check_env_file() -> None:
    env = ROOT / ".env"
    if not env.exists():
        bad(".env", "missing", "cp .env.example .env   (then paste your key)  [step M2]")
        return
    load_dotenv(env)
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        bad("GEMINI_API_KEY", "empty", "paste a key from aistudio.google.com/apikey  [step M2]")
    elif len(key) < 20:
        warn("GEMINI_API_KEY", f"suspiciously short ({len(key)} chars)", "check you copied it all")
    else:
        good("GEMINI_API_KEY", f"set ({len(key)} chars, not validated here)")


def check_ollama() -> None:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    try:
        resp = httpx.get(f"{url}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except Exception as exc:
        bad("ollama", f"unreachable at {url} ({type(exc).__name__})", "start Ollama, then retry")
        return

    models = [m["name"] for m in resp.json().get("models", [])]
    good("ollama", f"up at {url}")
    wanted = "qwen3:8b"
    if any(m.startswith("qwen3") for m in models):
        match = next(m for m in models if m.startswith("qwen3"))
        good("  model", match)
    else:
        bad("  model", f"{wanted} not pulled", f"ollama pull {wanted}   [step M3]")
    if models:
        print(f"{DIM}        models present: {', '.join(models)}{RESET}")


def check_gpu() -> None:
    if not shutil.which("nvidia-smi"):
        warn("gpu", "nvidia-smi not found", "CPU-only: verification will be slow")
        return
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip()
    except Exception as exc:
        warn("gpu", f"nvidia-smi failed ({type(exc).__name__})", "")
        return

    name, total, used = (p.strip() for p in out.split(",")[:3])
    free = int(total) - int(used)
    detail = f"{name}, {free} MiB free of {total} MiB"
    if int(total) < 10000:
        # CLAUDE.md DEVIATION-1: the PRD assumed 12 GB, this machine has 8.
        warn("gpu", detail, "8 GB: load generator and verifiers sequentially (DECISION-3)")
    else:
        good("gpu", detail)


def check_docker() -> None:
    if not shutil.which("docker"):
        bad("docker", "not installed", "needed for grobid + postgres from P1")
        return
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=25, check=True)
        good("docker", "daemon running")
    except Exception:
        bad("docker", "daemon not running", "start Docker Desktop")


def check_http(name: str, url: str, fix: str) -> None:
    try:
        httpx.get(url, timeout=5.0).raise_for_status()
        good(name, f"up at {url}")
    except Exception as exc:
        warn(name, f"unreachable ({type(exc).__name__})", fix)


def check_onedrive() -> None:
    if "OneDrive" not in str(ROOT):
        good("onedrive", "project is outside OneDrive")
        return
    venv = ROOT / ".venv"
    size_mb = 0
    if venv.exists():
        size_mb = sum(f.stat().st_size for f in venv.rglob("*") if f.is_file()) // (1024 * 1024)
    warn(
        "onedrive",
        f"project is inside OneDrive (.venv is {size_mb} MB)",
        "exclude .venv/ data/ results/ in OneDrive settings  [step M1]",
    )


def main() -> int:
    print("=" * 74)
    print("DOCTOR - environment readiness (no API calls, nothing costs quota)")
    print("=" * 74)

    check_python()
    check_env_file()
    check_ollama()
    check_gpu()
    check_docker()
    check_http("grobid", "http://localhost:8070/api/isalive", "make up   (needed from P1)")
    check_onedrive()

    print()
    colour = {"OK": GREEN, "WARN": YELLOW, "MISS": RED}
    for status, name, detail, fix in rows:
        print(f"{colour[status]}{status:>5}{RESET}  {name:<16} {detail}")
        if fix:
            print(f"{DIM}        fix: {fix}{RESET}")

    missing = [r for r in rows if r[0] == "MISS"]
    print()
    if missing:
        print(f"{RED}{len(missing)} blocking item(s).{RESET} Fix those, then: make doctor")
        return 1
    print(f"{GREEN}No blocking items.{RESET} Next: make smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
