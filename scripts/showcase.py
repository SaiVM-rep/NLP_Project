"""One command for the live showcase: start everything, check a PDF, rehearse it.

    make showcase                    # start + supervise everything; Ctrl+C stops it all
    make showcase-check FILE=x.pdf   # parse only: size, route, expected time (needs Docker)
    make rehearse FILE=x.pdf         # full run through the API (start `make showcase` first),
                                     # then save it as a demo example (demo/<name>.json)

`start` brings up Ollama, Docker (GROBID + Postgres), migrations, API, worker and UI as
child processes with logs in results/showcase/, waits until /healthz is green, prints a
free-RAM verdict for the "Check claims" box and then supervises: a crashed child is
restarted, and if free RAM stays under GUARD_FLOOR_MB the worker is killed, its running job
is failed with a readable message, and a fresh worker is started (DECISION-34). That keeps
the laptop responsive instead of paging itself to a halt in front of an audience.

Rehearsing a PDF the day before also fills the LLM cache: the live upload of the same file
then skips every Qwen call and only parses + verifies (minutes -> about one minute).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOGS = ROOT / "results" / "showcase"
API = "http://127.0.0.1:8000"
UI = "http://127.0.0.1:7860"
DOCKER_EXE = Path(r"C:\Program Files\Docker\Docker\Docker Desktop.exe")
# Free RAM (GB, all services idle) needed before ticking "Check claims". Measured
# 2026-10-31 with VS Code + Claude Code open: a first (uncached) full run of a new 8k-token
# paper drew 1.86 GB (2.53 -> 0.67 GB free), a cached repeat 1.8 GB; + GUARD_FLOOR_MB +
# ~0.7 GB margin.
CLAIMS_MIN_GB = 3.0
GUARD_FLOOR_MB = 400
# Whole qwen3:8b on the 8 GB GPU even at 16k: 13 -> 33 tok/s (DECISION-37). Also set as
# Windows user variables; these are used when this launcher has to start Ollama itself.
OLLAMA_ENV = {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"}
OOM_ERROR = (
    "Stopped: the laptop ran out of RAM. Untick 'Check claims' and upload again, "
    "or close other programs first."
)
CHILDREN = {
    "api": [PY, "-m", "uvicorn", "app.api.main:app", "--port", "8000"],
    "worker": [PY, "-m", "app.worker"],
    "ui": [PY, "-m", "app.ui"],
}
# command-line fragments of our own services, to clear stale copies before starting
STALE = ("-m app.ui", "-m app.worker", "uvicorn app.api.main:app")


# ---------------------------------------------------------------- decision logic (tested)


def ram_verdict(avail_gb: float) -> tuple[bool, str]:
    if avail_gb >= CLAIMS_MIN_GB:
        return True, f"{avail_gb:.1f} GB free: full pipeline OK, leave 'Check claims' ticked."
    return False, (
        f"{avail_gb:.1f} GB free (< {CLAIMS_MIN_GB:.0f} GB): untick 'Check claims' for live "
        "uploads, or close more programs and check again. Claim checking is still shown "
        "on the demo examples."
    )


class MemoryGuard:
    """Trips when free RAM stays under `floor_mb` for `samples` readings in a row."""

    def __init__(self, floor_mb: float = GUARD_FLOOR_MB, samples: int = 3) -> None:
        self.floor_mb, self.samples, self._low = floor_mb, samples, 0

    def update(self, avail_mb: float) -> bool:
        self._low = self._low + 1 if avail_mb < self.floor_mb else 0
        return self._low >= self.samples

    def reset(self) -> None:
        self._low = 0


@dataclass(frozen=True)
class PdfVerdict:
    route: str
    ok: bool
    message: str


def pdf_verdict(tokens: int, tau: int) -> PdfVerdict:
    if tokens <= 0:
        return PdfVerdict("-", False, "GROBID found no text: scanned/image-only PDF? Pick another.")
    if tokens <= tau:  # FR-2.2
        return PdfVerdict(
            "A",
            True,
            f"Route A (single pass, {tokens} <= {tau} tokens). First run ~5-10 min with "
            "claim checking; a repeat of the same file ~1-2 min (cached).",
        )
    return PdfVerdict(
        "B",
        False,
        # scale: PEGASUS 49,720 tokens took 23 min end to end (DECISION-37)
        f"Route B (map-reduce, {tokens} > {tau} tokens): first run about "
        f"{max(10, round(tokens / 2200))} min with claim checking. Prewarm it ahead "
        "(make rehearse): the live repeat is then a few minutes.",
    )


# ---------------------------------------------------------------- machine helpers


class _MemStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def avail_mb() -> float:
    m = _MemStatus()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))  # type: ignore[attr-defined]
    return m.ullAvailPhys / 2**20


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ok(url: str, timeout: float = 3) -> bool:
    try:
        return httpx.get(url, timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def wait_for(what: str, check: Any, limit_s: float) -> None:
    t0 = time.time()
    while not check():
        if time.time() - t0 > limit_s:
            raise SystemExit(f"FAILED: {what} not ready after {limit_s:.0f} s")
        time.sleep(2)
    say(f"{what} ready ({time.time() - t0:.0f} s)")


def run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False, **kw)


def docker_up() -> bool:
    return run(["docker", "info", "--format", "{{.ServerVersion}}"]).returncode == 0


def ensure_docker() -> None:
    if not docker_up():
        if not DOCKER_EXE.exists():
            raise SystemExit("FAILED: Docker Desktop not found; start it by hand.")
        say("starting Docker Desktop ...")
        subprocess.Popen([str(DOCKER_EXE)])
        wait_for("Docker engine", docker_up, 240)
    r = run(["docker", "compose", "up", "-d", "db", "grobid"])
    if r.returncode != 0:
        raise SystemExit(f"FAILED: docker compose up\n{r.stderr}")
    wait_for("GROBID", lambda: ok("http://localhost:8070/api/isalive"), 180)


def ensure_ollama() -> None:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    if not ok(f"{url}/api/tags"):
        say("starting Ollama ...")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
            env={**os.environ, **OLLAMA_ENV},
        )
        wait_for("Ollama", lambda: ok(f"{url}/api/tags"), 60)
    tags = httpx.get(f"{url}/api/tags", timeout=5).json()
    if not any(m["name"].startswith("qwen3:8b") for m in tags.get("models", [])):
        raise SystemExit("FAILED: Ollama has no qwen3:8b (ollama pull qwen3:8b, needs internet)")
    say("Ollama ready, qwen3:8b present")


def stop_stale() -> None:
    """Kill earlier copies of our own services (a stale UI would hold port 7860)."""
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "ForEach-Object { $_.ProcessId.ToString() + ' ' + $_.CommandLine }"
    )
    out = run(["powershell", "-NoProfile", "-Command", ps]).stdout
    for line in out.splitlines():
        pid, _, cmd = line.partition(" ")
        if pid.isdigit() and int(pid) != os.getpid() and any(s in cmd for s in STALE):
            run(["taskkill", "/F", "/T", "/PID", pid])
            say(f"stopped stale process {pid}: {cmd.strip()[-60:]}")


def unload_llm() -> None:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    try:
        httpx.post(f"{url}/api/generate", json={"model": "qwen3:8b", "keep_alive": 0}, timeout=30)
    except httpx.HTTPError as e:
        say(f"could not unload qwen3:8b: {e}")


def fail_running_jobs(error: str) -> list[str]:
    import psycopg

    dsn = os.getenv("DATABASE_URL", "postgresql://app:app@localhost:5432/app")
    with psycopg.connect(dsn, autocommit=True) as c:
        rows = c.execute(
            "UPDATE jobs SET status = 'failed', error = %s, locked_by = NULL,"
            " updated_at = clock_timestamp() WHERE status = 'running' RETURNING job_id",
            (error,),
        ).fetchall()
    return [str(r[0]) for r in rows]


# ---------------------------------------------------------------- commands


class Services:
    def __init__(self) -> None:
        self.procs: dict[str, subprocess.Popen[bytes]] = {}
        self.logs: dict[str, IO[bytes]] = {}

    def start(self, name: str) -> None:
        LOGS.mkdir(parents=True, exist_ok=True)
        log = self.logs.get(name) or (LOGS / f"{name}.log").open("ab")
        self.logs[name] = log
        self.procs[name] = subprocess.Popen(CHILDREN[name], cwd=ROOT, stdout=log, stderr=log)
        say(f"{name} started (pid {self.procs[name].pid}, log results/showcase/{name}.log)")

    def kill(self, name: str) -> None:
        p = self.procs.get(name)
        if p is not None and p.poll() is None:
            run(["taskkill", "/F", "/T", "/PID", str(p.pid)])
            p.wait(timeout=30)

    def stop_all(self) -> None:
        for name in self.procs:
            self.kill(name)
        for f in self.logs.values():
            f.close()


def cmd_start(a: argparse.Namespace) -> int:
    say(f"free RAM at start: {avail_mb() / 1024:.1f} GB")
    ensure_ollama()
    ensure_docker()
    r = run([PY, "-m", "app.db.migrate"])
    if r.returncode != 0:
        raise SystemExit(f"FAILED: migrate\n{r.stdout}\n{r.stderr}")
    say("database migrated")
    stop_stale()
    svc = Services()
    try:
        for name in CHILDREN:
            svc.start(name)
        wait_for("API /healthz (db, grobid, ollama)", lambda: ok(f"{API}/healthz"), 120)
        wait_for("UI", lambda: ok(UI), 120)
        time.sleep(3)  # let the worker finish importing before judging RAM
        claims_ok, msg = ram_verdict(avail_mb() / 1024)
        say(("OK   " if claims_ok else "WARN ") + msg)
        say(f"READY -> {UI}   (Ctrl+C here stops everything)")
        if not a.no_browser:
            webbrowser.open(UI)
        supervise(svc)
    except KeyboardInterrupt:
        say("stopping ...")
    finally:
        svc.stop_all()
        if not a.keep_docker:
            run(["docker", "compose", "stop"])
            say("services and Docker containers stopped")
    return 0


def supervise(svc: Services) -> None:
    guard = MemoryGuard()
    while True:
        time.sleep(1)
        if guard.update(avail_mb()):
            guard.reset()
            say(f"RAM guard: free RAM under {GUARD_FLOOR_MB} MB -> stopping the worker")
            svc.kill("worker")
            unload_llm()  # Ollama keeps Qwen in RAM after the worker dies
            jobs = fail_running_jobs(OOM_ERROR)
            say(f"failed job(s) {jobs or '-'} with an out-of-RAM message; restarting worker")
            svc.start("worker")
            continue
        for name, p in svc.procs.items():
            if p.poll() is not None:
                say(f"{name} exited with {p.returncode}; restarting (see its log)")
                svc.start(name)
                break


def cmd_check(a: argparse.Namespace) -> int:
    from app.core.config import load_config
    from app.ingest.grobid import GrobidClient, ingest_pdf
    from app.route.router import tau_for
    from app.route.tokens import estimate_tokens

    if not ok("http://localhost:8070/api/isalive"):
        ensure_docker()
    cfg = load_config(ROOT / "configs")
    t0 = time.time()
    with httpx.Client(timeout=300) as http:
        doc = ingest_pdf(Path(a.file), GrobidClient("http://localhost:8070", client=http))
    tokens = estimate_tokens(doc.full_text, cfg.route.token_margin)
    tau = tau_for(cfg.active_profile, cfg.route.tau_fraction)
    v = pdf_verdict(tokens, tau)
    print(f"title      {doc.title!r}")
    print(
        f"parsed     {time.time() - t0:.1f} s, {len(doc.sections)} sections, "
        f"{len(doc.sentences)} sentences, {len(doc.full_text.split())} words"
    )
    print(f"structure  {'ok' if doc.structure_ok else 'NOT found (flat text)'}")
    print(("OK   " if v.ok else "WARN ") + v.message)
    return 0 if v.ok else 1


def cmd_rehearse(a: argparse.Namespace) -> int:
    from app.ui.view import bundle_from_api, submit_job

    if not ok(f"{API}/healthz"):
        raise SystemExit("FAILED: services not running. Run `make showcase` in another terminal.")
    path = Path(a.file)
    low = start_mb = avail_mb()
    t0 = time.time()
    stages: dict[str, float] = {}
    low_by_stage: dict[str, float] = {}
    with httpx.Client(base_url=API, timeout=30) as c:
        jid = submit_job(c, str(path), verify=not a.no_claims)
        say(f"job {jid} submitted ({'claims off' if a.no_claims else 'full pipeline'})")
        while True:
            j = c.get(f"/v1/jobs/{jid}").json()
            now = avail_mb()
            low = min(low, now)
            stage = j.get("stage") or "-"
            low_by_stage[stage] = min(low_by_stage.get(stage, now), now)
            if j.get("stage") and j["stage"] not in stages:
                stages[j["stage"]] = round(time.time() - t0, 1)
                say(f"stage {j['stage']} at {stages[j['stage']]:.0f} s")
            if j["status"] in ("done", "failed"):
                break
            time.sleep(1)
        took = time.time() - t0
        row: dict[str, Any] = {
            "file": path.name,
            "job_id": jid,
            "status": j["status"],
            "error": j.get("error"),
            "seconds": round(took, 1),
            "stages_at_s": stages,
            "free_gb_start": round(start_mb / 1024, 2),
            "free_gb_low": round(low / 1024, 2),
            "free_gb_low_by_stage": {k: round(v / 1024, 2) for k, v in low_by_stage.items()},
            "claims": not a.no_claims,
        }
        if j["status"] == "done":
            b = bundle_from_api(c, jid)
            res = b.result
            row |= {
                "title": b.title,
                "route": res["route"],
                "tokens": res["tokens"],
                "key_points": len(res["key_points"]),
                "claims_n": len(res["claims"]),
                "metrics": res.get("metrics"),
            }
            if not a.no_claims:  # a claims-off bundle would shadow a full one of the same title
                name = a.name or re.sub(r"[^a-z0-9]+", "_", b.title.lower()).strip("_")[:40]
                out = ROOT / "demo" / f"{name or jid}.json"
                out.write_text(b.model_dump_json(), encoding="utf-8")
                row["demo_bundle"] = str(out.relative_to(ROOT))
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "rehearse.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M:%S"), **row}) + "\n")
    print(json.dumps(row, indent=2))
    if j["status"] != "done":
        return 1
    saved = row.get("demo_bundle")
    say(
        f"done in {took:.0f} s"
        + (f"; saved as demo example {saved} (restart `make showcase` to list it)" if saved else "")
    )
    return 0


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--keep-docker", action="store_true", help="leave containers up on exit")
    c = sub.add_parser("check")
    c.add_argument("file")
    r = sub.add_parser("rehearse")
    r.add_argument("file")
    r.add_argument("--name", help="demo example file name (default: from the title)")
    r.add_argument("--no-claims", action="store_true", help="skip claim checking")
    a = ap.parse_args()
    return {"start": cmd_start, "check": cmd_check, "rehearse": cmd_rehearse}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
