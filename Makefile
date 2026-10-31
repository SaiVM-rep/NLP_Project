# PRD §14. Targets: up down migrate test lint fmt demo demo-export exp report
#
# Windows note: these call .venv/Scripts/python.exe directly so they work from
# Git Bash without activating the venv. On Linux/macOS set PY=.venv/bin/python.

PY ?= .venv/Scripts/python.exe

.PHONY: help up down logs migrate test test-all smoke live-b live-kp api worker summarize doctor models lint fmt typecheck hooks check demo showcase showcase-check rehearse demo-export exp report clean

help:
	@echo "up        start db + grobid"
	@echo "live-kp   P5 live key points on stored TEI (Qwen3 + MiniLM)"
	@echo "models    download pinned model weights (verification itself is offline, FR-5.6)"
	@echo "down      stop services"
	@echo "migrate   apply db migrations"
	@echo "api       HTTP API on :8000 (P6)"
	@echo "worker    job worker: polls the queue, runs the pipeline (P6)"
	@echo "demo      P7 UI on :7860 (demo examples + PDF upload)"
	@echo "showcase  live showcase: start + supervise everything (Ctrl+C stops); see docs/DEMO_DAY.md"
	@echo "test      unit tests only (no network, no GPU)"
	@echo "test-all  unit + integration (needs 'make up')"
	@echo "check     lint + format check + types + unit tests"
	@echo "exp E=E1  run one experiment"
	@echo "report    regenerate every table and figure from results/"

up:
	docker compose up -d db grobid

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

migrate:
	$(PY) -m app.db.migrate

# P6 service: run both (after `make up` + `make migrate`), each in its own terminal
api:
	$(PY) -m uvicorn app.api.main:app --port 8000

worker:
	$(PY) -m app.worker $(ARGS)

# Default suite: no network, no GPU, no docker (NFR-8, and what CI runs).
test:
	$(PY) -m pytest tests/unit

test-all:
	$(PY) -m pytest

# P0 exit check. Needs a real API key (.env) and a pulled ollama model.
smoke:
	$(PY) scripts/smoke_p0.py $(ARGS)

# P4 exit: live Route B on stored TEI + per-level verification. Needs >= 8 GB free RAM.
live-b:
	$(PY) scripts/live_route_b.py $(ARGS)

# P5: key points on one stored-TEI paper, real Qwen3 + MiniLM
live-kp:
	$(PY) scripts/live_keypoints.py $(ARGS)

# P2 exit. make summarize FILE=paper.pdf [ARGS="--profile qwen3_8b_local"]
summarize:
	@test -n "$(FILE)" || (echo "usage: make summarize FILE=paper.pdf" && exit 1)
	$(PY) -m app.cli summarize "$(FILE)" $(ARGS)

# Is this machine set up? Read-only, safe to run any time.
doctor:
	$(PY) scripts/doctor.py

models:
	$(PY) -m app.verify.models --fetch

lint:
	$(PY) -m ruff check .

fmt:
	$(PY) -m black .
	$(PY) -m ruff check . --fix

typecheck:
	$(PY) -m mypy

hooks:
	$(PY) -m pre_commit install

check: lint typecheck
	$(PY) -m black --check .
	$(PY) -m pytest tests/unit

# P7 UI on http://127.0.0.1:7860 (localhost only). Demo examples load from demo/*.json;
# upload needs `make api` + `make worker`. Save a finished job: make demo-export JOB=<id>
demo:
	$(PY) -m app.ui $(ARGS)

# Live showcase, one terminal: Ollama + Docker + migrate + api + worker + UI, supervised.
# make showcase-check FILE=x.pdf (size/route) · make rehearse FILE=x.pdf (full run -> demo/)
showcase:
	$(PY) scripts/showcase.py start $(ARGS)

showcase-check:
	@test -n "$(FILE)" || (echo "usage: make showcase-check FILE=paper.pdf" && exit 1)
	$(PY) scripts/showcase.py check "$(FILE)"

rehearse:
	@test -n "$(FILE)" || (echo "usage: make rehearse FILE=paper.pdf" && exit 1)
	$(PY) scripts/showcase.py rehearse "$(FILE)" $(ARGS)

demo-export:
	$(PY) scripts/export_demo.py $(JOB) $(ARGS)

# Usage: make exp E=E1
exp:
	@test -n "$(E)" || (echo "usage: make exp E=E1" && exit 1)
	$(PY) -m eval.run --experiment $(E) $(ARGS)

report:
	$(PY) -m eval.report

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
