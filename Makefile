# PRD §14. Targets: up down migrate test lint fmt demo exp report
#
# Windows note: these call .venv/Scripts/python.exe directly so they work from
# Git Bash without activating the venv. On Linux/macOS set PY=.venv/bin/python.

PY ?= .venv/Scripts/python.exe

.PHONY: help up down logs migrate test test-all smoke summarize doctor models lint fmt typecheck hooks check demo exp report clean

help:
	@echo "up        start db + grobid"
	@echo "models    download pinned model weights (verification itself is offline, FR-5.6)"
	@echo "down      stop services"
	@echo "migrate   apply db migrations"
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

# Default suite: no network, no GPU, no docker (NFR-8, and what CI runs).
test:
	$(PY) -m pytest tests/unit

test-all:
	$(PY) -m pytest

# P0 exit check. Needs a real API key (.env) and a pulled ollama model.
smoke:
	$(PY) scripts/smoke_p0.py $(ARGS)

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

demo:
	$(PY) -m ui.gradio_app

# Usage: make exp E=E1
exp:
	@test -n "$(E)" || (echo "usage: make exp E=E1" && exit 1)
	$(PY) -m eval.run --experiment $(E)

report:
	$(PY) -m eval.report

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
