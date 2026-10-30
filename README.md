# Semantic Text Compression & Key-Point Extraction

**Transformer-Driven Framework for Semantic Text Compression and Key-Point Extraction from Long-Form Documents**
Team 9 — Sai Vishod Muralidaran, Raahul S Nayar, Rudhayesh Murali, Samay Chetan · Guide: Dr./Prof. Priyanka S Shivaramaiah

> **In one sentence:** you give it a long document (a research-paper PDF or a web article),
> and it gives back a short summary, the key points, and a check of whether every claim in
> that summary is actually backed by the original text, each linked to the exact sentence.

---

## Contents

1. [What works today](#1-what-works-today)
2. [Quick start](#2-quick-start)
3. [How it works, step by step](#3-how-it-works-step-by-step)
4. [The models: Qwen (used) and Gemini (optional)](#4-the-models-qwen-used-and-gemini-optional)
5. [Key ideas and terminology](#5-key-ideas-and-terminology)
6. [Project layout](#6-project-layout)
7. [Configuration](#7-configuration)
8. [Commands](#8-commands)
9. [Testing and code quality](#9-testing-and-code-quality)
10. [Troubleshooting](#10-troubleshooting)
11. [Roadmap](#11-roadmap)
12. [Tool stack](#12-tool-stack)

---

## 1. What works today

The project is built in phases (P0 → P9, defined in `PRD.md`). Status as of **2026-10-31**
(deadline: report + live demo on **Mon 2026-11-02**):

| Phase | What it is | Status |
| --- | --- | --- |
| **P0 Foundations** | Config, LLM client, cache, rate limiter, logging, Docker services | ✅ Done |
| **P1 Ingestion** | Turn a PDF or web article into clean, structured text | ✅ Done (20 PDFs, 10 URLs) |
| **P2 Baseline** | One-shot summary from the command line | ✅ Done |
| **P3 Evaluation** | Check every summary claim against the source | ✅ Done — verifier thresholds tuned on AggreFact dev (4,349 rows) and frozen |
| **P4 Route B** | Summarize documents too long for one prompt | ✅ Done — faithfulness per merge level |
| **P5 Key points** | Ranked key points linked to their source sentence | ✅ Done — attribution precision **0.667** on 24 key points (LLM-labelled reference) |
| **P6 Service** | HTTP API + job queue + database | ✅ Done — upload over HTTP, live progress, crash-safe worker |
| **P7 UI** | Two-pane web page with highlighting | ✅ Done — `make demo`, 3 precomputed papers, PDF upload |
| **P8 Experiments** | E1 (+E3), E5, SC4 — **pilot scope** (5 papers, 1 seed) | 🟡 Running unattended overnight (`scripts/overnight.py`) |
| P9 Report | Chapter 5 generated from logs (`make report`) | 🟡 Generator built; waits for P8 |

**Scope cut for the deadline:** E2, E4, the 50-document gold set and human annotation are
future work. **Qwen3-8B (local) is the only model used**; Gemini code is kept but unused
(`CLAUDE.md` DECISION-27). Both need the guide's OK.

**So right now you can:** open the web demo, pick one of three research papers, read its
summary, key points and every checked claim, and click any of them to jump to the exact
sentence in the paper that backs it. You can also feed in your own PDF (see §2) — everything
runs on your own laptop, nothing leaves the machine.

### Results so far

| What | Number | Source |
| --- | --- | --- |
| Verifier balanced accuracy, AggreFact dev | DeBERTa **0.675** (θ 0.10) · MiniCheck **0.766** (θ 0.5) | `results/tuning/aggrefact_dev.summary.json` |
| Demo papers, faithfulness (DeBERTa / MiniCheck) | SBERT 0.60 / 1.00 · FRANK 0.61 / 0.92 · TRUE 0.64 / 0.94 | `demo/*.json` |
| Key-point attribution precision@1 | **0.667** (16/24) | `results/attribution/summary.json` |
| Overlap metrics vs human ratings (SummEval, Kendall τ) | ROUGE-1: relevance 0.22, faithfulness 0.14 | `CLAUDE.md` §7 |
| E1 / E3 / E5 / SC4 | pending (overnight run) | `make report` → `results/report/tables.md` |

---

## 2. Quick start

### Open the demo (no GPU, no Docker, no internet)

```powershell
make demo          # then open http://127.0.0.1:7860
```

Pick an example → **Open**. Click any key point or claim: the paper on the left scrolls to
the sentence that backs it and highlights it. Each claim shows two icons, **DeBERTa first,
MiniCheck second**: 🟢 supported · 🟡 unsupported · 🔴 contradicted.

The examples are files in `demo/`. Build more (real pipeline, needs Ollama + GPU):
`.venv\Scripts\python.exe scripts\make_demo.py <name>` for a paper in `tests/fixtures/tei/`,
or save any finished API job with `make demo-export JOB=<job_id>`.

### Upload your own PDF in the web page

The **Upload a PDF** tab needs the full pipeline running behind it:

1. Free RAM: aim for **≥ 8 GB available** (quit games, launchers, extra browser tabs).
2. Start **Docker Desktop**, then `make up` (GROBID + Postgres) and `make migrate` (once).
3. `make api` in one terminal, `make worker` in another. **Ollama** must be running.
4. `make demo`, open the **Upload a PDF** tab, choose the file, **Summarize**.

Works with **text-based** PDFs under 25 MB (scanned images have no text → not supported).
A ≤ 10-page paper takes **~7 minutes**; papers over ~13k tokens go through Route B and can
take 30+ minutes.

### One-time setup

You need: **Windows**, **Docker Desktop**, **Ollama**, and **uv** (a fast Python package manager).

```powershell
cd C:\NLP_Project

# 1. Install Python 3.11 + all dependencies into .venv (torch comes with CUDA 12.8)
uv sync --group dev --extra ingest --extra ml --extra eval --extra service --extra ui

#    Download the pinned verifier/metric models once (~6 GB). Verification itself never
#    goes online after this.
make models

# 2. Secrets: copy the template. Optional keys: HF_TOKEN (gated AggreFact dataset),
#    GEMINI_API_KEY (only for a gemini_* profile; the default is local Qwen)
copy .env.example .env

# 3. Download the local model (~5.2 GB, once)
ollama pull qwen3:8b

# 4. Check everything is ready
.venv\Scripts\python.exe scripts\doctor.py
```

### Every session

1. Open **Docker Desktop** and wait until it says *running*.
2. Start the PDF parser:
   ```powershell
   docker compose up -d grobid
   ```
   Wait ~30 seconds for it to load.

### Use it over HTTP (P6)

```powershell
make up; make migrate          # Postgres + GROBID, then tables
make api                       # terminal 1: http://localhost:8000/docs
make worker                    # terminal 2: processes jobs one at a time
curl -F "file=@paper.pdf;type=application/pdf" -F 'options={"profile":"qwen3_8b_local"}' localhost:8000/v1/jobs
curl -N localhost:8000/v1/jobs/<job_id>/events    # live progress (SSE)
curl localhost:8000/v1/jobs/<job_id>/result        # summary, key points, claims, metrics
```

`options` (all optional): `profile`, `tau`, `keypoints` (true), `verify` (true). A URL job is
`curl -H "content-type: application/json" -d '{"url": "https://..."}' localhost:8000/v1/jobs`.
If the worker dies mid-job, the job is picked up again after the lease (2 min) and fails
cleanly after 3 tries.

### Summarize

**Default: Qwen (local, offline: the document never leaves your computer):**
```powershell
.venv\Scripts\python.exe -m app.cli summarize "C:\path\to\paper.pdf"
```

**Optional: Gemini (cloud, sends the text to Google, needs `GEMINI_API_KEY`):**
```powershell
.venv\Scripts\python.exe -m app.cli summarize "C:\path\to\paper.pdf" --profile gemini_full
```

> **Tip:** in File Explorer, **Shift + right-click** the PDF → **Copy as path**. It copies the
> path with quotes already around it, which you need if the path has spaces.

**What you get:**
- The summary printed in the terminal (~1 minute for a 10–15 page paper on Qwen).
- A JSON file in `results\summaries\<doc_id>.json` containing the summary, which route was used,
  the token count, the prompt hash, and the full parsed document (sections, paragraphs, sentences).
- Running the same PDF again is **instant**: the answer comes from the cache.

**Optional flags:**

| Flag | What it does |
| --- | --- |
| `--profile gemini_full` | Use Gemini instead of the local Qwen default |
| `--profile qwen_sim_8k` (or `_4k`, `_12k`) | Pretend Qwen has a smaller window (forces Route B, for experiments) |
| `--tau 5000` | Override the Route A/B threshold (see §5) |
| `--out some\folder` | Save the JSON somewhere else |

---

## 3. How it works, step by step

```
  your PDF
     │
     ▼
┌──────────────┐   PDF → TEI/XML (a structured format for documents)
│  1. GROBID   │   finds title, abstract, section headings, paragraphs, references
└──────┬───────┘
       ▼
┌──────────────┐   XML → our Document object
│ 2. TEI parser│   keeps sections + paragraphs, puts references aside
└──────┬───────┘
       ▼
┌──────────────┐   glues everything into ONE text string (full_text)
│ 3. Span      │   and records the exact character position of every
│    builder   │   section, paragraph and sentence (pySBD splits sentences)
└──────┬───────┘
       ▼
┌──────────────┐   "how long is this?"  tokens ≤ τ  → Route A (one prompt)
│ 4. Router    │                         tokens > τ  → Route B (map-reduce, P4)
└──────┬───────┘
       ▼
┌──────────────┐   fills the frozen prompt template, asks the LLM for JSON,
│ 5. Single-   │   validates it, retries once with a "repair" nudge if the JSON is broken
│    pass      │
└──────┬───────┘
       ▼
┌──────────────┐   cache → daily budget check → rate limiter → provider call
│ 6. LLM client│   → retry with backoff → save to cache → log the call
└──────┬───────┘
       ▼
   summary + JSON file
```

### Step 1: GROBID (PDF parsing)

A PDF is really just "draw these letters at these coordinates". It has no idea what a heading
or a paragraph is. **GROBID** is a machine-learning tool built specifically for scientific
papers. It reads the layout and outputs **TEI XML**, a standard format that marks up `<title>`,
`<abstract>`, `<head>` (section headings), `<p>` (paragraphs) and `<listBibl>` (references).

It runs as a **Docker container** (a self-contained mini-server) on `http://localhost:8070`.
We use the lightweight image `lfoppiano/grobid:0.8.1` (CRF models, ~500 MB) instead of the
full deep-learning image (~6 GB). The section/paragraph model is the same in both. See
DECISION-10 in `CLAUDE.md`.

Code: `src/app/ingest/grobid.py`

### Step 2: TEI parser

Reads GROBID's XML with **lxml** and pulls out:
- **Title**, falling back to the file name if GROBID can't find one.
- **Abstract**, which becomes the first section.
- **Body sections**, with their heading and level (`2.1` → level 2).
- **Paragraphs**. Formulas, figures and tables are skipped.
- **References**, stored separately in `references_text` and **never summarized** (FR-1.6).

If fewer than two sections with headings are found, the document is marked
**`structure_ok = false`** and a warning is logged. It is still summarized, but it is excluded
from experiments that depend on structure (FR-1.5).

Code: `src/app/ingest/tei.py`

### Step 3: Span builder (the backbone)

All the text is stored **once**, in a single string called `full_text`. Everything else is a
**span**: a pair of character positions `(start, end)` pointing into that string.

```
full_text = "Introduction\n\nTransformers use attention. They are fast.\n\n..."
                             ↑ start=14                  ↑ end=40
sentence span (14, 40) → "Transformers use attention."
```

Why bother? Because later every key point and every verified claim must point back to the
**exact sentence** in the source it came from, so the UI can highlight it. The project rule
(**SC2**) is: `full_text[start:end]` must reproduce the stored text, **always**, 100%.

Sentences are split with **pySBD**, which understands that "Dr. Smith et al. agree." and
"Fig. 2" are not sentence endings.

Code: `src/app/ingest/spans.py` · Data model: `src/app/core/schemas.py`

### Step 4: Router

Counts **tokens**, the word-pieces an LLM reads (roughly 1 token ≈ 4 characters of English),
and compares the count to a threshold **τ (tau)**:

- **Route A:** the document fits in one prompt, so summarize it in one go.
- **Route B:** it doesn't fit, so split it into chunks, summarize each, then merge the summaries
  (**map-reduce**). Chunks follow section boundaries; chunk summaries are merged in groups
  of 4 until one remains, and every node of that merge tree is kept.

`τ = 80% × the model's effective context`. The remaining 20% is left for the model's answer.

Code: `src/app/route/router.py`, `src/app/route/tokens.py`

### Step 5: Single-pass summarizer (Route A)

Fills the prompt template `prompts/single_pass.v1.txt` with the document and a target length
(10% of the input by default), and asks the model to answer as **JSON**:
`{"summary": "..."}`.

- If the reply isn't valid JSON, it retries **once** with a "repair" instruction.
- If it still fails, it stops with an error rather than guessing.
- Markdown code fences (```` ```json ````) around the reply are tolerated.

Prompts are **frozen**. They are never edited in place, because changing a prompt changes
results. A change means a new file (`single_pass.v2.txt`). Each prompt's **SHA-256 hash** is
saved with every result, so you can always tell which prompt produced which summary.

Code: `src/app/summarize/single_pass.py`

### Step 6: LLM client

One interface for every model. In order, each call goes through:

1. **Cache.** Has this exact (provider, model, prompt, settings, seed) been asked before? If
   so, return the saved answer instantly at zero cost. This makes reruns free and
   reproducible.
2. **Budget guard.** Stop before the daily free-tier request cap is hit.
3. **Rate limiter.** An exact **sliding window** per minute for requests and for tokens
   (a token bucket was replaced: it admitted up to 2x the limit in a minute, DECISION-19).
4. **Provider call.** Gemini (REST API over `httpx`) or Ollama (local REST API).
5. **Retry with backoff.** On a temporary error (HTTP 429 / 5xx / timeout), wait (exponential
   backoff with jitter) and try again.
6. **Fallback policy.** `fail` (default: never silently switch models mid-experiment) or
   `ollama` (fall back to local, for demos only).
7. **Call log.** Records tokens, latency, retries and whether it was a cache hit. The prompt
   itself is never logged, only its hash, so document text never ends up in logs.

Code: `src/app/llm/`

---

## 4. The models: Qwen (used) and Gemini (optional)

**Qwen3-8B is the default and the only model used in the results** (DECISION-27): no API key,
no internet, no daily quota. Gemini stays available as an option.

| | **Gemini** (`gemini-3.6-flash`, optional) | **Qwen3-8B** (`qwen3:8b` via Ollama, default) |
| --- | --- | --- |
| Where it runs | Google's servers (cloud API) | Your GPU (RTX 4070 Laptop, 8 GB) |
| Privacy | Text is sent to Google | Nothing leaves your machine |
| Cost | Free tier, with daily/minute limits | Free, unlimited |
| Context window | ~1,048,576 tokens | **16,384** tokens on this GPU |
| Longest doc for Route A | Effectively any paper | ~13,000 tokens (~25–30 pages) |
| Speed | A few seconds | ~1 minute per paper |
| Needs | `GEMINI_API_KEY` in `.env` | `ollama pull qwen3:8b` |

**Why only 16,384 for Qwen?** The model weights take ~5.2 GB. The **KV cache** (the model's
working memory for the prompt) grows with context length, and at 32k tokens it needs ~4.5 GB
more, which doesn't fit in 8 GB. At 16k the measured peak is ~6.2 GB.

**Sequential model residency.** Because the GPU is small, models are loaded one at a time and
unloaded after use (`keep_alive=0`), so the verifier models (DeBERTa, MiniCheck) get the GPU
after Qwen is done.

**Simulated profiles.** `qwen_sim_4k/8k/12k` use the same Qwen model but *pretend* it has a
smaller window, so experiments can force Route B. (`sim_8k/16k/32k` are the Gemini versions.)

**One request at a time on Ollama.** Local profiles set `max_concurrency: 1`: Ollama serves one
call at a time on this GPU, so sending 4 at once only queued them past the 600 s timeout
(this killed the first overnight E1 run; DECISION-30).

---

## 5. Key ideas and terminology

| Term | Plain meaning |
| --- | --- |
| **LLM** | Large Language Model: the AI that writes the summary (Gemini, Qwen) |
| **Abstractive summary** | Written in the model's own words, not copy-pasted sentences (that would be *extractive*) |
| **Token** | A word-piece the model reads; ~4 characters of English |
| **Context window** | The maximum number of tokens a model can read at once |
| **Effective context** | The budget *we* choose to allow (can be smaller than the real window) |
| **τ (tau)** | The token threshold that decides Route A vs Route B |
| **Route A / single-pass** | Whole document in one prompt |
| **Route B / map-reduce** | Chunk → summarize each chunk (*map*) → merge summaries up a tree (*reduce*) |
| **Merge tree** | The record of which partial summaries were merged into which, level by level |
| **Span** | `(start, end)` character positions into `full_text` |
| **`full_text`** | The single stored copy of the document's text |
| **`structure_ok`** | Did GROBID find real sections? `false` = flat text |
| **TEI XML** | Text Encoding Initiative, the structured format GROBID outputs |
| **Key point** | A short, standalone statement of an important idea, linked to its source span |
| **Claim decomposition** | Splitting a summary sentence into atomic facts ("X uses Y", "Y is fast") |
| **NLI (entailment)** | Natural Language Inference: does text A *support*, *contradict*, or say *nothing about* claim B? |
| **Faithfulness** | % of summary claims actually supported by the source (the anti-hallucination metric) |
| **Hallucination** | The model stating something that isn't in the source |
| **ROUGE / BERTScore** | Standard metrics comparing a summary against a reference summary |
| **Profile** | A named model + context budget + rate limits, in `configs/models.yaml` |
| **Cache hit** | Answer served from disk; no API call made |
| **Seed** | Fixed random number so the same input gives the same output |
| **Run manifest** | A record of the git commit, config, prompt hashes and seeds behind a result |

---

## 6. Project layout

```
NLP_Project/
├── README.md                 ← you are here
├── PRD.md                    ← the full spec (requirements, experiments, design)
├── CLAUDE.md                 ← build ledger: what's done, verified, and decided
├── configs/
│   ├── default.yaml          ← run settings (tau, chunk size, temperature, seed, …)
│   └── models.yaml           ← model profiles (qwen3_8b_local default, qwen_sim_*, gemini_*)
├── prompts/
│   ├── single_pass.v1.txt    ← Route A prompt (frozen)
│   ├── chunk_summary.v1.txt  ← Route B: one chunk
│   ├── merge.v1.txt          ← Route B: merge partial summaries
│   └── claim_decompose.v2.txt← verification: summary sentence → atomic claims
├── src/app/
│   ├── cli.py                ← `python -m app.cli summarize …`
│   ├── core/                 ← config, schemas (Document/Span/Section), logging, run manifest
│   ├── ingest/               ← grobid.py, tei.py, spans.py
│   ├── route/                ← tokens.py, router.py, chunker.py (section-aware + uniform B2)
│   ├── summarize/            ← single_pass.py (Route A), map_reduce.py (Route B), common.py
│   ├── llm/                  ← client, cache, limiter, providers/{gemini,ollama}.py
│   ├── verify/               ← decompose, evidence, embedder, nli (DeBERTa), minicheck, pipeline
│   ├── keypoints/            ← candidates, rank (PageRank + centroid), cluster, attribute, pipeline
│   ├── api/  worker/  db/    ← P6: FastAPI + SSE, lease-based job worker, Postgres/in-memory store
│   └── ui/                   ← P7: render.py (highlighting), view.py, app.py (Gradio page)
├── scripts/
│   ├── doctor.py             ← "is this machine ready?" check
│   ├── smoke_p0.py           ← live Gemini + Ollama call check
│   ├── make_demo.py          ← build demo/*.json with the real pipeline (no Docker)
│   ├── tune_aggrefact.py     ← verifier thresholds on AggreFact dev; --split test = SC4
│   ├── attr_sheet.py / attr_score.py ← key-point attribution sheets + scoring
│   └── overnight.py          ← unattended, resumable GPU chain: demo → E1 → E5 → SC4 → report
├── eval/                     ← data, metrics, BERTScore, perturb, tune, run (E1/E5), report, attribution
│   └── gold/                 ← attribution_labels.csv (24 key points, LLM-labelled)
├── demo/                     ← precomputed demo papers for the UI (sbert, frank, true)
├── tests/unit/               ← fast tests: no network, no GPU, no Docker
├── tests/integration/        ← live GROBID / GPU / Ollama tests
├── results/                  ← summaries, tuning, E1/E5 runs, report tables + figures (not in git)
├── docker-compose.yml        ← GROBID + Postgres (+ api/worker later)
├── Dockerfile                ← API/worker image (`docker build` not yet verified)
├── Makefile                  ← shortcuts (make test, make summarize, …)
└── pyproject.toml            ← dependencies + tool settings
```

---

## 7. Configuration

- **`.env`** holds secrets and local paths. It is never committed. Template: `.env.example`.

  | Key | Purpose |
  | --- | --- |
  | `GEMINI_API_KEY` | Your Google AI Studio key |
  | `GROBID_URL` | Default `http://localhost:8070` |
  | `OLLAMA_URL` | Default `http://localhost:11434` |
  | `LLM_CACHE_DIR`, `HF_HOME` | LLM response cache and model weights, kept outside the repo |
  | `HF_HUB_DISABLE_XET`, `HF_HUB_DISABLE_SYMLINKS_WARNING` | Hugging Face download workarounds for this machine |
  | `HF_TOKEN` | Only for the gated AggreFact dataset (never commit it) |

- **`configs/default.yaml`** holds tunable settings: `profile` (`qwen3_8b_local`), `tau_fraction`
  (0.8), `compression_pct` (10), `temperature` (0.2), `seed` (1234), chunk sizes, key-point
  thresholds, `fallback` policy. **Verifier thresholds are frozen** from AggreFact dev
  (DECISION-28): DeBERTa `theta_e` 0.10, MiniCheck `theta_e` 0.5, `theta_c` 0.5.

- **`configs/models.yaml`** holds model profiles. Unknown keys are rejected (`extra="forbid"`),
  so a typo fails loudly instead of being silently ignored.

---

## 8. Commands

`make` is installed via winget. If `make` isn't found, add its folder to `PATH` (see
`CLAUDE.md` §0), or run the Python commands directly.

| Command | What it does |
| --- | --- |
| `make demo` | **The web page** on http://127.0.0.1:7860 (demo papers + PDF upload) |
| `make demo-export JOB=<id>` | Save a finished API job as a demo paper in `demo/` |
| `make summarize FILE="paper.pdf"` | Summarize with Qwen (local default) |
| `make summarize FILE="paper.pdf" ARGS="--profile gemini_full"` | Summarize with Gemini |
| `make up` / `make down` | Start / stop GROBID + Postgres |
| `make migrate` | Create / update the database tables (run once after `make up`) |
| `make api` / `make worker` | Run the HTTP API on port 8000 / the job worker — each in its own terminal |
| `make doctor` | Check Python, `.env`, Ollama model, Docker, GPU |
| `make smoke` | One live Gemini call + one live Ollama call (uses a little quota) |
| `make live-b` | Live Route B on the stored SBERT paper + faithfulness per merge level, with RAM/GPU peaks (needs free RAM, see §9; `ARGS="--min-free-gb 4"` to lower the guard) |
| `make live-kp` | Live key points on the stored SBERT paper (Qwen3 + MiniLM), each printed next to its source sentence |
| `make models` | Download the pinned model weights and write a half-size bf16 copy of MiniCheck (the only step that downloads models) |
| `python -m eval.data --check` | Fetch, parse and count every dataset (AggreFact needs `HF_TOKEN`) |
| `make exp E=E1` / `make exp E=E5 ARGS="--e1-dir results/E1/<run>"` | Pilot experiments → `results/<E>/<run_id>/` (manifest, per-doc JSONL, aggregate CSV) |
| `make report` | Chapter-5 tables + figures from the newest runs → `results/report/` |
| `python scripts/tune_aggrefact.py [--split test]` | Verifier threshold sweep on AggreFact dev / SC4 on test at frozen θ |
| `python scripts/overnight.py` | Unattended resumable chain (start detached: see the script docstring) |
| `make test` | Unit tests (no network/GPU/Docker needed) |
| `make check` | Lint (ruff) + format (black) + types (mypy) + unit tests |
| `make fmt` | Auto-format the code |
| `make hooks` | Install the git pre-commit hooks |

---

## 9. Testing and code quality

- **TDD (test-driven development)** is mandatory: write the failing test first, watch it fail,
  then write the code.
- **538 unit tests** run in about a minute with **no network, no GPU and no Docker**. Network
  calls are faked with `httpx.MockTransport`; models are tiny random-weight stand-ins.
- **Integration tests** (`pytest tests/integration`) need GROBID, the GPU and/or Ollama. One of
  them proves verification runs with the network switched off (FR-5.6).
- **Memory:** the GPU tests need ~4 GB free RAM (MiniCheck now loads a 1.5 GB bf16 copy; the original 3.1 GB file needed ~5 GB) and up to
  6.2 GB of GPU memory (Qwen3). Close Docker Desktop when not running the GROBID tests. The
  first live Route B run (`make live-b`) started with 4.4 GB free and bottomed out at 0.09 GB —
  aim for 8 GB free (close VS Code, browsers, games, background vendor apps).
- **ruff** (lint), **black** (formatting) and **mypy --strict** (type checking) all must pass:
  `make check`.
- **pre-commit** runs all of the above automatically before each git commit.

---

## 10. Troubleshooting

| You see | Why | Fix |
| --- | --- | --- |
| `GROBID unreachable … run docker compose up -d grobid` | Docker Desktop or GROBID isn't running | Start Docker Desktop, then `docker compose up -d grobid` |
| `ollama 500: token repeat limit reached` | The prompt was cut off and the model looped. **Fixed 2026-10-29** (we now send `num_ctx`) | Make sure you're on the latest code |
| Any error mentioning `ollama` / port 11434 | Ollama isn't running | `ollama serve` in another window |
| `model returned invalid JSON twice` | The model ignored the output format twice | Retry; bad replies are never cached, so a rerun asks again |
| Hugging Face download stuck at 0 bytes | The `hf_xet` backend stalls on this machine | Keep `HF_HUB_DISABLE_XET=1` in `.env` |
| `GEMINI_API_KEY is empty` | No key in `.env` | Add it, or use `--profile qwen3_8b_local` |
| `… is not a PDF` / `limit is 25 MB` | Wrong file type / too big | Use a real PDF under 25 MB |
| `warning: GROBID found no section structure` | Not a paper-style PDF | Not an error; it summarizes the flat text |
| Title shows the file name | GROBID couldn't find a title block | Cosmetic only |
| Empty or garbage output | A **scanned** PDF (images, no text layer) | Needs OCR, which isn't supported |
| `ollama transport:` (empty message) in a long Route B run | Calls queued past the 600 s timeout | Fixed: local profiles use `max_concurrency: 1`; E1 retries itself |
| Upload tab says the API is not reachable | `make api` / `make worker` / Docker not running | See §2 "Upload your own PDF" |

---

## 11. Roadmap

**Before the deadline (2026-11-02):**
- Overnight chain finishes E1 (+E3), E5, SC4 → `make report` → report chapter 5.
- Rehearse the live PDF upload (Docker + API + worker) after the GPU run is done.
- Team member spot-checks 8 key-point labels (`results/attribution/sbert.sheet.csv`) for an
  agreement number; the guide confirms the scope cut and Qwen-only setup.

**Future work (reported as limitations):** E2 (τ sweep) and E4 (section-aware vs uniform
chunks); a human-annotated 50-document gold set and claim-level human labels (Cohen's κ);
tuning `theta_attr`/`dup_distance` on that gold set; NLI reranking for key-point attribution;
`theta_c` tuning (needs 3-way labels); OCR for scanned PDFs; `docker build` for API/worker.

---

## 12. Tool stack

| Area | Tool |
| --- | --- |
| Language | Python 3.11 (managed by **uv**, in `.venv`) |
| PDF parsing | **GROBID** 0.8.1 (lightweight image, Docker) |
| XML parsing | **lxml** |
| Sentence splitting | **pySBD** |
| Web extraction | **trafilatura** |
| Cloud LLM (optional) | **Google Gemini** `gemini-3.6-flash` via REST (`httpx`) |
| Local LLM (default) | **Qwen3-8B** (Q4_K_M) via **Ollama** |
| Config / validation | **pydantic**, **PyYAML**, **python-dotenv** |
| Containers | **Docker Desktop**, **Docker Compose** |
| Database | **PostgreSQL 16** (psycopg 3), plain SQL migrations |
| Web API / UI | **FastAPI** + SSE, **Gradio** 6 |
| Verifiers | **DeBERTa-v3 NLI**, **MiniCheck** Flan-T5 (Hugging Face `transformers`, CUDA 12.8 torch) |
| Embeddings / metrics | **sentence-transformers** (MiniLM), **rouge-score**, **bert-score**, **scipy** |
| Testing / quality | **pytest**, **ruff**, **black**, **mypy**, **pre-commit** |
| Build shortcuts | **GNU Make** |
