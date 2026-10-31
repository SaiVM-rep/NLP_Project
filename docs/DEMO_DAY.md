# Demo day checklist (Mon 2026-11-02)

Everything runs on the laptop. No internet is needed during the demo.

## 30 minutes before

1. **Plug in the laptop.** Windows power settings: sleep = never, lid close = do nothing.
2. **Free RAM** (aim for ≥ 8 GB available — Task Manager → Performance → Memory):
   quit games/launchers, Spotify, WhatsApp, Creative Cloud, extra browser windows,
   **VS Code and Claude Code** (together ~3 GB). Measured in the 2026-10-31 rehearsal:
   Docker VM 4 GB (capped by `C:\Users\saivi\.wslconfig`), UI 0.8, API 0.7, worker ~3 GB
   before the verifiers, MiniCheck +2.3 GB. With VS Code + Claude Code open, the full
   pipeline did **not** fit (RAM watchdog stopped the worker); with "Check claims" off it
   did (low point 1.5 GB free).
3. **Start Ollama** (tray icon, or `ollama serve`). Check: `ollama list` shows `qwen3:8b`.
4. **Start Docker Desktop**, wait for *Engine running*. Then, in `C:\NLP_Project`:
   ```powershell
   make up          # GROBID + Postgres
   make migrate     # safe to repeat
   ```
5. **Start the services**, each in its own terminal:
   ```powershell
   make api         # terminal 1 — http://127.0.0.1:8000/healthz must say "ok" for db, grobid, ollama
   make worker      # terminal 2
   make demo        # terminal 3 — http://127.0.0.1:7860
   ```
6. **Warm up once:** upload the short demo PDF (below) and let it finish, so models are
   cached and nothing surprises you live.

## During the demo

1. **Demo examples first** (instant, can't fail): pick *Sentence-BERT* → Open.
   - Read the header: route, model, faithfulness for both verifiers.
   - Click a **key point** → the paper scrolls to its source sentence (offsets, SC2).
   - Click a **claim with 🔴🟢** → DeBERTa says contradicted, MiniCheck says supported:
     the verifiers disagree, which is exactly what E5 measures.
2. **Live upload** (~7 min for ≤ 10 pages): *Upload a PDF* tab → choose file → Summarize.
   Start it early and come back to it, or show a finished one.
   **Under ~8 GB free: untick "Check claims"** → summary + key points only (no verifier
   load). Claim checking is then shown on the demo examples. `sbert.pdf` is cached: ~20 s.
3. **Results**: open `results/report/tables.md` and the two figures in `results/report/`.

## Talking points

- Two routes: whole paper in one prompt (A) vs section-aware map-reduce with a merge tree (B).
- Every claim is checked locally by two verifiers; thresholds tuned on AggreFact dev, never test.
- Everything is a span over one stored text, so every highlight is exact.
- Honest limits: pilot scale (5 papers, 1 seed), Qwen only, LLM-labelled attribution reference.

## If something breaks

| Symptom | Fix |
| --- | --- |
| Upload tab: "API not reachable" | `make api` not running (terminal 1) |
| Upload stays at "ingest" | Docker/GROBID not up: `make up`, wait 30 s |
| Job fails mentioning ollama | Ollama not running, or RAM too low — close apps, retry |
| Upload hangs at "verify" / laptop crawls | RAM exhausted — untick "Check claims" and retry |
| Page won't load | `make demo` not running (terminal 3) |
| Anything else | Fall back to the demo examples — they need nothing but `make demo` |

## Short demo PDF

`tests\fixtures\pdfs\sbert.pdf` (10 pages, cached → fast) — or any text-based paper ≤ 10 pages.
