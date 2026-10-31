# Demo day checklist (Mon 2026-11-02)

Everything runs on the laptop. No internet is needed during the demo.
Commands run in PowerShell in `C:\NLP_Project` (make is on PATH, see CLAUDE.md §0).

## The day before: rehearse YOUR PDF (once, ~10 min)

1. Pick a **text-based** paper (not a scan), ideally ≤ ~12 pages. Check it:
   ```powershell
   make showcase-check FILE="C:\path\to\my_paper.pdf"
   ```
   `OK Route A` → good. `WARN Route B` → longer paper: a first run takes ~20–25 min
   (measured: 55-page PEGASUS 20.7 min from cold, `docs/LONG_PDF_PERFORMANCE.md`). Rehearse it ahead; the cache makes the
   live repeat fast.
2. Start everything (terminal 1, leave it open):
   ```powershell
   make showcase
   ```
3. Run your paper through the full pipeline (terminal 2):
   ```powershell
   make rehearse FILE="C:\path\to\my_paper.pdf"
   ```
   It prints time, route, key points, claims and faithfulness, and saves the result as a
   **demo example** (`demo/<title>.json`) — your paper then also opens instantly from the
   *Demo examples* tab, which is the fallback if anything goes wrong live.
4. Why this matters: every LLM call is cached. The live upload of the **same file** then skips
   Qwen entirely and only parses + verifies: measured **7.2 min first run → 41 s repeat**
   (FactCC, 8k tokens, claims on), same scores. Results go to `results/showcase/rehearse.jsonl`.
5. Ctrl+C in terminal 1 stops everything (services + Docker containers).

## 30 minutes before

1. **Restart the laptop** (frees ~5 GB that Windows/drivers hold after a long session).
   Plug in; power settings: sleep = never, lid close = do nothing.
2. Open only what you need: one browser window. Close VS Code, Claude Code, games, Spotify,
   WhatsApp, Creative Cloud.
3. Start Docker Desktop (or let the launcher do it — it starts Docker and Ollama itself).
4. Terminal: `make showcase`. It prints each step and ends with
   `READY -> http://127.0.0.1:7860` plus a RAM verdict:
   - `OK  … leave 'Check claims' ticked` (≥ 3 GB free) → full pipeline.
   - `WARN … untick 'Check claims'` → still works; claims are shown on the demo examples.
   Measured on 2026-10-31 even with VS Code + Claude Code open: a new paper with claims on
   finished (low point 0.67 GB free).
5. The browser opens on the page. Leave the terminal open (it supervises everything).

## During the demo

1. **Demo examples first** (instant, can't fail): pick *Sentence-BERT* → Open.
   - Header: route, model, faithfulness for both verifiers.
   - Click a **key point** → the paper scrolls to its source sentence (offsets, SC2).
   - Click a **claim with 🔴🟢** → DeBERTa says contradicted, MiniCheck says supported:
     the verifiers disagree, which is exactly what E5 measures.
2. **Live upload of your paper**: *Upload a PDF* tab → choose the rehearsed file → Summarize.
   Status shows the stage (ingest → summarize → keypoints → verify); ~1 min when rehearsed.
   A paper the system has never seen: ~5–10 min (Route A) — start it, talk, come back.
3. **Results**: open `results/report/tables.md` and the two figures in `results/report/`.

## Talking points

- Two routes: whole paper in one prompt (A) vs section-aware map-reduce with a merge tree (B).
- Every claim is checked locally by two verifiers; thresholds tuned on AggreFact dev, never test.
- Everything is a span over one stored text, so every highlight is exact.
- Honest limits: pilot scale (5 papers, 1 seed), Qwen only, LLM-labelled attribution reference.

## If something breaks

| Symptom | Fix |
| --- | --- |
| Launcher prints `FAILED: …` | It says what: Docker not starting → open Docker Desktop by hand, rerun; no `qwen3:8b` → `ollama pull qwen3:8b` (internet) |
| "Job failed: Stopped: the laptop ran out of RAM" | The RAM guard stopped it (laptop stays usable, worker restarts by itself). Untick "Check claims" and upload again |
| Upload stays at "ingest" | GROBID busy/not up — wait 30 s; check terminal 1 |
| Job fails mentioning ollama | Ollama crashed — rerun `make showcase` (it starts Ollama) |
| Page won't load | Terminal 1 closed? Rerun `make showcase` |
| Anything else | Demo examples tab: your rehearsed paper is there too, needs nothing live |

Logs: `results/showcase/{api,worker,ui}.log`.

## Manual fallback (if the launcher itself won't run)

```powershell
make up; make migrate
make api        # terminal 1
make worker     # terminal 2
make demo       # terminal 3 -> http://127.0.0.1:7860
```

## Short demo PDF

`tests\fixtures\pdfs\sbert.pdf` (10 pages, cached → ~35 s with claims on).
