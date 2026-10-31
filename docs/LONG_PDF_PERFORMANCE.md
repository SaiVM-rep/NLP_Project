# Long-PDF performance: what was slow, what changed, what it measures now

Measured on the demo laptop (RTX 4070 Laptop 8 GB, 15.5 GB RAM, Qwen3-8B Q4_K_M through
Ollama), 2026-10-31. Decisions: CLAUDE.md DECISION-35, -37, -38.

## Result

**PEGASUS (55 pages, 30,482 words, Route B), first run with an empty LLM cache, upload
settings, claim checking on: 20.7 min** (was estimated at 1.5–3 h before the changes).

| Stage | Seconds | What runs |
| --- | --- | --- |
| Summary (Route B) | 494 | 87 chunk summaries (162 s), 31 merges (331 s) |
| Key points | 253 | 79 section calls (236 s), MiniLM ranking + attribution |
| Claim checking | 494 | 37 decomposition calls (51 s), MiniLM evidence, DeBERTa, MiniCheck (~443 s) |
| **Total** | **1,241** | 234 LLM calls, 115k tokens in, 29k tokens out |

Output: 469-word summary (no compression needed), 8 key points, 99 claims;
DeBERTa 63 supported / 20 unsupported / 16 contradicted (0.64), MiniCheck 97 supported (0.98).
Free RAM low point 0.5 GB with VS Code + Claude Code open and Docker stopped.

Every run, in order (`timed_upload.jsonl`, same pipeline as the worker):

| Run | Cache | Summary s | Key points s | Verify s | Total | Summary words | Claims |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PEGASUS, after DECISION-37 | partly warm | 647 | 256 | 496 | 23.3 min | 468 | 99 |
| T5 (67 pp, 21,046 words), before DECISION-38 | leaves warm | 340 | 15 | 1,563 | 32.0 min | **1,989** | 356 |
| T5, after DECISION-38 | warm | 23 | 15 | 294 | 5.5 min | 327 | 64 |
| **PEGASUS, all changes, empty cache** | **cold** | **494** | **253** | **494** | **20.7 min** | **469** | **99** |

A rehearsed paper (cache warm) runs in minutes: only parsing and the verifiers run again.

## What was slow (measured, in order of impact)

1. **Qwen3 "thinking" was on for every summary call.** Cached summaries showed 13.7
   generated tokens per summary word (key points 1.65, decomposition 2.3, both already
   think-off). On real PEGASUS chunks: 11–80 s per chunk call, and one call ran 15 min until
   Ollama returned HTTP 500 — with the 600 s client timeout and 3 retries, one chunk could
   cost 30–40 min. Thinking off: 3.5 s per call, length on target (399 / 404 words).
2. **Verifying every merge-tree node.** PEGASUS Route B has 118 nodes, ~750 summary
   sentences and ~1,500 claims; MiniCheck scores each claim against every 500-word chunk of
   the paper (61 for PEGASUS), ~4–5 s per claim.
3. **Summary length.** The PRD's 10% rule asks a 30,000-word paper for a 3,300-word summary,
   and Qwen ignores merge targets: it concatenates the children (T5: 200 → 279 → 818 → 2,100
   words up the tree for a 500-word target), so the claim count stays high.
4. **The 16k context window did not fit the 8 GB GPU** (6 of 37 layers on the CPU,
   13–25 tok/s), and **`keep_alive=0` reloaded the model before every call** (~9 s of a
   13 s key-point call).

## What changed

| Change | Where | Effect (measured) |
| --- | --- | --- |
| Model stays loaded between calls, unloaded at job end and before the verifiers | `cli.OLLAMA_KEEP_ALIVE`, `LiveStages.aclose` (DECISION-35) | key-point call 13 s → ~3 s |
| Small window (4096/8192) for key points + decomposition | `ModelProfile.small_ctx`, `route.tokens.small_context` | whole model on the GPU, 25 → 43 tok/s |
| Summaries without thinking | `summary.think: false` (DECISION-37) | chunk call ~35 s → 3.5 s; E1 175 → 54 min |
| Ollama flash attention + 8-bit KV cache | Windows user vars `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0` | 16k fully on the GPU: 13.2 → 32.8 tok/s |
| Uploads verify the final summary only | `JobOptions.verify_scope: final` | ~1,500 → ~100 claims on PEGASUS |
| Uploads cap summaries at 500 words | `JobOptions.max_summary_words: 500` | smaller targets everywhere |
| Over-long final summary compressed once, sentence count enforced by the schema | `prompts/compress.v1.txt`, `compress_if_long` (DECISION-38) | T5 summary 1,989 → 327 words, verify 26 → 4.9 min |

Experiments (`eval/run.py`) keep per-level verification and the 10% rule; they were rerun
with thinking off (E1 run `20261031T121144-afa706`, E5 `20261031T130616-ae3547`):
faithfulness did not drop (DeBERTa A 0.68 → 0.81, B 0.61 → 0.70; MiniCheck within 0.02).

## Tried and rejected (measured)

| Idea | Result |
| --- | --- |
| `OLLAMA_NUM_PARALLEL=4` | small calls 2.4 s instead of 3.9 s, but the 16k summary call dropped to 15/37 layers on the GPU at 4.2 tok/s (6× slower) — the setting is server-wide |
| `OLLAMA_NUM_PARALLEL=2` + 8-bit KV | small calls 1.1 s, but the 16k call 29/37 layers, 9.6 tok/s (slower than today) |
| Loading the model with `use_mmap` | free-RAM dip during load 0.7 → 4.3 GB |
| GROBID heap cap `-Xmx1g` | no gain: GROBID's 3 GB is native model memory, not heap |
| Re-asking an over-long summary with the limit stated | still ~2,000 words |
| Re-summarizing an over-long summary with the single-pass prompt | 2,100 → 2,100 words (copied verbatim) |
| A smaller model, MiniCheck on only the top retrieved chunks | not tried: a model deviation / a change to how the verifier was validated (SC4) |

## Known issue found in this run

**Key points of PEGASUS come mostly from its appendix.** 5 of 8 key points are sentences of
the section *"I Example Model Outputs"* (sample news summaries the model generated, e.g.
"This is the first time a bee species has been listed as endangered in Hawaii."). The same
class of error as FRANK key point 3 (an example row from an error table). Key-point
candidates are generated per section with no notion of appendix or example sections.
Candidate fix: skip appendix / example-output sections when generating candidates; needs a
test and a re-check of the attribution numbers. **Not fixed yet.** For the live demo, prefer
papers without long example appendices, or check the key points after the rehearsal.

## Where the remaining 20 minutes go

Merges 5.5 min (Qwen writes ~13k tokens it is later asked to shorten), key points 4.2 min
(79 sections, many of them small subsections), MiniCheck ~7.4 min (99 claims × 61 chunks).
These are the next levers; each changes either the summary or the verifier and needs its own
check before adoption.

## Reproduce

```powershell
# parse once (needs GROBID: make up), then Docker can be stopped
.venv\Scripts\python.exe <scratch>\timed_upload.py parse pegasus
# empty cache = a true first run
$env:LLM_CACHE_DIR = "<empty folder>"; .venv\Scripts\python.exe <scratch>\timed_upload.py run pegasus
```

The scratch scripts (`timed_upload.py`, `bench_map.py`, `bench_big.py`, `load_dip.py`,
`profile_long.py`) are not in the repo; the supported path for a user is `make rehearse FILE=…`
(same pipeline through the API).
