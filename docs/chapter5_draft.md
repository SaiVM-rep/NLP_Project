# Chapter 5 — Results and Discussion

> Draft text for the team report. **Tables and figures are not typed here** (SC5): the
> `<!-- table: … -->` / `<!-- figure: … -->` markers are filled from `results/report/` by
> `make chapter5`, which writes `docs/chapter5.md` (the file to paste into the report).
> Run ids: E1 `20261031T121144-afa706`, E5 `20261031T130616-ae3547` (summaries without
> thinking, DECISION-37); the earlier thinking-on E1 run `20261030T215019-e30656` is kept for
> §5.4. Numbers quoted in the prose were copied from those files on 2026-10-31; if a run is
> repeated, re-check them.

## 5.1 Experimental setup actually run

The PRD planned five experiments on a 50-document gold set with human annotation. Because of
the deadline, a **pilot** was run instead (scope decided for the deadline; guide confirmation
pending):

- **Documents:** 5 CC-licensed arXiv papers parsed by GROBID (Sentence-BERT, DeBERTaV3, FRANK,
  TRUE, ChatGPT factuality), 6.7k–9.7k tokens each. The paper's own abstract is removed from
  the input and used as the ROUGE reference.
- **Model:** Qwen3-8B (Q4_K_M) through Ollama on an 8 GB laptop GPU, for every generation step
  (DECISION-27). Gemini was not used for any result.
- **Verifiers:** DeBERTa-v3-large NLI and MiniCheck (Flan-T5-Large), both run locally and
  offline, one model on the GPU at a time.
- **One seed** (1234), temperature 0.2, all LLM calls cached and logged.
- **No hidden reasoning:** Qwen3's "thinking" mode is switched off for every call (summaries,
  key points, claim decomposition). An earlier E1 run with thinking on for summaries is
  compared in §5.4.

Experiments run: **E1** (Route A vs forced Route B), **E3** (faithfulness per merge level,
from the E1 Route B runs), **E5** (verifiers under controlled perturbations), **SC4**
(verifier accuracy on the AggreFact test split). E2 (context-budget sweep) and E4
(section-aware vs uniform chunking) were not run (§5.8).

## 5.2 Verifier calibration (Table 5.1, Table 5.2)

Entailment thresholds were tuned on a stratified sample of the **LLM-AggreFact dev** split
(4349 claims, 400 per sub-benchmark) and frozen before any test data was loaded
(DECISION-28).

- **DeBERTa** works best with a low threshold, θ_e = 0.10 (balanced accuracy 0.675 vs 0.660 at
  the default 0.5). Its entailment scores against a few retrieved sentences run low when the
  evidence for a claim is spread over several sentences.
- **MiniCheck** is flat between 0.25 and 0.6 (0.765–0.766). The sweep's best value (0.22) beats
  0.5 by less than 0.001, so the published default **0.5 was kept**. Table 5.1 shows both.

<!-- table: Verifier thresholds -->

**SC4 — held-out test split** (4358 claims, thresholds frozen): MiniCheck reaches **F1 0.809,
balanced accuracy 0.771**; DeBERTa F1 0.648, balanced accuracy 0.666. Test accuracy is within
0.01 of dev for both verifiers, so the thresholds did not overfit the dev sample. MiniCheck's
0.77 is in the range published for Flan-T5-Large on LLM-AggreFact, which supports our
re-implementation. DeBERTa's recall is low (0.54) even at θ_e = 0.10: it misses many supported
claims, and MiniCheck is better on precision too (0.845 vs 0.810). Both verifiers are weakest on AggreFact-CNN and
ExpertQA (MiniCheck 0.55 on ExpertQA).

<!-- table: SC4 -->

## 5.3 E1 — single pass (Route A) vs map-reduce (Route B) (Tables 5.3, 5.4)

All five papers fit the 13,107-token window, so Route A is a single prompt; Route B is forced
with a 4,000-token budget (17–46 merge-tree nodes per paper).

<!-- table: E1 -->

| Finding | Evidence |
| --- | --- |
| Route A is more faithful than Route B | mean faithfulness A 0.813 vs B 0.700 (DeBERTa), 0.979 vs 0.959 (MiniCheck). Each verifier prefers A on 4/5 papers (DeBERTa: B higher on Sentence-BERT; MiniCheck: B higher on TRUE), Table 5.4 |
| Route A has higher ROUGE | ROUGE-1 0.485 vs 0.307, on 5/5 papers |
| **But the ROUGE gap is mostly length** | B summaries are 2.5× longer (582 vs 235 words, column *Words/summary*): the merge target is the sum of the children's targets, so the final summary grows with the number of chunks. A longer summary against a ~150-word abstract lowers ROUGE precision and F1 regardless of quality |
| Route B costs ~7× more time | 571 vs 81 s per paper (more LLM calls, run one at a time on one GPU) |

<!-- table: E1 per paper -->

**Reading:** when a paper fits the context window, sending it whole is cheaper and more
faithful. The ROUGE comparison should not be read as a quality ranking because the two
routes produce summaries of different length; faithfulness is a per-claim rate and is less
sensitive to length. A length-matched Route B run (≈4 GPU hours) is the main follow-up.

## 5.4 Summary thinking on vs off (Table 5.5)

The first E1 run used Qwen3's default "thinking" for summary calls; the final run switched it
off (DECISION-37) after profiling showed it generated about 14 tokens of hidden reasoning per
summary word. Turning it off made summarization **3× (Route A) to 4× (Route B) faster** (256 →
81 s and 2324 → 571 s per paper) and did **not** reduce faithfulness: DeBERTa rose from 0.684 to
0.813 (A) and from 0.611 to 0.700 (B); MiniCheck moved from 0.988 to 0.979 (A) and from 0.944 to
0.959 (B); ROUGE-1 rose on both routes. Summary length was unchanged (~235 / ~580 words). The
direction of every E1 finding is the same in both runs.

<!-- table: E1 thinking -->

## 5.5 E3 — does faithfulness degrade up the merge tree? (Figure 5.1, Table 5.6)

Faithfulness is close to **flat** across merge levels: DeBERTa 0.71 / 0.71 / 0.70 / 0.69 and
MiniCheck 0.96 / 0.96 / 0.96 / 0.95 for levels 0–3 (level 3 exists for only 3 of the 5 papers),
a decline of at most 0.02 per verifier. On these papers, recursive merging did **not** introduce
substantial extra unfaithfulness; most of the A–B gap in E1 is already present in the level-0
chunk summaries. With 5
papers and trees at most 4 levels deep, this cannot rule out degradation in much longer
documents.

<!-- table: E3 -->

<!-- figure: e3_levels.png | Faithfulness per merge level (Route B arm of E1) -->

## 5.6 E5 — verifier behaviour under perturbation (Figure 5.2, Table 5.7)

Claims from the E1 summaries were perturbed by rule (negation, number/entity corruption, an
inserted assertion) or paraphrased by the LLM; a correct verifier rejects the first three
and still supports paraphrases.

<!-- table: E5 -->

<!-- figure: e5_perturbations.png | Share of perturbed claims judged correctly, per verifier -->

- **Negation** (41 items): DeBERTa 0.76 correct, MiniCheck 0.73.
- **Corruption** (19): DeBERTa 0.89, MiniCheck 0.74.
- **Paraphrase** (50): MiniCheck 1.00, DeBERTa 0.88.
- **Assertion** (5): DeBERTa 0.60, MiniCheck **0.00**. The inserted sentence is the fixed
  meta-claim *"The summary is fully supported by the document."*; MiniCheck accepted it for
  every paper. Only 5 items, but it shows MiniCheck can label a content-free claim as
  supported.

**Reading:** the verifiers fail in complementary ways. MiniCheck is more robust to rewording
and better calibrated overall (SC4), while DeBERTa is stricter about negations and changed
facts. Reporting both, as the system does, is more informative than either alone.

## 5.7 Key-point attribution

For the 24 key points of the three demo papers, the sentence the system links each key point
to was checked against a reference labelling: **precision@1 = 0.667** (16/24; per paper 0.50,
0.75, 0.75). The reference was produced **by an LLM annotator (Claude), blind to the system's
choice** — it is not a human gold set, and no human agreement check was completed before
submission (DECISION-29). Most
misses are sentences about the same entity but a different claim, which embedding similarity
cannot separate.

**Exploratory NLI re-ranking.** Re-ranking MiniLM's top-5 candidates by DeBERTa entailment
(premise = candidate sentence, hypothesis = key point) raises precision@1 to **0.875**
(21/24, every paper 0.875); one of the three remaining misses has no correct sentence in the
top 5 at all. This was measured on the same 24 key points, with no held-out set and nothing
tuned, so it is reported as exploratory evidence that a cross-encoder fixes most
same-entity confusions, not as a validated improvement. It is not used by the demo system.
(`scripts/attr_rerank.py`, `results/attribution/summary_rerank.json`.)

## 5.8 Limitations

- **Pilot scale:** 5 papers, 1 seed, no confidence intervals; differences of a few points are
  within noise.
- **One model:** Qwen3-8B only; no cloud model, so results may not transfer to larger LLMs.
- **Length confound in E1** (§5.3): Route B summaries are 2.5× longer.
- **Not run:** E2 (budget sweep), E4 (section-aware vs uniform chunking), the 50-document gold
  set, and human claim annotation with inter-annotator agreement.
- **Thresholds:** θ_c (contradiction) is untuned, because AggreFact labels are binary.
- **No LLM judge** in E5; perturbations are rule-based except paraphrase.
- **Attribution reference** is LLM-labelled, 24 items.
- **Upload settings differ from the experiments:** for live uploads the system verifies only
  the final summary and caps it at 500 words, to keep long papers interactive (a 55-page
  paper in 23 min, DECISION-37); the experiments verify every merge level and use the PRD's
  10% length rule.
- **Token counts** use a characters/4 estimate. Checked against the Qwen3 tokenizer it misses
  the ±5% target (FR-2.1): on our GROBID-parsed papers it overestimates by 3–16% (median 8%),
  which only errs toward Route B; on preprocessed arXiv/PubMed text the error is −25% to +36%
  (`scripts/token_check.py`, `results/token_check.json`).
- **Deviations from the PRD** (guide to confirm): Qwen-only, pilot scope, claim-decomposition
  prompt v2 (DECISION-15).

## 5.9 Summary of answers to the research questions

1. *Does map-reduce hurt faithfulness when the document fits?* Yes, moderately: both
   verifiers rate Route A higher on average and on 4 of 5 papers (DeBERTa 0.81 vs 0.70,
   MiniCheck 0.98 vs 0.96). Map-reduce is also ~7× slower and, as built, 2.5× longer.
2. *Does faithfulness decay with merge depth?* Only marginally on these papers (≤ 0.02 from
   level 0 to level 3); the gap is already present in the chunk summaries.
3. *Can local verifiers be trusted?* MiniCheck reaches F1 0.81 on held-out AggreFact and
   transfers from dev to test; both verifiers have known blind spots (E5), so their labels
   are a screening signal, not ground truth.
