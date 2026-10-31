# Chapter 5 — Results and Discussion (draft)

> Draft text for the team report. **Tables and figures are not typed here** (SC5): paste them
> from `results/report/tables.md`, `results/report/e3_levels.png` and
> `results/report/e5_perturbations.png`, all produced by `make report` from the run logs.
> Run ids: E1 `20261030T215019-e30656`, E5 `20261031T004522-290644`.
> Numbers quoted in the prose below were copied from those files on 2026-10-31; if a run is
> repeated, re-check them.

## 5.1 Experimental setup actually run

The PRD planned five experiments on a 50-document gold set with human annotation. Because of
the deadline, a **pilot** was run instead (scope agreed with the team; guide confirmation
pending):

- **Documents:** 5 CC-licensed arXiv papers parsed by GROBID (Sentence-BERT, DeBERTaV3, FRANK,
  TRUE, ChatGPT factuality), 6.7k–9.7k tokens each. The paper's own abstract is removed from
  the input and used as the ROUGE reference.
- **Model:** Qwen3-8B (Q4_K_M) through Ollama on an 8 GB laptop GPU, for every generation step
  (DECISION-27). Gemini was not used for any result.
- **Verifiers:** DeBERTa-v3-large NLI and MiniCheck (Flan-T5-Large), both run locally and
  offline, one model on the GPU at a time.
- **One seed** (1234), temperature 0.2, all LLM calls cached and logged.

Experiments run: **E1** (Route A vs forced Route B), **E3** (faithfulness per merge level,
from the E1 Route B runs), **E5** (verifiers under controlled perturbations), **SC4**
(verifier accuracy on the AggreFact test split). E2 (context-budget sweep) and E4
(section-aware vs uniform chunking) were not run (§5.7).

## 5.2 Verifier calibration (Table 5.1, Table 5.2)

Entailment thresholds were tuned on a stratified sample of the **LLM-AggreFact dev** split
(4349 claims, 400 per sub-benchmark) and frozen before any test data was loaded
(DECISION-28).

- **DeBERTa** works best with a low threshold, θ_e = 0.10 (balanced accuracy 0.675 vs 0.660 at
  the default 0.5). Its entailment scores against a few retrieved sentences run low when the
  evidence for a claim is spread over several sentences.
- **MiniCheck** is flat between 0.25 and 0.6 (0.765–0.766). The sweep's best value (0.22) beats
  0.5 by less than 0.001, so the published default **0.5 was kept**. Table 5.1 shows both.

**SC4 — held-out test split** (4358 claims, thresholds frozen): MiniCheck reaches **F1 0.809,
balanced accuracy 0.771**; DeBERTa F1 0.648, balanced accuracy 0.666. Test accuracy is within
0.01 of dev for both verifiers, so the thresholds did not overfit the dev sample. MiniCheck's
0.77 is in the range published for Flan-T5-Large on LLM-AggreFact, which supports our
re-implementation. DeBERTa's recall is low (0.54) even at θ_e = 0.10: it misses many supported
claims, and MiniCheck is better on precision too (0.845 vs 0.810). Both verifiers are weakest on AggreFact-CNN and
ExpertQA (MiniCheck 0.55 on ExpertQA).

## 5.3 E1 — single pass (Route A) vs map-reduce (Route B) (Table 5.3)

All five papers fit the 13,107-token window, so Route A is a single prompt; Route B is forced
with a 4,000-token budget (17–46 merge-tree nodes per paper).

| Finding | Evidence |
| --- | --- |
| Route A is at least as faithful as Route B | mean faithfulness A 0.684 vs B 0.611 (DeBERTa), 0.988 vs 0.944 (MiniCheck). MiniCheck prefers A on 5/5 papers; DeBERTa on 3/5 (B is higher on Sentence-BERT and TRUE) |
| Route A has higher ROUGE | ROUGE-1 0.436 vs 0.273, on 5/5 papers |
| **But the ROUGE gap is mostly length** | B summaries are 2.5× longer (576 vs 234 words, column *Words/summary*): the merge target is the sum of the children's targets, so the final summary grows with the number of chunks. A longer summary against a ~150-word abstract lowers ROUGE precision and F1 regardless of quality |
| Route B costs ~9× more time | 2324 vs 256 s per paper (more LLM calls, run one at a time on one GPU) |

**Reading:** when a paper fits the context window, sending it whole is cheaper and no less
faithful. The ROUGE comparison should not be read as a quality ranking because the two
routes produce summaries of different length; faithfulness is a per-claim rate and is less
sensitive to length. A length-matched Route B run (≈4 GPU hours) is the main follow-up.

## 5.4 E3 — does faithfulness degrade up the merge tree? (Figure 5.1, Table 5.4)

Faithfulness is essentially **flat** across merge levels: DeBERTa 0.63 / 0.60 / 0.61 / 0.62
and MiniCheck 0.94 / 0.94 / 0.94 / 0.95 for levels 0–3 (level 3 exists for only 3 of the 5
papers). On these papers, recursive merging did **not** introduce measurable extra
unfaithfulness; the drop seen in E1 is already present in the level-0 chunk summaries. With 5
papers and trees at most 4 levels deep, this cannot rule out degradation in much longer
documents.

## 5.5 E5 — verifier behaviour under perturbation (Figure 5.2, Table 5.5)

Claims from the E1 summaries were perturbed by rule (negation, number/entity corruption, an
inserted assertion) or paraphrased by the LLM; a correct verifier rejects the first three
and still supports paraphrases.

- **Negation** (28 items): DeBERTa 0.75 correct, MiniCheck 0.64.
- **Corruption** (25): DeBERTa 0.92, MiniCheck 0.80.
- **Paraphrase** (50): MiniCheck 0.98, DeBERTa 0.90.
- **Assertion** (5): DeBERTa 0.60, MiniCheck **0.00**. The inserted sentence is the fixed
  meta-claim *"The summary is fully supported by the document."*; MiniCheck accepted it for
  every paper. Only 5 items, but it shows MiniCheck can label a content-free claim as
  supported.

**Reading:** the verifiers fail in complementary ways. MiniCheck is more robust to rewording
and better calibrated overall (SC4), while DeBERTa is stricter about negations and changed
facts. Reporting both, as the system does, is more informative than either alone.

## 5.6 Key-point attribution

For the 24 key points of the three demo papers, the sentence the system links each key point
to was checked against a reference labelling: **precision@1 = 0.667** (16/24; per paper 0.50,
0.75, 0.75). The reference was produced **by an LLM annotator (Claude), blind to the system's
choice, and spot-checked by a team member** — it is not a human gold set (DECISION-29). Most
misses are sentences about the same entity but a different claim, which embedding similarity
cannot separate; re-ranking the top candidates with the NLI model is the natural fix.

## 5.7 Limitations

- **Pilot scale:** 5 papers, 1 seed, no confidence intervals; differences of a few points are
  within noise.
- **One model:** Qwen3-8B only; no cloud model, so results may not transfer to larger LLMs.
- **Length confound in E1** (§5.3): Route B summaries are 2.5× longer.
- **Not run:** E2 (budget sweep), E4 (section-aware vs uniform chunking), the 50-document gold
  set, and human claim annotation with inter-annotator agreement.
- **Thresholds:** θ_c (contradiction) is untuned, because AggreFact labels are binary.
- **No LLM judge** in E5; perturbations are rule-based except paraphrase.
- **Attribution reference** is LLM-labelled, 24 items.
- **Token counts** use a characters/4 estimate (not checked against the tokenizer).
- **Deviations from the PRD** (guide to confirm): Qwen-only, pilot scope, claim-decomposition
  prompt v2 (DECISION-15).

## 5.8 Summary of answers to the research questions

1. *Does map-reduce hurt faithfulness when the document fits?* Not significantly with
   DeBERTa (mixed per paper); slightly by MiniCheck. It is much slower and, as built, longer.
2. *Does faithfulness decay with merge depth?* Not on these papers (flat across levels 0–3).
3. *Can local verifiers be trusted?* MiniCheck reaches F1 0.81 on held-out AggreFact and
   transfers from dev to test; both verifiers have known blind spots (E5), so their labels
   are a screening signal, not ground truth.
