<!-- Vendored verbatim from multivalue2-bundle/PAPER_CONFIG.md of the source quantization
     repository (Apache-2.0; LICENSE added at commit 0d1885e, 2026-08-26 -- see
     bitcliff/pipeline/LICENSE_AUDIT.md section 7). The bundle documents the paper draft at
     commit 8071fc44d91b15842c57cbf26a92bdd968b0d522. Extracted 2026-08-26.
     Everything below this comment is byte-identical to the source file. -->
# PAPER_CONFIG.md — the exact configuration behind the published dz numbers

Status caveat first: **"published" here means the paper draft**
(`paper/paper.md` in the source repo, drafted for NeurIPS format) at commit
`8071fc44d91b15842c57cbf26a92bdd968b0d522`. Nothing is in print; numbers below
are quoted from that draft and from the project's run records, and could still
change before submission.

## Common configuration (every multivalue2 number in the paper)

| setting | value |
|---|---|
| model / tokenizer | `Qwen/Qwen2.5-1.5B-Instruct` |
| dtype / device | float32 / Apple MPS |
| variant | `multivalue2` (frozen by calibration; enforced in code) |
| `target_tokens` | 4096 (prompt = 4143 tokens incl. chat template) |
| depths | (0.1, 0.5, 0.9) cycled by item index |
| item seed | **2024** (all headline numbers); **2025** only for disjoint re-measurement sets |
| batch size | 2 (verified to change no number; preflight asserts batch-invariance) |
| NLL grading | mean CE over the 10-token answer span (see ANSWER_TOKENS.md) |
| boolean grading | greedy, `max_new_tokens=32`, conjunctive substring (see GRADING.md) |
| quantizer group size | 64 (both `nf4` and `int_group`) |
| statistics | paired per item; sign-flip permutation p; dz = meanΔ/sdΔ (ddof=1); McNemar exact |

Item sets at seed 2024 are **nested**: n=24 ⊂ n=48 ⊂ n=96 (per-item RNG keyed
by index). The 20 samples in this bundle are the first 20 of that stream.

## The runs and their dz values

| run (config) | quantizer | n | with gen. | headline multivalue2 numbers |
|---|---|---|---|---|
| `rung1b_calibration2` (selection run) | nf4 @ 4 bits | 24 | yes | acc 0.917→0.667, McNemar p=0.0312; ΔNLL +0.0415, **dz +0.395**, perm p=0.0635 |
| `rung1b_metric_check` | nf4 @ 4 bits | 96 | yes | acc baseline 0.9167 (replicated exactly); ΔNLL +0.0291, **dz +0.221**, perm p=0.0304; McNemar p=0.0043; flip directionality 0.818 (22 discordant pairs) |
| `rung1b_precision_4bit` | int_group @ 4 | 48 | yes | **dz +0.205**, acc 0.896→0.771 (perm p=0.158, McNemar p=0.109 — fails significance, not the 0.2 threshold) |
| `rung1b_precision_3bit` | int_group @ 3 | 48 | yes | **dz +1.876**, acc 0.896→0.167 — the 3.2× asymmetry headline (arithmetic dz +0.589) |
| `rung1b_precision_2bit` | int_group @ 2 | 48 | yes | **dz +9.061**, acc 0.896→0.000 |
| `rung1b_nf4_anchor` | nf4 vs int_group @ 4, paired | 96 retr. | yes | format difference −0.0064, p=0.816 — failed rejection, not equivalence |
| `rung2_layer_sweep` (the component map) | int_group @ 3, per-component | 96 | **no** (NLL-only sweep; baseline gets both) | baseline acc 0.917 @ n=96, baseline NLL 0.2011, noise band 0.0288 (14.3% of baseline); 26/56 components under the adopted gate, 21/56 adequately powered |
| `rung2_l27_fresh_items` (disjoint re-measure) | int_group @ 3 | 48 | — | seed **2025**, zero shared item ids: `mlp_block__L27` retrieval −0.0295 (dz −1.194), p=0.0001 |

**Which dz to quote for comparability:** the project's own guidance is explicit —
use the **n=96 numbers (dz +0.221 for whole-model nf4)**, because `multivalue2`
was *selected* partly for its large calibration effect, so the n=24 dz +0.395 is
inflated by selection (winner's curse; the paper devotes §4.4 to exactly this).
The abstract's numbers are: NF4 retrieval **dz +0.221** at n=96 (1.01× the noise
band), `int_group`@4 **+0.205** at n=48, and 3-bit **+1.876** at n=48.

## Drift between the paper version and current code

Verified 2026-08-26, working tree at commit `8071fc4` (tree dirty, but only in
figure-building scripts and paper files — `src/data.py`, `src/evaluate.py`,
`src/metrics.py` are clean):

- **No drift in item construction.** This bundle's generator was checked
  in-process against the current `src/data.build_retrieval_items` for
  (multivalue2, n=20, t=4096, seed=2024): content digest
  `9220589bd8607bd0ff3be5bdcfecd23df07cac82d354d992468b15b60f398972`
  identical on both sides (digest covers item ids, all token ids, masks, and
  golds).
- **No drift in grading.** The substring rule and NLL span construction quoted
  in GRADING.md are the current code, which is the code that produced every
  post-calibration number.
- **Historical drift that predates every multivalue2 number** (listed for
  honesty, affects nothing here): before the project's determinism fix,
  `_distinct_values` was `PYTHONHASHSEED`-dependent and documents were not
  reproducible across processes. All `multivalue2` selection and measurement
  happened **after** that fix. Pre-fix numbers (the docs/00 §7 amendment ladder
  figures) are flagged unreproducible in the source project and none involve
  `multivalue2`.
- **Gate drift within the paper's own history:** the significance gate changed
  from the pre-registered noise-band rule to the dz≥0.2 rule ("B1", adopted
  2026-08-04) *before the map was drawn*; the paper reports both verdicts on
  every map. If BitCliff quotes gate verdicts (rather than dz values), say
  which gate.

## Comparability warnings for BitCliff

1. **Items are tokenizer-bound.** Construction is in token space: the filler
   window length, needle positions, and "4096 tokens" are all defined against
   the Qwen2.5 tokenizer. Building with a different tokenizer yields
   *different documents* (same seeds, different content), and per-token answer
   spans of different lengths. Same-generator ≠ same-items across models.
   Comparisons across models are task-level, never item-level, unless the
   models share a tokenizer.
2. **dz is metric-bound.** Published dz values are on teacher-forced answer-span
   NLL with the 10-token span including separators, paired per item, sd with
   ddof=1. A dz computed on accuracy, on logit divergence, or on a
   digits-only span is not comparable and should not be presented next to
   these numbers without saying so.
3. **Effects are precision- and format-bound.** The project's own preliminary
   result: component importance may not transfer 3-bit→4-bit (2 of 6 re-measured
   components reverse sign; ρ=+0.600, n.s.). Quote the quantizer
   (`nf4` vs `int_group`), bits, and group size 64 with any number.
4. **Baseline accuracy depends on n**: 0.917 at n=24 and n=96, 0.896 at n=48 —
   same distribution, different draws; not a discrepancy.
