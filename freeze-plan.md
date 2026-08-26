# BitCliff Freeze Plan — DRAFT for review

**Status: PLAN-FIRST DOCUMENT. Nothing in here executes until you approve it.**

This plan produces "the freeze" (spec §10): the public registration committed and
third-party timestamped before any confirmatory data exists, with the verified
file lists, the twin problems round-trip-checked, and the licenses audited. No
GPU money is spent and nothing confirmatory runs before the freeze commit.

It folds in: the original freeze deliverables, the `multivalue2-bundle/`
(read in full: TASK.md, MECHANISM.md, PAPER_CONFIG.md, GRADING.md,
ANSWER_TOKENS.md, PROVENANCE.md, generator), and your seven directives of
2026-08-26. Where a directive conflicts with older pilot text, the directive
wins and the older text has already been amended (PILOT_NOTES.md).

---

## 0. Naming rule (directive 1a) — applies to every deliverable

The capability measured by multivalue2 is renamed **"long-context retrieval"**
everywhere — matrix column, site copy, Picker language, PREREG, code identifiers
(`longctx_retrieval` as the suite key). Bare "retrieval" never appears. Picker
copy maps it to RAG and long-document use. The pilot's placeholder retrieval
suite is retired at integration time; its code path is deleted, not renamed —
no conclusions from it survive (already so marked in PILOT_NOTES.md).

## 1. PREREG.md draft (for your edit)

One document, structured as below, produced as a draft you edit before the
timestamped commit. Proposed content per section:

1. **Questions.** The five registered questions restated with these updates:
   - **Q4 (contamination):** the mechanism (i) anchor ("in-context retrieval
     also relies on memorization") is **void, not conditional** — MECHANISM.md
     is binding: multivalue2 is pure in-context lookup with zero parametric
     recall by construction. PREREG states plainly that **no measured evidence
     favors either sign** of the contamination effect; the test stays
     two-sided with no preferred outcome. The new closed-book factual QA suite
     (below) gives mechanism (i) a direct measurement: its degradation curve is
     memorized-content damage *observed*, not assumed, and Q4's discussion
     links to it.
   - **Q5 (k-quant transfer):** unchanged in substance. The pilot's
     IQ2_M > Q2_K arithmetic observation is cited only as an exploratory
     candidate finding (expected IQ-over-K pattern near 2.5 bpw), never as
     evidence.
   - **Headline hygiene (directive 7):** anywhere paper-repo numbers are
     cited, only the n=96 figures are quoted (retrieval dz +0.221 nf4@4bit;
     3-bit +1.876 at n=48 as the paper's own abstract states them). The n=24
     calibration figure (dz +0.395) is selection-inflated and is **never
     quoted**. A one-line citation-hygiene rule to this effect goes in PREREG
     so drift can't reintroduce it.
2. **Suites registered for launch (four):**
   - **Long-context retrieval** — multivalue2, two configurations (§2).
   - **Arithmetic** — GSM8K + embargoed twins (§4).
   - **Closed-book factual QA** — new, n=500 (§3), pending your approval of
     source and grading rule.
   - (Instruction-following and additional long-context lengths remain
     Phase 2, outside this registration.)
3. **Fixed parameters:** `max_tokens: 1024` (registered, does not churn);
   deterministic decoding (greedy, temp 0, top_k 1, seed 42) with the
   determinism scope statement (fixed hardware+software config; machine +
   library version recorded per output, as the pipeline already does).
4. **Difficulty calibration (disclosed as pilot-legitimate):** PREREG
   registers the **rule**, mirroring the paper's own pre-registered
   calibration: per model, the scored-suite configuration is the hardest
   setting whose F16 accuracy lies in **[0.6, 0.85]**, chosen using the
   documented knobs only (multivalue2: variant N, `target_tokens`, depth
   cycle; closed-book QA: popularity mix; arithmetic: none — GSM8K as-is),
   measured before any confirmatory quant runs, with the chosen configs and
   their hashes appended as a dated PREREG amendment. Registering the rule
   rather than pre-baked values keeps the ordering honest: no GPU spend before
   the freeze commit, and the per-model calibration happens after the rule is
   frozen but before confirmatory data.
5. **Statistics and the four cell states:** §6's text inlined.
6. **Tokenizer-bound comparability (directive 3):** items are constructed in
   token space; the same seed builds **different documents per model**.
   Durability and every cross-model claim are **task-level, never
   item-level** (unless models share a tokenizer). Stated in PREREG and on
   the methodology page.
7. **Answer-token spec (directive 4):** for the item-divergence question, the
   primary aggregation is over the **full 10-token answer span** per
   ANSWER_TOKENS.md — paper-consistent, with the ~20% separator dilution
   disclosed in the same paragraph. The **digits-only (8-position) mean is
   registered as a sensitivity analysis**, labeled as departing from the
   paper's metric. Alignment convention (token p scored from hidden state
   p−1) stated as in ANSWER_TOKENS.md.
8. **Grading (directive 5):** the paper rule **verbatim** for long-context
   retrieval — conjunctive plain-substring, order-insensitive, greedy
   `max_new_tokens=32`-equivalent budget within the 1024 global cap.
   Methodology notes that substring false positives are symmetric across
   paired conditions and absorbed by the paired design.
9. **Exploratory-pilot declaration:** the pilot (run pilot-0a, including the
   in-house deranged rungs) was exploratory, informed parameter and curation
   choices, and its numbers are discarded; difficulty calibration per §4 is
   disclosed as pilot-legitimate.
10. **Embargoes and dataset scope:** §8's text inlined (multivalue2 prompts
    excluded from publication; twins embargoed until the paper extension).

## 2. multivalue2 integration — two configurations (directive 2)

**Engineering.** `generate_multivalue2.py` is vendored into the pipeline as
the `longctx_retrieval` suite. Items are token-id sequences, not text prompts,
so the suite carries its own generation path (llama-cpp evaluation from token
ids, bypassing the chat-template re-render; the generator already splices the
chat template in token space). Grading per §1.8. Per-model tokenizer comes
from the model's own HF tokenizer (the generator's existing interface); the
GGUF and HF tokenizers for a given model are the same vocabulary — asserted at
build time via a token-roundtrip spot check.

**2a. Q5-comparability run (1.5B only).** Exact PAPER_CONFIG.md settings:
Qwen2.5-1.5B tokenizer, corpus `sgoel9/paul_graham_essays` pinned at sha256
`b6135331…f9329e` (the generator refuses a mismatch), variant multivalue2,
`target_tokens 4096`, depths (0.1, 0.5, 0.9) cycled, **seed 2024, n=96**
(nested-stream lineage documented: n=24 ⊂ n=48 ⊂ n=96, this run uses the same
stream as the paper's headline numbers; item-digest cross-check against the
bundle's `9220589b…` for the first 20). Purpose: bridge the paper's lab-quant
dz numbers to BitCliff's GGUF k-quants on identical items — direct input to
Q5. **These prompts are never displayed on the site and never published;
outputs and statistics only** (PG-essays license is unresolved per
PROVENANCE.md, and the paper items stay clean for the extension).

**2b. Site and dataset runs (all models).** Corpus swapped to a pinned,
hashed **public-domain Project Gutenberg text**. Proposal for your approval:
**"The Count of Monte Cristo" (PG #1184, English translation, ~2.6M chars)**
— long enough for arbitrary 4k–16k windows, continuous prose, unambiguously
public domain; the Gutenberg header/footer boilerplate is stripped by a
documented rule and the resulting text sha256-pinned; the generator's
corpus-hash gate updated to accept exactly this hash. Difficulty then
recalibrated **per model** to the F16 [0.6, 0.85] band via the registered rule
(§1.4) using N / `target_tokens` / depth. These items feed the site, the
published dataset (minus prompts, §8), and all cross-model tables.

## 3. Fourth suite: closed-book factual QA (directive 1c) — PROPOSAL

Measures **parametric factual recall** — the mechanism nothing else measures,
the folklore meaning of "quants forget facts," and the direct observation arm
for Q4.

- **Source (for your approval):** **PopQA** (`akariasai/PopQA` on HF; Mallen
  et al. 2023). 14k entity-centric questions, each with a `possible_answers`
  alias list and subject-entity Wikipedia popularity — the alias lists give
  strict grading, the popularity field gives a difficulty knob. License to
  verify against the HF API during execution (believed MIT); **fallback if
  the audit fails: TriviaQA** (unfiltered, no-context split, Apache-2.0,
  alias lists included).
- **Sampling:** n=500, fixed seed, stratified by popularity decile. The
  popularity mix is the calibration knob for the F16 [0.6, 0.85] band
  (long-tail facts fail first), applied per the registered rule and disclosed.
- **Prompt:** closed-book, no context: `Answer with just the answer:
  {question}`.
- **Grading rule (for your approval):** correct iff **any alias** from the
  item's alias list occurs in the completion under: lowercase both sides,
  normalize whitespace and strip punctuation at alias boundaries, and require
  the alias as a **word-boundary-anchored** substring. No fuzzy matching, no
  partial credit, no grader model. (Deliberately stricter than multivalue2's
  unanchored digit-substring rule — aliases are words, where unanchored
  matching is meaningfully lax; the difference and its reason are stated in
  PREREG.)
- **Q4 linkage:** this suite's degradation curve is mechanism (i) observed;
  PREREG's contamination section cites it as the measured reference for how
  memorized content degrades, while the GSM8K-twin test remains the two-sided
  contamination instrument for arithmetic.

## 4. GSM8K twins — construction plan, round-trip verifier, license audit

- **Construction:** from a templatable subset of GSM8K test items: names
  swapped from a fixed pool; numbers resampled within magnitude bands subject
  to each template's constraints (integrality, divisibility, positivity of
  intermediates) so difficulty is preserved; answers recomputed
  programmatically.
- **Round-trip verifier (executable, part of the pipeline):** each template
  carries a solution program. Verification: (a) the program run on the
  ORIGINAL numbers must reproduce the original GSM8K `####` answer exactly —
  proving the template faithfully encodes the problem; (b) the twin's answer
  is the program run on the resampled numbers; (c) the twin text re-parses
  back to the same variable assignment (text→numbers→text round trip). Any
  failure discards the template, never patches it. Fallback (spec §14): the
  smaller fully-vetted template set.
- **Registered analysis rules:** comparison restricted to items where F16
  solves both original and twin; two-sided; no preferred outcome.
- **Embargo:** instantiated twins stay out of the Random pool and the
  published dataset until the paper extension ships; the construction method
  (this section) is public from day one.
- **License audit (executed and recorded in the freeze commit):** GSM8K (MIT
  — verify); PopQA/TriviaQA per §3; `sgoel9/paul_graham_essays` — documented
  as unresolved, which is why it is quarantined to the never-published 2a run;
  Gutenberg PG #1184 — public domain, rule + hash recorded; model output
  redistribution — Qwen2.5 (Apache-2.0) and Llama 3.1 (Community License —
  verify output-publication terms and record the verbatim clause);
  **the multivalue2 generator itself — the source repo has NO LICENSE file**:
  fine for internal use (you are the author), but publishing the generator in
  the open pipeline (deliverable §8) requires you to add a license to the
  source repo or grant one for the vendored copy. **Blocked on you (§9).**

## 5. Verified reference-model file lists with SHAs

Enumerate against the live HF API, download, sha256, and commit manifests for:
- **Llama-3.1-8B-Instruct** — bartowski imatrix ladder, Q8_0 → lowest
  published, exact filenames.
- **Qwen2.5-7B-Instruct** — bartowski imatrix ladder, same.
- **Shootout files:** unsloth and mradermacher at Q4_K_M and Q3_K_M for one
  or both reference models; mradermacher's paired calibrated (imatrix "i1")
  and uncalibrated (static) versions of the same quants; Qwen's official
  GGUFs for the official-vs-community arm.
The pipeline's existing manifest machinery records uploader / imatrix status /
sha256 per file; ladders are pinned to these hashes in the registration.

## 6. Four-state and margin text (proposed numbers for your edit)

- **Margin:** M = **3 percentage points** absolute accuracy per suite
  (proposed; you set the final number per suite).
- **Per-cell inference (quant vs F16, paired on identical items):** McNemar
  exact for the accuracy difference; two-sided 95% CI on Δaccuracy by paired
  bootstrap (10k resamples).
- **Cell states:**
  - **Damaged:** CI excludes 0 and the point estimate exceeds M loss.
  - **Small real loss:** CI excludes 0 and the point estimate is within M;
    the signed loss is shown (e.g. "−1.2pp").
  - **Equivalent:** the entire CI lies within ±M (equivalence by
    CI-inclusion, the TOST-equivalent rule).
  - **Indeterminate:** none of the above; never rounded up to "free".
- **Cliff:** per capability, the highest-precision file whose cell is
  Damaged; non-monotonic rungs flagged, never smoothed (with the IQ-vs-K
  ~2.5 bpw note where applicable).
- **Multiplicity policy (proposed):** states are per-cell descriptive
  verdicts at fixed α with no cross-cell correction; stated openly in PREREG
  (the alternative — Holm across a ladder — is listed for your call).
- **Truncation:** reported per cell alongside states, per the pipeline's
  existing separation.

## 7. Blind-check protocol (guess-the-quant gate)

- Materials: mid-ladder only (Q6_K / Q5_K_M / Q4_K_M / Q3_K_M), outputs from
  the confirmatory 1.5B run, 10 rounds, distance scoring (exact / adjacent /
  otherwise nothing).
- Two raters who have never seen any outputs; the pipeline operator is not
  blind and never rates. Pass criterion (proposed): both raters exceed
  adjacent-inclusive chance at p<0.05 (exact binomial). If a second naive
  rater cannot be found: the game stays out (spec §8 fallback).
- Protocol, materials hash, and pass rule are in PREREG before any rater sees
  anything.

## 8. Dataset output spec (directive 6)

Published per-item records (outputs, grades, divergence, fragility signals)
for all suites, with these exclusions:
- **multivalue2 prompts are excluded from publication** in both
  configurations. Shipped instead: the generator (pending §4 license), the
  seeds, the corpus pointer + pinned hash (Gutenberg for 2b; the 2a corpus
  pointer documented but its text never shipped), per-item metadata (item id,
  key, depths, `n_answer_tokens`, config), model outputs, and a
  **reconstruction recipe** (tokenizer + generator + seed + corpus hash →
  bit-identical items, verified by the digest mechanism the bundle already
  provides).
- Twins embargoed per §4.
- Closed-book QA items are public-source; published normally.

## 9. Blocked on you (short list)

1. Approve/adjust the **closed-book QA source and grading rule** (§3).
2. Approve the **Gutenberg corpus choice** (PG #1184) or name another (§2b).
3. Approve **margin M = 3pp** and the multiplicity choice (§6).
4. **Add a LICENSE** to the `quantization` repo (or grant terms for the
   vendored generator) before the open pipeline ships (§4, §8).
5. **Edit and sign off PREREG.md**, then choose the timestamp mechanism —
   proposed: **OpenTimestamps** on the PREREG commit hash (free,
   Bitcoin-anchored, verifiable offline).
6. Confirm the **shootout scope** (§5: which reference model(s) carry the
   uploader shootout at Q4+Q3).

## 10. Execution order (after your approval of this plan)

1. **F1 — engineering:** vendor the generator; `longctx_retrieval` suite +
   token-id generation path; closed-book QA suite; twins builder + round-trip
   verifier; delete the placeholder retrieval suite; rename surfaces to
   long-context retrieval. (All local, tested, no confirmatory runs.)
2. **F2 — audits & lists:** license audit (§4); reference-model file
   enumeration + download + hashing (§5). Downloads are free; no GPU spend.
3. **F3 — PREREG assembly:** draft per §1 with §6/§7/§8 text inlined; hand
   to you for edit.
4. **F4 — the freeze commit:** your edited PREREG + manifests + twins
   templates (hashes only for embargoed instances) committed; OpenTimestamps
   stamp; freeze declared. Difficulty calibration then runs under the
   registered rule (F16-only, local), its amendment is appended and stamped,
   and 0B confirmatory begins per aws-ops.

**Not in this plan:** anything from Phase 0B onward (confirmatory runs, site,
launch) — those keep their own plans.
