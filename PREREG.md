# BitCliff — Public Pre-Registration

**STATUS: DRAFT FOR USER EDIT. NOT YET FROZEN. NOT YET TIMESTAMPED.**

This document registers, before any confirmatory data exists, every question
BitCliff will answer, every suite it will run, every parameter, every grading
rule, every statistical rule, and every embargo. When it is frozen (§15), the
commit containing the user-edited version of this file is stamped with
OpenTimestamps, so "the rules were decided after seeing the data" is
impossible by construction.

**How to read the markers:**

- `[DECISION REQUIRED — …]` — an open decision the user must make before the
  freeze commit. The freeze cannot happen while any of these remain.
- `[USER-EDITABLE — …]` — a registered value the user may still change during
  the edit pass; the value shown is the current ruling or proposal.
- `[TO BE FILLED AT F4]` / `[AMENDMENT SLOT]` — mechanical placeholders filled
  at freeze time or by a registered amendment procedure (§7).

Evidence artifacts cited by filename below live in this repository (paths
relative to the repo root) and are committed alongside this document; where a
cited artifact is itself excluded from the repo (corpora, model files, the
embargoed twin instances), its sha256 recorded in the cited manifest is the
durable record.

---

## 1. Naming rule

The capability measured by the multivalue2 task is named **"long-context
retrieval"** everywhere — in this registration, in code (suite key
`longctx_retrieval`), in every table, and in all site and social copy. The
bare word "retrieval" never appears as a capability or suite name. Site copy
maps the capability to RAG and long-document use. The pilot's placeholder
suite that previously carried a similar name was exploratory, is deleted (not
renamed), and no conclusion from it survives (§12; `bitcliff/pipeline/PILOT_NOTES.md`).

---

## 2. The five registered questions

All five questions are registered two-sided with no preferred outcome. Every
verdict — including null and inconvenient verdicts — is published.

**Q1 — Does corpus KLD rank files correctly?** Does the divergence metric the
community cites (corpus KLD, token-averaged over generic text) rank quantized
files in the order of their measured capability damage?

**Q2 — Does at-answer divergence predict item flips?** Does divergence
measured at the moment of answering predict which specific questions a quant
gets wrong, at the item level?

**Q3 — Can F16 predict its own fragile answers?** Can the uncompressed model
alone identify which of its own answers are fragile, before any quant is run?

**Q4 — Which direction does training-data contamination bend arithmetic
measurements?** If models partially memorized GSM8K, "arithmetic damage"
partly measures how memorization degrades, not how calculation degrades.
Instrument: the GSM8K-twin comparison (§3.3), two-sided.

- **The mechanism-(i) anchor is void, not conditional.** The originally
  hypothesized anchor for one direction — "in-context retrieval also relies
  on memorization" — is void:
  `bitcliff/pipeline/vendor/bundle-docs/MECHANISM.md` is binding.
  multivalue2 is pure in-context lookup with **zero parametric recall by
  construction** (every passcode is drawn fresh per item and exists nowhere
  outside that item's prompt); it cannot detect knowledge loss even in
  principle. Accordingly, **no measured evidence favors either sign of the
  contamination effect**, and this registration states that plainly. The test
  stays two-sided with no preferred outcome.
- **Mechanism (i) gets a direct measurement instead of an assumption.** The
  closed-book factual QA suite (§3.4) measures parametric factual recall —
  memorized-content damage *observed*, not inferred. Q4's discussion will
  cite that suite's degradation curve as the measured reference for how
  memorized content degrades under quantization, while the GSM8K-twin test
  remains the contamination instrument for arithmetic.

**Q5 — Does the published asymmetry transfer to k-quants?** The paper found
long-context retrieval damaged roughly 3.2× more than arithmetic at 3-bit
(`int_group` @ 3-bit, dz +1.876 vs arithmetic dz +0.589, n=48 — the
`int_group` scheme only, per §13). Do the
downloadable GGUF k-quants show the same profile? The launch copy never
asserts the 3.2× figure for k-quants; configuration 2a (§3.1) is the bridge
measurement. The pilot's IQ2_M > Q2_K arithmetic observation (0.400 vs 0.325
at ~2.5 bpw, run `pilot-0a`) is cited **only as an exploratory candidate
finding** — the expected IQ-over-K pattern near 2.5 bpw — never as evidence;
those numbers are discarded per §12.

---

## 3. Registered suites (four scored suites)

Four scored suites are registered for launch: **long-context retrieval**
(`longctx_retrieval`), **arithmetic — GSM8K originals** (`arithmetic`),
**arithmetic — twins** (`arithmetic_twins`, embargoed), and **closed-book
factual QA** (`factual_qa`). Unscored spectacle prompts exist for the
playground only and enter no statistic. Instruction-following and additional
long-context lengths are Phase 2, outside this registration.

### 3.1 Long-context retrieval (`longctx_retrieval`)

The multivalue2 task, vendored from the source project (Apache-2.0 as of
commit `0d1885e`, 2026-08-26; `bitcliff/pipeline/LICENSE_AUDIT.md` §7). Each item hides two
4-digit passcodes behind NATO-alphabet keys at registered depths inside ~4k
tokens of filler; the model must emit both values. Per
`bitcliff/pipeline/vendor/bundle-docs/MECHANISM.md`, success is in-context copying (attention
transport), not stored knowledge — stated wherever the capability is
described.

Items are **token-id sequences**, not text prompts; evaluation runs from
token ids, bypassing chat-template re-rendering (the generator splices the
chat template in token space). The GGUF and HF tokenizers for a given model
are asserted identical by
`suites/longctx_retrieval.py::assert_tokenizer_match` — both tokenizers
encode each sample string (HF with `add_special_tokens=False`; llama-cpp's
`tokenize` with BOS/special-token handling configured to match) and any
id-sequence mismatch raises, aborting the run. The check is **asserted
before each confirmatory generation run over a registered 20-string
sample**: the question strings of the run's first 20 items, in id order.

**Configuration 2a — Q5 comparability run (Qwen2.5-1.5B-Instruct only).**
Exact `bitcliff/pipeline/vendor/bundle-docs/PAPER_CONFIG.md` settings:
Qwen2.5-1.5B tokenizer;
corpus `sgoel9/paul_graham_essays` pinned at sha256
`b6135331a3132d08cb84262870ae8f9d9acb6bae4cd7f0278926a64c38f9329e` (the
generator refuses a mismatch; independently recomputed and verified —
`bitcliff/pipeline/CORPUS_MANIFEST.md` §2); variant `multivalue2`; `target_tokens` 4096;
depths (0.1, 0.5, 0.9) cycled; **seed 2024, n=96**. Lineage: the seed-2024
item sets are nested (n=24 ⊂ n=48 ⊂ n=96, per-item RNG keyed by index), and
this run uses the same stream as the paper's headline numbers;
item-construction fidelity is verified by digest — the vendored generator
with the real tokenizer reproduces the bundle's first-20-item digest
`9220589bd8607bd0ff3be5bdcfecd23df07cac82d354d992468b15b60f398972` exactly
(`bitcliff/pipeline/CORPUS_MANIFEST.md` §3). Purpose: bridge the paper's lab-quant dz
numbers to GGUF k-quants on identical items — direct input to Q5.
**These prompts are never displayed on the site and never published; outputs
and statistics only.** The PG-essays corpus's underlying redistribution terms
are unresolved (`bitcliff/pipeline/LICENSE_AUDIT.md` §4: the uploader's MIT tag cannot
license Paul Graham's text), so the corpus is quarantined to this
never-published run; the paper's items also stay clean for the paper
extension.

**Configuration 2b — site and dataset runs (all models).** Corpus: **"The
Count of Monte Cristo," Project Gutenberg #1184** (binding ruling
2026-08-26). Exact provenance per `bitcliff/pipeline/CORPUS_MANIFEST.md` §1:

- Source: `https://www.gutenberg.org/cache/epub/1184/pg1184.txt`, retrieved
  2026-08-26, `text/plain; charset=utf-8`, 2,787,124 bytes.
- Raw download sha256:
  `64f8d5cfa51fcecb904abf7312d395d512a71817e7359b91288beb50517c3836`.
- Strip rule (verbatim): text strictly between the line beginning
  `*** START OF THE PROJECT GUTENBERG EBOOK …` and the line beginning
  `*** END OF THE PROJECT GUTENBERG EBOOK …`, both marker lines excluded; no
  other normalization.
- Stripped text: 2,688,565 chars / 2,767,256 bytes (UTF-8), sha256
  `0a21a13834b5215876bd4019af8fbc436abbfbb61b2826db62223eb990071443`. The
  generator's corpus-hash gate accepts exactly this hash for 2b.

Difficulty for 2b is calibrated per model under the registered rule of §7
(knobs: variant, chosen along the registered ladder order, and
`target_tokens`; the depth cycle is fixed). These items feed the site,
the published dataset (minus prompts, §11), and all cross-model tables.

**Grading — the paper rule, verbatim**
(`bitcliff/pipeline/vendor/bundle-docs/GRADING.md`;
pipeline implementation `suites/longctx_retrieval.py`):

```python
hit = all(s in text for s in item.match_strings)
```

An item is correct iff **every** gold passcode appears as a plain substring
of the decoded completion. Conjunctive, order-insensitive,
separator-insensitive; no fuzzy matching, no partial credit, no grader model.
Generation for this suite is greedy with a **32-token answer budget**
(`max_new_tokens=32`-equivalent, registered per-suite override within the
1024 global cap of §6). Known laxity, registered as such: the substring match
is not boundary-anchored, so a gold code inside a longer digit run counts.
**Symmetry note:** substring false positives are symmetric across paired
conditions — F16 and quant are graded by the identical rule on identical
items — and are absorbed by the paired design; they bias levels, not paired
differences, except where a quant's emission style differs, which the
published per-item outputs make auditable.

**Answer-token spec** (`bitcliff/pipeline/vendor/bundle-docs/ANSWER_TOKENS.md`): for the
item-divergence question (Q2), the primary aggregation is over the **full
10-token answer span** (`"v1, v2"` under the Qwen2.5 tokenizer: 8 digits + 2
separators), paper-consistent; the ~20% separator dilution of the per-token
mean is disclosed here in the same breath. The **digits-only 8-position mean
is registered as a sensitivity analysis**, labeled as departing from the
paper's metric. Alignment convention: the token at position p is scored from
the hidden state at position p−1; logits for the first answer token come from
the last prompt token; right-padding only, so padding never precedes a scored
position.

**Tokenizer-bound comparability rule.** Items are constructed in token
space: the same seed builds **different documents for models with different
tokenizers**, with different answer-span lengths. Therefore durability and
every cross-model claim are **task-level, never item-level**, unless the
models share a tokenizer. This rule is registered and repeated on the
methodology page.

### 3.2 Arithmetic — GSM8K originals (`arithmetic`)

GSM8K test split (`openai/gsm8k`, MIT — verified, `bitcliff/pipeline/LICENSE_AUDIT.md`
§3). **Registered confirmatory item set: n = 500
`[USER-EDITABLE — proposed]`, sampled with fixed seed 3141
`[USER-EDITABLE — proposed; fresh, distinct from every exploratory seed
(pilot arithmetic 1301, twins 1301, factual_qa characterization 7411)]`.**
The seed and n are fixed by this registration and never float (§7 amendment
scope). Prompt appended with the registered instruction ("Solve step by
step, then give the final answer on its own line as: `#### <number>`").
Grading: the final `#### <number>` (falling back to the last number in the
output), normalized, exact match against the gold answer. No difficulty knob
(§7): GSM8K is used as-is.

### 3.3 Arithmetic — twins (`arithmetic_twins`, embargoed)

**Construction.** From a templatable subset of GSM8K test items: names
swapped from a fixed pool; numbers resampled within magnitude bands subject
to each template's constraints (integrality, divisibility, positivity of
intermediates) so difficulty is preserved; answers recomputed
programmatically. The twins cannot have been memorized during training,
because they did not exist.

**Round-trip verifier (executable, part of the pipeline;
`src/bitcliff_pipeline/twins/`).** Each template carries a solution program.
Verification: (a) the program run on the ORIGINAL numbers must reproduce the
original GSM8K `####` answer exactly — proving the template faithfully
encodes the problem; (b) the twin's answer is the program run on the
resampled numbers; (c) the twin text re-parses back to the same variable
assignment (text→numbers→text round trip). Any failure discards the
template, never patches it.

**Verified template set: 47 templates**
(`src/bitcliff_pipeline/twins/templates.py`), each passing the round-trip
verifier against the real GSM8K text (enforced by
`tests/test_twins.py::test_every_template_verifies_against_real_gsm8k_text`;
skipped items are recorded with reasons in `SKIPPED_ITEMS`). The instantiated
twin set (`private/twins/twin_set_seed1301.jsonl`, seed 1301, 47 records) is
**embargoed** — the file is excluded from the repository and the published
dataset; its sha256 is committed here as the integrity anchor:
`2002447536db8560c7160fc80ec7ba4f1c56074a1a949c1322114e8263d9e14b`.

**Registered analysis rules.** The `arithmetic_twins` suite evaluates **all
47 original+twin pairs — both members of each pair run in this same suite**
(the originals here are the 47 templated GSM8K items themselves, evaluated
alongside their twins under identical settings; the §3.2 sample is a
separate item set and is not the contamination comparator). The shipped
runnable path is
`src/bitcliff_pipeline/suites/arithmetic_twins.py::build_pair_items` — wired
through `__main__.build_items` via an `arithmetic_twins: {seed: 1301}` config
block — producing the 94 pair items (ids
`arithmetic_twins-{seed}-orig|twin-{gsm8k_index:03d}`), each prompted with
the §3.2 answer instruction and graded by the identical §3.2 rule. The
original-vs-twin contamination comparison operates **within those pairs**
and is restricted to pairs where **the F16 model solves both the original
and the twin** — comparing on different populations would rig the result
mechanically. Two-sided; no preferred outcome. A cut headline ships as its
own honest post.

**Embargo.** Instantiated twins stay out of the public Random pool and the
published dataset until the paper extension ships; the construction method
(this section, and the template/verifier source code) is public from day one.

### 3.4 Closed-book factual QA (`factual_qa`)

Measures **parametric factual recall** — the mechanism nothing else in this
registration measures, the folklore meaning of "quants forget facts," and the
direct observation arm for Q4.

**Source: PopQA** (`akariasai/PopQA`, test split; Mallen et al. 2023).
License: **MIT via the canonical source release**
(`github.com/AlexTMallen/adaptive-retrieval`, which ships `data/popQA.tsv`
under its MIT LICENSE; the HF mirror card carries no license tag). The audit
was executed and committed **before any suite code was written**
(`bitcliff/pipeline/LICENSE_AUDIT.md` §1, verified 2026-08-26), per binding ruling.
**Named fallback: TriviaQA** (`mandarjoshi/trivia_qa`, unfiltered no-context
split; HF license tag reads `unknown` — a fresh audit against the original
release terms is required before any switch; `bitcliff/pipeline/LICENSE_AUDIT.md` §2).

**Sampling:** n=500 `[USER-EDITABLE — currently 500 per ruling 2026-08-26]`,
**confirmatory fixed seed 2718** `[USER-EDITABLE — proposed; deliberately
distinct from the characterization run's seed 7411]`, **stratified by
subject-entity popularity decile** (`s_pop`, Wikipedia monthly pageviews):
records sorted by popularity, split into 10 contiguous deciles, draws per
decile per the registered mix with remainders to the earliest deciles
(`suites/factual_qa.py`). **The popularity MIX is the only calibration knob
(§7, long-tail facts fail first); the seed never floats** — it is fixed by
this registration and is outside the §7 amendment scope.

**Prompt** (registered): `Answer with just the answer: {question}` —
closed-book, no context. Per-suite answer budget: 64 tokens within the
global cap `[USER-EDITABLE — carried from the characterization config; not
separately ruled]`.

**Grading rule — the amended rule (subject-echo guard).** What follows is a
**normative restatement; the executable rule is quoted verbatim in
`bitcliff/pipeline/GRADER_CHARACTERIZATION.md` §7.1** and implemented in
`suites/factual_qa.py:grade` — on any divergence between this prose and that
code listing, the code listing governs. Correct iff any alias from the item's `possible_answers` list occurs in the
completion under: lowercase both sides, collapse whitespace, and require the
alias as a **word-boundary-anchored** substring
(`(?<!\w)alias(?!\w)` after normalization) — no fuzzy matching, no partial
credit, no grader model; **plus the subject-echo guard**: an alias that
already word-boundary-matches inside the *question text itself* is dropped
from the set of aliases eligible to score a hit; if every alias is
disqualified this way, the item grades wrong regardless of output.
Word-boundary anchoring is deliberately stricter than the multivalue2
digit-substring rule — aliases are words, where unanchored matching is
meaningfully lax — and this difference and its reason are registered here.

**Characterization protocol and measured results** (binding ruling: a
30-item characterization runs before the rule freezes;
`bitcliff/pipeline/GRADER_CHARACTERIZATION.md`, run `factualqa-explore`, exploratory,
all accuracy numbers discarded per §12):

- Protocol: 500 items × 3 rungs (F16 / Q4_K_M / Q2_K); per round, 5 graded
  `correct` and 5 graded `wrong` records per rung drawn by fixed seed
  (30 adjudicated items/round); bar: **PASS iff grader-FP = 0/15 and
  grader-FN ≤ 2/15**.
- **Round 1 (original word-boundary rule): FAILED** — FP 1/15. The FP is a
  real, reproducible mechanism: when the question's subject-entity name is
  itself a valid alias for the expected answer (the "Asti" case), an output
  that merely echoes the subject scores correct with a wrong actual answer.
  Ruling: repair the grader, not the dataset (the mechanism would recur in
  TriviaQA too). The subject-echo guard above is the repair; re-grading the
  full run flipped 14 grades across the ladder, four of the five flipped
  items showing the same subject-echo mechanism — independent confirmation
  the guard generalizes.
- **Round 2 (amended rule, fresh adjudication seed): FP 0/15 — the round-1
  mechanism is eliminated — but FN 3/15, exceeding the ≤ 2/15 bar.** The
  three FNs split: two are a pre-existing PopQA alias-list phrasing gap
  ("Roman Catholicism"/"Catholicism" correct but the alias lists carry only
  the "…Church" forms — present under the original rule too, just not sampled
  in round 1); one is the guard's own disclosed collateral (the "Tennis"
  case: the only correct alias is embedded in the entity's own name and is
  therefore disqualified).
- Known systematic bias, disclosed: **1/500 items** in this draw has an
  empty effective-alias set (every alias occurs in its own question) and is
  auto-wrong for every model, capping maximum accuracy at 99.8% on this
  draw. Small, but systematic, not random.

**[DECISION REQUIRED — user selects exactly one branch before
timestamping. The frozen PREREG states the chosen branch and deletes the
others; none is the default.]**

- **Branch (a) — accept the amended rule with the measured FN rate.**
  Freeze the rule as stated above, citing round 2 (FP 0/15, FN 3/15) as its
  measured characterization. The registered argument: alias-gap FNs are
  properties of the item, not the model — the same deficient alias list
  grades both F16 and every quant on the same item, so these FNs are
  **symmetric across paired conditions and absorbed by the paired design**,
  exactly mirroring the multivalue2 substring-symmetry note (§3.1) but in
  the false-negative direction; they bias absolute accuracy down, not the
  paired Δ, except where the quant's answer surface form differs — which the
  published per-item records make auditable. The 1/500 empty-set cap and the
  ~20% sampled FN rate are disclosed verbatim in the methodology page.
  Cost: absolute accuracies understate truth; the suite's paired verdicts
  remain sound under the symmetry argument.
- **Branch (b) — alias-augmentation pass before freeze.** Before the freeze
  commit, run a documented, deterministic augmentation over the alias lists
  of the sampled items only (e.g., add "-ism"/"-ity" religion-name forms and
  other rule-generated variants; every added alias listed in a committed
  manifest), then re-characterize (round 3, fresh seed, same 30-item
  protocol, same bar) and freeze only on a PASS. Cost: delays the freeze by
  one characterization cycle; the augmentation rule itself must be frozen
  with this document so it cannot be tuned against results.
- **Branch (c) — TriviaQA fallback.** Switch the suite to
  `mandarjoshi/trivia_qa` (unfiltered, no-context), alias lists included,
  same sampling structure (popularity stratification replaced by a
  registered difficulty knob for that dataset), subject to: a fresh license
  audit (the HF tag is `unknown`), and a fresh 30-item characterization
  under the same bar. Note the recorded caution: the subject-echo mechanism
  is a property of aliasing generally, not of PopQA specifically — the guard
  (and its collateral) likely travels with the switch.

**Q4 linkage** (whatever branch is chosen): this suite's degradation curve
is mechanism (i) *observed*; Q4's discussion cites it as the measured
reference for how memorized content degrades, while the GSM8K-twin test
remains the two-sided contamination instrument for arithmetic.

---

## 4. Models, ladders, and pinned files

Three models: Qwen2.5-1.5B-Instruct (spectacle; the only live-prompt model),
and the reference pair Llama-3.1-8B-Instruct + Qwen2.5-7B-Instruct
(precompute-only). A quant level is a **file, not a label**: every result is
pinned to uploader, imatrix status, and sha256. The registered ladders are
pinned to the committed manifests (`bitcliff/pipeline/reference-manifests/`),
enumerated live from HF and hashed on 2026-08-26:

| manifest | repo | revision sha | files |
|---|---|---|---|
| `llama-3.1-8b-bartowski.json` | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | `bf5b95e96dac0462e2a09145ec66cae9a3f12067` | 24 |
| `qwen2.5-7b-bartowski.json` | `bartowski/Qwen2.5-7B-Instruct-GGUF` | `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031` | 24 |
| `qwen2.5-7b-official.json` | `Qwen/Qwen2.5-7B-Instruct-GGUF` | `bb5d59e06d9551d752d08b292a50eb208b07ab1f` | 19 |
| `shootout-8b.json` | unsloth `600b0020…`, mradermacher static `2dc24684…`, mradermacher i1 `15a6b2be…` | (three repos) | 2 each |

Per-file sha256 values are inside the manifests and are not restated here.
When an uploader re-quantizes, pages get a "newer file exists" badge; results
never silently point at stale data.

**In-house spectacle rungs.** The 1.5B ladder's sub-2-bit bottom
(IQ1_S / IQ1_M / IQ2_XXS) was quantized in-house because no public repo
carried them (provenance, build commands, calibration corpus and all file
hashes: `bitcliff/pipeline/INHOUSE_QUANTS.md`; llama.cpp commit
`bf942164697d2d62c2237a17b677dc2c017ea8e7`). They are `spectacle_only: true`
and `uploader: bitcliff-inhouse`, and they **never appear in reference
tables, cliff badges, durability comparisons, or download recommendations**
— enforced in the reporting code, not by convention.

---

## 5. Uploader shootout — registered scope

Two arms, each with its own scope and manifest — registered exactly:

- **Arm 1 — uploader shootout: Llama-3.1-8B-Instruct only, at Q4_K_M and
  Q3_K_M only** (binding ruling 2026-08-26). unsloth vs bartowski vs
  mradermacher, with mradermacher's paired static and imatrix ("i1") files
  as the calibration isolation. Files pinned in
  `bitcliff/pipeline/reference-manifests/shootout-8b.json` (bartowski's
  Q4_K_M/Q3_K_M come from its ladder manifest,
  `llama-3.1-8b-bartowski.json`).
- **Arm 2 — official vs community: a separate registered arm on
  Qwen2.5-7B-Instruct**, comparing Qwen's official GGUFs
  (`qwen2.5-7b-official.json`) against bartowski's
  (`qwen2.5-7b-bartowski.json`) at the same quant labels: **Q4_K_M and
  Q3_K_M** `[USER-EDITABLE — proposed for symmetry with Arm 1]`. Meta ships
  no official GGUFs, so this arm runs on the Qwen side only.

**Registered follow-up trigger (not scope creep):** if any same-label
comparison in either arm shows a visible effect — operationally, a pair of
same-label files whose paired-difference CI (per §8's machinery, quant vs
quant on identical items) **excludes 0** — then extending the uploader
shootout to Qwen2.5-7B-Instruct becomes a **registered follow-up
measurement** under the same rules, run after the launch analyses.
`[USER-EDITABLE — the user may widen this trigger (e.g., to include
differing §8 cell states against F16); the CI-only form is registered
because cell-state differences can fire on power differences alone.]`
Absent the trigger, no 7B uploader shootout runs.

---

## 6. Fixed generation parameters and determinism scope

- **Global length budget: `max_tokens: 1024`** — registered, does not churn.
  (Pilot evidence it is generous: at 640 the F16 model clipped exactly 1/90
  outputs, an unscored spectacle item; both scored suites had zero F16
  truncations. `bitcliff/pipeline/PILOT_NOTES.md`.)
- **Per-suite answer budgets** within the global cap: `longctx_retrieval` 32
  tokens (paper-equivalent, §3.1); `factual_qa` 64 tokens
  `[USER-EDITABLE — see §3.4]`; arithmetic suites use the global budget.
- **Deterministic decoding:** greedy, `temperature 0.0`, `top_k 1`,
  `seed 42`.
- **Determinism scope, stated honestly:** determinism holds within a fixed
  hardware + software configuration, not across arbitrary machines. Every
  output record carries the machine and library fingerprint
  (`platform.platform()` / architecture / `llama-cpp-python` version), as the
  pipeline already does. The curated playground prompts are regenerated on
  the exact serving machine.
- **Truncation is its own verdict:** an answer that never arrives within the
  budget counts as wrong (an endless loop is damage), but truncation is
  tracked and reported separately per cell, so "wrong" never silently absorbs
  "cut off."

---

## 7. Difficulty calibration — the registered rule

This registration freezes the **rule**, not pre-baked per-model values —
mirroring the paper's own pre-registered calibration, and keeping the
ordering honest: no GPU spend before the freeze commit; per-model calibration
happens after the rule is frozen but before any confirmatory quant data.

**The rule:** per model, the scored-suite configuration is the hardest
setting whose **F16 accuracy lies in [0.6, 0.85]**, chosen using the
documented knobs only, measured **F16-only, before any confirmatory quant
runs**. "Hardest" is decidable because each knob carries a registered total
order:

- `longctx_retrieval`: the knobs are the **variant** and **`target_tokens`**;
  the depth cycle is fixed at (0.1, 0.5, 0.9). The variant order is the
  paper's own calibrated ladder
  (`bitcliff/pipeline/vendor/bundle-docs/TASK.md`; the order is registered
  here in full, so this document is self-contained), a total order from
  easiest to hardest:
  `single < multikey4 < multikey8 < multikey12 < multivalue2 < multiquery2 <
  multiquery3 < multivalue3 < multiquery4 < multivalue4`.
  The search runs at `target_tokens` 4096 first, then 8192; **"hardest" =
  the variant furthest along that ladder with F16 in-band, ties broken
  toward the larger `target_tokens`.**
- `factual_qa`: the knob is the **popularity mix**, a weight vector over the
  10 popularity deciles (decile 1 = most popular, decile 10 = least
  popular by `s_pop`), chosen from this registered candidate set only —
  written out, most tail-heavy first:
  - **M1 — uniform** (most tail-heavy):
    (0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10).
  - **M2 — linear-tail-heavy** (popular-skewed, tail mass tapering
    linearly), w_i = (11−i)/55:
    (0.182, 0.164, 0.145, 0.127, 0.109, 0.091, 0.073, 0.055, 0.036, 0.018).
  - **M3 — step-tail-heavy** (popular-weighted with a fixed 20% tail
    floor): (0.16, 0.16, 0.16, 0.16, 0.16, 0.04, 0.04, 0.04, 0.04, 0.04).

  **Mix→counts mapping (registered):** per-decile draw counts are the
  **largest-remainder apportionment** of weight_i × n — floor each raw
  count, then hand the shortfall to the deciles with the largest fractional
  remainders, **ties broken toward the lower decile index** — after which
  the seeded per-decile draw proceeds in decile order exactly as in the
  uniform case (implemented as the optional `weights` argument of
  `suites/factual_qa.py::items_from_records`; the uniform M1 mix reproduces
  §3.4's remainders-to-the-earliest-deciles rule exactly).

  **"Hardest" = the most tail-heavy candidate with F16 in-band**, in the
  order M1 > M2 > M3. If no candidate lands in-band, the suite runs at M3
  and the out-of-band F16 value is disclosed in the amendment. Exploratory
  observation carried forward: the even 10-decile draw put the 1.5B F16 at
  **0.120** under the original grader
  (`bitcliff/pipeline/GRADER_CHARACTERIZATION.md` §2) and **0.110** under
  the amended grader (ibid. §7.2 re-grade), so the knob must move a fair
  distance toward the popular end.
- `arithmetic` / `arithmetic_twins`: **none** — GSM8K as-is.

**Amendment mechanics and scope:** the chosen per-model knob settings —
and only those — plus the resulting item hashes are appended to this
document as a **dated amendment**
(`[AMENDMENT SLOT — difficulty calibration, appended and stamped at F4+]`),
committed and OpenTimestamps-stamped like the original. **What an amendment
may record: the `longctx_retrieval` variant and `target_tokens` per model,
and the `factual_qa` mix candidate per model. What is fixed now and may NOT
be amended: every sampling seed (§3.2 arithmetic seed, §3.4 factual_qa
confirmatory seed, twins seed 1301, multivalue2 seed 2024 for 2a), every n,
the candidate sets and total orders above, and every registered rule.**
Calibration is disclosed as pilot-legitimate (§12).

---

## 8. Statistics: margin, per-cell inference, cell states, multiplicity

- **Margin: M = 3 percentage points absolute accuracy, per suite.**
  `[USER-EDITABLE — 3pp is the ruling of 2026-08-26; the user may set a
  different value per suite during the edit pass, before timestamping.]`
- **Per-cell inference** (quant vs F16, paired on identical items):
  McNemar exact test on the accuracy difference; two-sided 95% CI on
  Δaccuracy by paired bootstrap (10,000 resamples).
- **The four cell states** (every matrix cell is exactly one):
  1. **Damaged** — CI excludes 0 and the point estimate exceeds M loss.
  2. **Small real loss** — CI excludes 0 and the point estimate is within M;
     the signed loss is shown (e.g. "−1.2pp").
  3. **Equivalent** — the entire CI lies within ±M (**equivalence by
     CI-inclusion**, the TOST-equivalent rule). Never claimed from mere
     absence of evidence.
  4. **Indeterminate** — none of the above; never rounded up to "free."
- **The cliff:** per capability, the highest-precision file whose cell is
  Damaged. Non-monotonic rungs are flagged, never smoothed (with the
  IQ-vs-K ~2.5 bpw note where applicable).
- **Multiplicity — the dual rule, registered here, not post hoc:**
  1. **Per-cell verdicts are descriptive, at fixed α = 0.05, with no
     multiplicity correction across cells.** The matrix is a map, and every
     cell shows its own evidence.
  2. **Any headline claim that aggregates across cells** — a cliff position,
     a cross-suite asymmetry, a durability comparison, any sentence of the
     form "capability X is damaged from rung Y down" — **is stated as a
     claim only if it survives Holm–Bonferroni correction over the family of
     per-cell tests it aggregates across.** A headline that fails Holm is
     reported as descriptive-only, with that label.
  This dual rule is part of the registration; readers should hold the
  project to it.
- **Truncation** reported per cell alongside states (§6).
- **Both flip directions always shown:** items F16 got right that the quant
  lost, and items the quant randomly gained.

---

## 9. Durability comparison

The two reference models' retention curves overlaid per capability, each
normalized to its own F16 baseline (removing tokenizer, template, and
starting-accuracy differences), cliff positions compared directly. The claim
is class-level ("7–8B"); the residual size gap is disclosed; the 1.5B appears
nowhere on that page. Cross-model comparisons obey the task-level-only rule
(§3.1). Headline durability claims fall under the Holm side of the dual rule
(§8).

---

## 10. Blind-check protocol (guess-the-quant gate)

Registered before any rater sees anything:

- **Materials:** mid-ladder only — Q6_K / Q5_K_M / Q4_K_M / Q3_K_M — outputs
  from the confirmatory 1.5B run. The materials file's hash is recorded
  before the first rating session
  `[TO BE FILLED before 0C — materials sha256]`.
- **Round design (registered):** 10 rounds. The true label of each round is
  drawn **uniformly with replacement** from the 4 mid-ladder rungs, using a
  fixed seed chosen and published **after the materials are frozen** and
  before any rater sees anything
  `[TO BE FILLED before 0C — label-sequence seed]`.
- **Raters:** two raters who have never seen any outputs. The pipeline
  operator is not blind and never rates.
- **Scoring:** distance scoring — full credit exact, partial credit
  adjacent, otherwise nothing (the top of the ladder is genuinely
  indistinguishable; exact-only scoring would make the game feel broken).
- **Null and test (registered):** a round counts as an adjacent-inclusive
  success if the rater's guess is the true rung or an adjacent rung. Under
  the null of uninformed uniform guessing, the per-round success probability
  is determined by the round's true label: **2/4 for the end rungs (Q6_K,
  Q3_K_M), 3/4 for the middle rungs (Q5_K_M, Q4_K_M)**. The test is an
  **exact binomial test against p₀ = the mean of those per-round null
  probabilities over the realized 10-label sequence**, one-sided
  (greater), **p < 0.05, per rater**.
- **Pass rule:** **both** raters pass their individual test.
  `[USER-EDITABLE — proposed threshold]`
- **Fallback:** if a second naive rater cannot be found, or the gate fails,
  **the game stays out** until a naive rater passes it.

---

## 11. Dataset scope, embargoes, and the reconstruction recipe

Published per-item records (outputs, grades, divergence, fragility signals)
for all suites, with these registered exclusions:

- **multivalue2 prompts are excluded from publication in both
  configurations.** Shipped instead: the generator (Apache-2.0, §3.1), the
  seeds, the corpus pointer + pinned hash (Gutenberg PG #1184 hashes for 2b,
  §3.1; the 2a corpus pointer documented but its text never shipped),
  per-item metadata (item id, key, depths, `n_answer_tokens`, config), model
  outputs, and a **reconstruction recipe**: tokenizer + generator + seed +
  corpus hash → bit-identical items, verified by the digest mechanism the
  bundle provides (demonstrated: `bitcliff/pipeline/CORPUS_MANIFEST.md` §3).
- **Twins embargoed** per §3.3 — hashes public, instances withheld until the
  paper extension ships.
- **Closed-book QA items are public-source and published normally.**
- Spectacle-rung outputs are published as spectacle, never as reference data.

---

## 12. Exploratory-pilot declaration

Declared here so nobody discovers it themselves:

- **Run `pilot-0a`** (2026-08-26; 8 published rungs + the three in-house
  sub-2-bit rungs, 90 items/rung) was **exploratory**. It informed the
  length budget, curation choices, and the decision to build in-house
  spectacle rungs. **Every number from it is discarded**, including the
  placeholder suite that preceded `longctx_retrieval` (deleted, no
  conclusions survive) and the IQ2_M > Q2_K arithmetic observation (kept
  only as Q5's exploratory candidate, §2).
- **Run `factualqa-explore`** (2026-08-26; 500 PopQA items × 3 rungs, both
  grading rounds) was **exploratory**. Its accuracy/retention numbers are
  discarded. Only two things survive: the grader FP/FN characterization
  (§3.4) and the F16 difficulty-band observation feeding the popularity-mix
  knob (§7).
- **Difficulty calibration (§7) is disclosed as pilot-legitimate:** F16-only
  runs, executed after the freeze under the frozen rule, appended as dated
  amendments.

---

## 13. Citation hygiene (binding rule)

Where paper-repo numbers are cited, **only the n=96 figures are quoted** for
whole-model 4-bit long-context retrieval — **dz +0.221 (nf4 @ 4-bit,
n=96)** — with the paper's own companions as the abstract states them:
int_group @ 4-bit dz +0.205 (n=48) and 3-bit dz +1.876 (n=48, the ~3.2×
asymmetry headline vs arithmetic dz +0.589). **The n=24 calibration figure
(dz +0.395 — stated here once, solely to identify the banned figure) is
selection-inflated (winner's curse; the paper's §4.4) and is
never quoted, anywhere, by anyone on this project.** Also registered:
quoted dz values are metric-bound (teacher-forced answer-span NLL, 10-token
span, ddof=1) and quantizer-bound (`nf4` vs `int_group`, bits, group size
64) — every quoted number carries those qualifiers. Source:
`bitcliff/pipeline/vendor/bundle-docs/PAPER_CONFIG.md` (paper draft at commit
`8071fc44d91b15842c57cbf26a92bdd968b0d522`; nothing is in print).

---

## 14. Scope disclaimers

- Results are **per model and per size**. Nothing here extrapolates across
  model families, sizes, or quantization schemes not measured.
- The in-house spectacle rungs (§4) never enter reference tables, cliffs,
  durability, or recommendations — enforced in code.
- F16 is the reference baseline, never a download recommendation.
- This is not a general evaluation harness and not a model-quality
  leaderboard; retention is within-model by construction.

---

## 15. Freeze and timestamp mechanics

Per binding ruling 2026-08-26 (ruling 4):

1. The user edits this draft; every `[DECISION REQUIRED]` is resolved and
   every `[USER-EDITABLE]` value confirmed or changed.
2. The edited PREREG.md is committed together with the reference manifests,
   the twins templates and verifier (embargoed instances by hash only), and
   the evidence documents cited here — including the six bundle documents
   vendored verbatim into `bitcliff/pipeline/vendor/bundle-docs/` (each
   carries a provenance header naming the source repo, its Apache-2.0
   license, the paper-draft commit `8071fc44d91b15842c57cbf26a92bdd968b0d522`,
   and the 2026-08-26 extraction date), enumerated with their sha256 hashes
   as committed:

   | vendored file | sha256 |
   |---|---|
   | `vendor/bundle-docs/MECHANISM.md` | `b32641b97ea227062a358daf3d9c4d4c34d27213c4f3b2840744d5d367120418` |
   | `vendor/bundle-docs/PAPER_CONFIG.md` | `dfd2c6835dfef5936797343c17d7c68f9d60d10fb2d90b12130ca1f361f0d19e` |
   | `vendor/bundle-docs/GRADING.md` | `c0cd7bb6be0aa9b8e5feff639126958c674cc124c33143ab72fccc1a2fe27879` |
   | `vendor/bundle-docs/ANSWER_TOKENS.md` | `5d7999ec2d29d446b4b752f0298cbb0ea92125424d28310e92a1e243b671a638` |
   | `vendor/bundle-docs/TASK.md` | `33a0001ce8ea58f71a014e615d21ba4c4a33ffd3ca573dca4634a80084c04886` |
   | `vendor/bundle-docs/PROVENANCE.md` | `04bc3682e64299bb2516f44ed450e1c9c397923ea095646c98dfc20c7d04c25f` |

   **This is the freeze commit.** No
   confirmatory run exists before it and no GPU money is spent before it.
3. The freeze commit hash is stamped with **OpenTimestamps** (free,
   Bitcoin-anchored, verifiable offline); the `.ots` proof is committed once
   the anchor confirms.
4. The freeze commit hash is **cross-recorded in
   `bitcliff/pipeline/PILOT_NOTES.md`** ("Freeze cross-reference" section),
   so the exploratory record and the registration point at each other.
5. Amendments (the §7 calibration values, any registered follow-up trigger
   firing per §5) are appended as dated sections, committed, and stamped the
   same way. Amendments fill registered slots; they never alter registered
   rules.

PREREG commit: `[TO BE FILLED AT F4]`
OpenTimestamps proof: `[TO BE FILLED AT F4]`
