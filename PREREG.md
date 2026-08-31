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
`[USER-EDITABLE — confirmed by user 2026-08-26]`, sampled with fixed seed 3141
`[USER-EDITABLE — confirmed by user 2026-08-26; fresh, distinct from every
exploratory seed (pilot arithmetic 1301, twins 1301, factual_qa
characterization 7411)]`.**
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

**Sampling:** n=500 `[USER-EDITABLE — confirmed by user 2026-08-26]`,
**confirmatory fixed seed 2718** `[USER-EDITABLE — confirmed by user
2026-08-26; deliberately distinct from the characterization run's seed
7411]`, **stratified by
subject-entity popularity decile** (`s_pop`, Wikipedia monthly pageviews):
records sorted by popularity, split into 10 contiguous deciles, draws per
decile per the registered mix with remainders to the earliest deciles
(`suites/factual_qa.py`). **The popularity MIX is the only calibration knob
(§7, long-tail facts fail first); the seed never floats** — it is fixed by
this registration and is outside the §7 amendment scope.

**Prompt** (registered): `Answer with just the answer: {question}` —
closed-book, no context. Per-suite answer budget: 64 tokens within the
global cap `[USER-EDITABLE — confirmed by user 2026-08-26; carried from the
characterization config]`.

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

**Registered resolution (user decision, 2026-08-26): branch (b) —
alias-augmentation pass before freeze.** Before the freeze commit, the
alias lists of the 500 sampled items received a documented, mechanical,
output-blind augmentation pass; a round-3 re-characterization on a fresh
sample follows as a separate, later step.

**The augmentation rule (verbatim, as implemented):** English label +
English aliases of the object entity, fetched for every sampled record,
output-blind. For each of the 500 records sampled by
`factual_qa.picked_records(seed=7411)`, the object entity's Wikidata QID is
derived from the record's `o_uri` field (**not** `obj_id` — `obj_id` is an
internal PopQA numeric identifier and does not correspond to the Wikidata
QID; verified against `datasets-server`, where a record with
`obj_id=2834605` carries `o_uri` encoding `Q82955` — the numbers do not
correspond), and the Wikidata API (`wbgetentities`, `props=aliases|labels`,
`languages=en`, `<=50` ids/call, `bitcliff/pipeline/scripts/
fetch_wikidata_aliases.py`) is queried for that QID's English label and
English aliases. Every value the API returns becomes an eligible
augmentation alias for that record — no per-item selection or filtering by
hand, and the mapping was built and committed **before any model output was
read**. Mapping file: `bitcliff/pipeline/data/
popqa_wikidata_aliases_seed7411.json` (committed, registered evidence),
header records retrieval date 2026-08-26, the rule text above, the query
endpoint, and counts. At grade time, `factual_qa.items_from_records`'s
`alias_augmentation` parameter unions each item's original
`possible_answers` with that record's mapping entries — deduped
case-insensitively, original aliases and their order preserved, new
aliases appended in the order given — and the subject-echo guard above then
applies to the augmented list unchanged (no grader-logic change was
needed). Measured coverage on this draw: 500 sampled records -> 437 unique
object QIDs, all 437 resolved (0 missing), 1264 aliases fetched total
(mean 2.89/QID); 177/500 sampled items received at least one alias not
already present in `possible_answers`.

**Round-3 protocol (registered):** a fresh 30-item sample — 15 items graded
`correct` and 15 graded `wrong` under the augmented alias lists, drawn
across the three rungs (F16/Q4_K_M/Q2_K), by a **new** fixed seed distinct
from every seed used in rounds 1–2 — independently adjudicated by the same
method as rounds 1 and 2. **Bar (as the user set it 2026-08-26): PASS iff
grader-FP <= 1/15 AND grader-FN <= 2/15** (the FP side loosens from the
round-1/2 bar of 0/15; the FN side is unchanged). **Registered consequence,
without further debate: if round 3 fails this bar on the fresh sample, the
TriviaQA fallback (former branch (c), kept below) executes** — the suite
switches to `mandarjoshi/trivia_qa` per its documented protocol, subject to
its own fresh license audit and a fresh 30-item characterization under the
same bar. Round 3's result, its seed, and (if triggered) the fallback are
recorded as a dated amendment to this section when that step completes.
Round 3 result: fresh 30-item sample, seed `20260828`, FP 0/15, FN 0/15 —
PASS; recorded as dated amendment 2026-08-26 (full table:
`bitcliff/pipeline/GRADER_CHARACTERIZATION.md` §9). The TriviaQA fallback
above does not execute.

**Rejected alternatives, kept as a short record:**

- **Former branch (a) — accept the amended rule with the measured FN rate
  (round 2: FP 0/15, FN 3/15), no augmentation.** Rejected: the FN rate
  (3/15) was left unaddressed by this branch; the user preferred to repair
  the alias-list gap directly (branch (b), above) rather than rely solely
  on the paired-design symmetry argument to absorb it.
- **Former branch (c) — TriviaQA fallback, as an immediate switch.**
  Rejected as an immediate action, and instead registered above as the
  round-3 failure consequence: the subject-echo mechanism is a property of
  aliasing generally, not of PopQA specifically, so an immediate switch
  would not by itself have closed the FN gap; PopQA's license is already
  cleared (MIT via the canonical release, `LICENSE_AUDIT.md` §1) while
  TriviaQA's is not (HF tag `unknown`, fresh audit required before use).

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
  Q3_K_M** `[USER-EDITABLE — confirmed by user 2026-08-26; symmetry with
  Arm 1]`. Meta ships
  no official GGUFs, so this arm runs on the Qwen side only.

**Registered follow-up trigger (not scope creep) — widened per user
decision 2026-08-26:** if any same-label comparison in either arm shows a
visible effect — operationally, **either** (i) a pair of same-label files
whose paired-difference CI (per §8's machinery, quant vs quant on
identical items) **excludes 0**, **or** (ii) same-label pairs landing in
different §8 cell states (e.g. one file Equivalent against F16, its
same-label counterpart Damaged or Small real loss) — then extending the
uploader shootout to Qwen2.5-7B-Instruct becomes a **registered follow-up
measurement** under the same rules, run after the launch analyses. Widened
back to this original two-arm form (CI-excludes-0 OR differing cell
states) per user decision 2026-08-26 — over-running a cheap shootout is
preferred to under-detecting an uploader effect. Absent the trigger, no 7B
uploader shootout runs.

---

## 6. Fixed generation parameters and determinism scope

- **Global length budget: `max_tokens: 1024`** — registered, does not churn.
  (Pilot evidence it is generous: at 640 the F16 model clipped exactly 1/90
  outputs, an unscored spectacle item; both scored suites had zero F16
  truncations. `bitcliff/pipeline/PILOT_NOTES.md`.)
- **Per-suite answer budgets** within the global cap: `longctx_retrieval` 32
  tokens (paper-equivalent, §3.1); `factual_qa` 64 tokens
  `[USER-EDITABLE — confirmed by user 2026-08-26; see §3.4]`; arithmetic
  suites use the global budget.
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
(`[AMENDMENT SLOT — difficulty calibration]` — **filled by Amendment 1, appended at the end of this document, 2026-08-29**),
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
  `[USER-EDITABLE — confirmed by user 2026-08-26; the user may still set a
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
- **Pass rule:** **both** raters pass their individual test (each at
  `p < 0.05`, above).
  `[USER-EDITABLE — confirmed by user 2026-08-26]`
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
  grading rounds, and the augmented-alias re-grade below) was
  **exploratory**. Its accuracy/retention numbers are discarded. Only
  things surviving: the grader FP/FN characterization (§3.4), the F16
  difficulty-band observation feeding the popularity-mix knob (§7), and the
  disclosure below.
- **Grader characterization disclosure (§3.4):** both characterization
  rounds run against `factualqa-explore` **failed their bar**. Round 1
  (original word-boundary rule) failed on **FP 1/15** (the subject-echo
  mechanism — an output that merely echoes the question's subject-entity
  name scores correct with a wrong actual answer). Round 2 (rule amended
  with the subject-echo guard, fresh adjudication seed) fixed the FP failure
  (**FP 0/15**) but failed on **FN 3/15**, exceeding the <=2/15 bar (a
  pre-existing PopQA alias-list phrasing gap plus one instance of the
  guard's own disclosed collateral). **In response, two repairs were
  adopted:** the subject-echo guard (grading-rule change, §3.4) and the
  mechanical Wikidata alias augmentation (branch (b), §3.4) — built and
  committed output-blind, before any model output was read for this task.
  **Round 3** — a fresh 30-item sample, new seed, same adjudication
  method, bar FP <= 1/15 AND FN <= 2/15 — runs on the augmented alias lists
  as a separate, later step; its outcome (and, if it fails, the TriviaQA
  fallback it triggers) is appended here as a dated amendment.
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

PREREG commit: `5e6882b7a10c5e4670052855380e8646911db5f3` (the no-ff merge of `freeze-prep` into `main`; filled by the follow-up commit, since the freeze commit's hash cannot appear inside itself)
OpenTimestamps proof: `freeze/freeze-commit-hash.txt.ots` (receipt over `freeze/freeze-commit-hash.txt` containing `5e6882b7a10c5e4670052855380e8646911db5f3`; Bitcoin-attested: block headers 964240/964307, upgraded receipt committed 2026-08-28, commit `0b4658a`)


---

# Amendment 1 (2026-08-29): difficulty calibration

Appended per §7's amendment mechanics and §15.5, after user review of the
standalone draft (`AMENDMENT_DRAFT.md`, commit `35c3228`) and explicit user
approval ("stamp it", 2026-08-29). Committed and OpenTimestamps-stamped like
the original; this amendment's own commit hash and receipt are recorded by
the follow-up commit (the hash cannot appear inside its own commit):

Amendment 1 commit: `00a1228ca6df39ddb5971e87920d4d0b78913cf5` (filled by the follow-up commit)
Amendment 1 OpenTimestamps proof: `freeze/amendment1-commit-hash.txt.ots` (receipt over `freeze/amendment1-commit-hash.txt` containing `00a1228ca6df39ddb5971e87920d4d0b78913cf5`; Bitcoin-attested: block headers 964530/964545/964549, upgraded receipt committed 2026-08-29, commit `62243b2`)

Produced by: `bitcliff/pipeline/scripts/calibrate_f16.py` (task 1a, this
overnight session). Full measurement provenance:
`bitcliff/pipeline/calibration/<model-id>/measurements.jsonl` (every
individual measurement: knob values, n, seed, accuracy, item-set sha256,
wall time) and `summary.json` (the derived chosen/terminal-state summary)
per model, all committed.

---

## A. Registration-gap fix: 2b `longctx_retrieval` item-set n and seed (dated 2026-08-27)

**What PREREG omitted.** §3.1 registers config 2b's corpus (PG-1184,
verified sha256), knobs (variant, `target_tokens`), the fixed depth
cycle, and the grading rule — but it never states an item count (`n`) or
a sampling seed for 2b, unlike every other suite (§3.2 arithmetic:
n=500/seed=3141; §3.3 twins: seed=1301; §3.4 factual_qa:
n=500/seed=2718). §7 fixes "every n" and "every sampling seed" as
non-amendable, which only compounds the gap: there was no registered
value to hold fixed for 2b. This was logged as `OPEN_QUESTIONS.md` §2
before any 2b calibration ran.

**What was done, and when.** Calibration needed concrete values to
produce curves at all, so it ran at **n=96, seed=2024** — the same
values §3.1 config 2a already registers for the paper-comparability run
— labeling every measurement "provisional" pending this decision. On
**2026-08-27**, by user ruling (relayed during this session), **n=96 and
seed=2024 are REGISTERED for 2b**, mirroring 2a. This is a disclosed gap
fix, not a §7 knob-setting amendment: §7's amendment scope covers only
"the `longctx_retrieval` variant and `target_tokens` per model, and the
`factual_qa` mix candidate per model" — n and seed are explicitly
non-amendable there. Registering a value PREREG never stated in the
first place is a different, and larger, kind of change, disclosed here
in its own right rather than folded silently into the per-model knob
amendment below.

**Consequence.** Every longctx measurement this session produced under
n=96/seed=2024 is the real calibration data, not provisional. Item-set
hashes below are therefore unblocked and included.

**Correction on the `provisional_note` field.** An earlier draft of this
section claimed that field's presence in the raw `measurements.jsonl`
records was accurate provenance of *when* each line was written relative
to this ruling. That claim is **false** and is retracted here: the note
is a static string `scripts/calibrate_f16.py` writes unconditionally into
every longctx record, with no time-awareness of this resolution at all.
It is temporally accurate only for the qwen2.5-1.5b-instruct measurements
and the qwen2.5-7b-instruct measurements taken before this ruling landed;
llama-3.1-8b-instruct's entire longctx set (timestamped 2026-08-28) and
part of qwen2.5-7b-instruct's t=8192 batch **postdate** the 2026-08-27
ruling yet still carry the stale "provisional (pending 2b n/seed
registration...)" text, because the field was never rewritten after the
ruling. It is retained as-written (not edited after the fact) rather than
corrected, but it must **not** be read as reliable real-time provenance —
each record's own `timestamp` field is the reliable source for when a
measurement actually ran.

---

## A2. Registration-gap fix: `longctx_retrieval` out-of-band-high fallback (dated 2026-08-28)

**What §7 omitted.** §7 registers a fallback for `factual_qa`'s
out-of-band case ("If no candidate lands in-band, the suite runs at M3
and the out-of-band F16 value is disclosed in the amendment") but
registers **none** for `longctx_retrieval`. When qwen2.5-7b-instruct's
and llama-3.1-8b-instruct's calibration runs found every setting in the
registered candidate space (variant ladder × `target_tokens` ∈ {4096,
8192}) scoring above the [0.6, 0.85] band — including the hardest
registered setting — the harness had no registered rule to apply and
this was logged as a genuine gap, `OPEN_QUESTIONS.md` §5.

**What was done, and when.** On **2026-08-28**, by user ruling, option
(a) from §5 was chosen: **the registered rule (stated verbatim, as
ruled)** is —

> "when no longctx setting in the registered candidate space lands in
> [0.6, 0.85], the suite runs at the hardest registered setting
> (multivalue4 @ target_tokens 8192) and the out-of-band F16 accuracy is
> disclosed — the exact analogue of the factual_qa M3 rule."

This is a disclosed gap fix, not a §7 knob-setting amendment, for the
same reason as §A above: §7's amendment scope covers only the per-model
knob *settings* chosen from an already-registered fallback; here no
fallback existed to choose from; registering the fallback rule itself is
a different, and larger, kind of change, disclosed here in its own
right.

**Consequence, disclosed:** equivalence bounds near a ~1.0 baseline are
weaker — longctx cells for these models can resolve Damaged vs not, but
tight equivalence claims may be limited.

**Alternative considered and rejected.** Extending the candidate space
(e.g. `target_tokens` 16384) was considered and **REJECTED** by the user
ruling — the registered candidate set stands.

Applied in §C below: qwen2.5-7b-instruct and llama-3.1-8b-instruct both
now have a chosen longctx setting (multivalue4 @ 8192) under this rule,
with their disclosed out-of-band F16 accuracies.

---

## B. Disclosure: factual_qa alias-augmentation mapping widened to the union of M1/M2/M3 (seed 2718)

PREREG §3.4 branch (b)'s augmentation rule ("English label + English
aliases of the object entity, fetched for every sampled record,
output-blind") was executed, for the characterization seed (7411),
against a single draw (`picked_records(seed=7411)`, uniform weights —
the only mix that existed at that time).

For the confirmatory seed (2718), building the calibration harness
surfaced that a single-draw mapping is not sufficient once the §7 mix
knob is in play: `picked_records` draws each of the ten popularity
deciles with `rng.sample(decile, take)` against ONE shared,
sequentially-advancing `random.Random(seed)`; the three registered mixes
(M1/M2/M3) apportion a different `take` per decile, which shifts how
much of that shared RNG stream each decile consumes and therefore which
SPECIFIC records every later decile draws too — not merely their
proportions. At seed=2718, n=500, M1/M2/M3 each draw 500 records with
444/450/444 unique object QIDs respectively — only partial overlap
between mixes, not the same set redistributed. These counts are
**independently reproducible**, not merely asserted here: run
`uv run python scripts/calibrate_f16.py --print-mix-overlap` (no model
needed, network access to load `akariasai/PopQA` only) — it calls
`mix_qid_sets`/`mix_overlap_report` (pure, deterministic given the
dataset) and prints exactly:
```json
{
  "sizes": {"M1": 444, "M2": 450, "M3": 444},
  "pairwise_overlap": {"M1&M2": 120, "M1&M3": 142, "M2&M3": 347},
  "union_size": 841
}
```

**Resolution (approved by user ruling during this session):**
`scripts/calibrate_f16.py::ensure_alias_mapping` unions
`picked_records(seed=2718, weights=w)`'s object QIDs over all three
registered mixes before fetching — the identical mechanical,
output-blind, no-per-item-judgment rule; only the input QID set is
widened to match what calibration actually grades. Written and committed
**before any factual_qa generation for this seed**, preserving the
output-blind ordering.

Mapping file: `bitcliff/pipeline/data/popqa_wikidata_aliases_seed2718.json`.
Its `_meta` block records: 841 unique object QIDs (union of 444 + 450 +
444 with overlap), 841/841 resolved (0 missing), 2398 aliases fetched
total (mean 2.851/QID), the per-mix QID counts, retrieval date
2026-08-26, and this rationale. Full detail: `OPEN_QUESTIONS.md` §3.

---

## C. Per-model calibration results

All three models measured **F16-only**, before any confirmatory quant
run, per §7. `factual_qa`: n=500, confirmatory seed 2718 (registered,
never amended), 64-token budget, greedy, alias-augmented per §B above.
`longctx_retrieval`: corpus PG-1184 (2b, verified sha256
`0a21a13834b5215876bd4019af8fbc436abbfbb61b2826db62223eb990071443`),
n=96, seed=2024 (registered per §A above), 32-token budget, greedy,
depths fixed at (0.1, 0.5, 0.9). §3.1's tokenizer-match assertion
(GGUF vs. HF tokenizer, over the first 20 items' question strings)
PASSED for every model before its first generation.

### qwen2.5-1.5b-instruct

**factual_qa** — M1 (uniform) 0.114, M2 (linear-tail-heavy) 0.112, M3
(step-tail-heavy) 0.110. None in [0.6, 0.85].
**Chosen mix: M3 — out-of-band, F16 accuracy 0.110, disclosed** per §7's
registered fallback (no candidate in-band → run at M3, disclose the
out-of-band value). Item-set sha256 (M3):
`2e53ca0e73e9cbdd5d7bac672857ba918ff7b74d2405ddf62f37aa9a630091c6`.

**longctx_retrieval** — binary search at t=4096 hit a real monotonicity
violation (multivalue3 0.917 scored higher than the easier multivalue2's
0.833, beyond the ±0.05 noise band) and correctly fell back to measuring
the full ladder:

| variant | t=4096 accuracy |
|---|---|
| single | 1.000 |
| multikey4 | 0.979 |
| multikey8 | 0.990 |
| multikey12 | 0.969 |
| multivalue2 | 0.833 |
| multiquery2 | 0.990 |
| multiquery3 | 0.990 |
| multivalue3 | 0.917 |
| multiquery4 | 0.958 |
| multivalue4 | 0.646 |

Hardest in-band at t=4096: multivalue4 (0.646). Same variant at t=8192:
0.750, still in-band → 8192 wins the tie (§7); no harder variant exists
past multivalue4 on the registered ladder.

**Tie-break interpretation — confirmed by user 2026-08-28:** both t
in-band for multivalue4; t=8192 per the registered tie-break.

**Chosen: variant=multivalue4, target_tokens=8192, F16 accuracy=0.750.**
Item-set sha256: `ed18db130e4a035eef4bd581a8c53c7a17daf63085499d9123497991bd257f9c`.

### qwen2.5-7b-instruct

**factual_qa** — M1 0.140, M2 0.196, M3 0.192. None in [0.6, 0.85].
**Chosen mix: M3 — out-of-band, F16 accuracy 0.192, disclosed.**
Item-set sha256 (M3): `2e53ca0e73e9cbdd5d7bac672857ba918ff7b74d2405ddf62f37aa9a630091c6`
(identical to the 1.5B's — item construction is model-independent).

**longctx_retrieval** — t=4096: multivalue2/multivalue3/multiquery4 all
1.000, multivalue4 (hardest) 0.990 — too easy everywhere, nothing
in-band even at the ladder's end. Per the registered "4096 first, then
8192" search, the full ladder was searched again at t=8192:
multivalue2/multivalue3/multiquery4/multivalue4 all 1.000 — **still
nothing in-band anywhere in the registered candidate space.**

Terminal state at measurement time: `above_band_everywhere`. Per §A2's
registered fallback (user ruling, 2026-08-28), the suite runs at the
hardest registered setting with the out-of-band value disclosed.

**Chosen: variant=multivalue4, target_tokens=8192, F16 accuracy=1.000
(out-of-band-high, disclosed).** Item-set sha256:
`ed18db130e4a035eef4bd581a8c53c7a17daf63085499d9123497991bd257f9c`.

### llama-3.1-8b-instruct

**factual_qa** — M1 0.248, M2 0.330, M3 0.314. None in [0.6, 0.85].
**Chosen mix: M3 — out-of-band, F16 accuracy 0.314, disclosed.**
Item-set sha256 (M3): `2e53ca0e73e9cbdd5d7bac672857ba918ff7b74d2405ddf62f37aa9a630091c6`
(identical to the other two models').

**longctx_retrieval** — t=4096: multivalue2/multivalue3/multivalue4 all
1.000, multiquery4 0.979 — too easy everywhere. Full ladder searched
again at t=8192: multivalue2 0.990, multivalue3 1.000, multiquery4
0.990, multivalue4 (hardest) 0.990 — **still nothing in-band.**

Terminal state at measurement time: `above_band_everywhere` (same as the
7B). Per §A2's registered fallback (user ruling, 2026-08-28), the suite
runs at the hardest registered setting with the out-of-band value
disclosed.

**Chosen: variant=multivalue4, target_tokens=8192, F16 accuracy=0.9896
(out-of-band-high, disclosed).** Item-set sha256:
`9973be4a98d860e84f8002be91a80b2808a371466c0f60d6a2f073e79ad59bff`.

---

## D. Summary table

| model | factual_qa mix | factual_qa F16 acc | in-band? | longctx variant | longctx t | longctx F16 acc | in-band? |
|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b-instruct | M3 | 0.110 | no (disclosed) | multivalue4 | 8192 | 0.750 | **yes** |
| qwen2.5-7b-instruct | M3 | 0.192 | no (disclosed) | multivalue4 (§A2 fallback) | 8192 | 1.000 | no (disclosed, out-of-band-high) |
| llama-3.1-8b-instruct | M3 | 0.314 | no (disclosed) | multivalue4 (§A2 fallback) | 8192 | 0.9896 | no (disclosed, out-of-band-high) |

**Only the 1.5B has a fully in-band configuration for both suites** (and
only for longctx — its own factual_qa also falls back to M3
out-of-band). Both reference models (7B, 8B-class) score above the
[0.6, 0.85] band on every point in the registered longctx candidate
space, and below/near it (never above 0.33) on every registered
factual_qa mix — the opposite pattern from longctx. This asymmetry (one
suite saturates high, the other stays low, for the same larger models)
is itself worth carrying into the methodology writeup. Per §A2,
equivalence bounds near a ~1.0 baseline are weaker for the 7B's and
Llama-8B's longctx cells — they can resolve Damaged vs not, but tight
equivalence claims may be limited.

---

## E. What this amendment does NOT do

- Does not modify PREREG.md.
- Is not OpenTimestamps-stamped.
- Does not extend the registered longctx candidate set — a
  `target_tokens` 16384 (or similar) extension was considered and
  REJECTED by the user ruling that resolved `OPEN_QUESTIONS.md` §5 (§A2
  above); the registered candidate set (the fixed variant ladder ×
  `target_tokens` ∈ {4096, 8192}) stands unchanged.
- Does not change any registered seed, n, candidate set, or total order
  (§A and §A2 above are gap-fills of a value/rule PREREG never stated,
  not changes to a registered one).

---

# Amendment 2 (2026-08-30): the registered reference ladder

Appended per the disclosed-gap mechanics of Amendment 1 §A/§A2 (a
registration-gap fix beyond §7's knob-amendment scope, disclosed in its
own right), after user review of the standalone draft
(`AMENDMENT2_DRAFT.md`) and explicit user approval ("stamp it",
2026-08-31). Committed and OpenTimestamps-stamped like the original; this
amendment's own commit hash and receipt are recorded by the follow-up
commit:

Amendment 2 commit: `22fbaba04694b3b3ef9701c00802540208ed18f2` (filled by the follow-up commit)
Amendment 2 OpenTimestamps proof: `freeze/amendment2-commit-hash.txt.ots` (receipt over `freeze/amendment2-commit-hash.txt` containing `22fbaba04694b3b3ef9701c00802540208ed18f2`; submitted to 4 calendar servers 2026-08-31, pending Bitcoin attestation — run `ots upgrade` after ~a day and commit the upgraded receipt)

## A. What PREREG omitted

§4 registers the reference ladders as "pinned to the committed manifests
(`bitcliff/pipeline/reference-manifests/`)" — provenance pinning, file by
file, sha256 by sha256 — but never enumerates WHICH of each manifest's 24
files constitute the confirmatory ladder. The manifests carry everything
the uploader ships (including _S/_L/_XL size variants, the Q4_0 family,
ARM-repacked files, and full-precision conversions); no registered
sentence selects the run matrix from them. The ladder is the
row-definition of every reference table, so this gap must be closed
before any confirmatory generation. Logged as `OPEN_QUESTIONS.md` §6
before any confirmatory run; resolved by user ruling 2026-08-30.

## B. Sources the ruling carries into registered text

The design doc as of the freeze (`claude/IDEA-v1.md` §4 — at
`claude/IDEA.md` until 2026-08-30, preserved verbatim at the v1 path;
unregistered until now) defines the ladder shape, quoted verbatim:

> **The ladder** runs from full precision down to the deranged zone: F16
> (converted locally as the reference, never a download recommendation),
> then Q8, Q6, Q5, Q4, Q3, down to the lowest level the tracked uploaders
> actually publish, expected around IQ2_XXS, with IQ1_S included only
> where it exists.

and, for the file-not-label rule and the canonical uploader (ibid.):

> **A quant level is a file, not a label.** […] The canonical ladder uses
> bartowski's imatrix quants.

The file-level pilot precedent (`plan.md`, verified against the live 1.5B
repo 2026-08-26, quoted verbatim):

> **Pilot ladder (verified against the live HF repo
> `bartowski/Qwen2.5-1.5B-Instruct-GGUF` on 2026-08-26):** Q8_0, Q6_K,
> Q5_K_M, Q4_K_M, Q3_K_M, Q2_K, IQ2_M. The repo publishes nothing below
> IQ2_M […]

## C. The registered rule (user ruling, 2026-08-30)

**Per reference model, the confirmatory ladder is: F16 (the local
official-repo conversion, baseline only, never a download
recommendation) + Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K, plus the
lowest rung the tracked uploader publishes below Q2_K** — one file per
rung, drawn from the pinned bartowski imatrix manifests (PREREG §4
table), each result pinned to that file's sha256.

**Manifest resolution, stated plainly:** IDEA-v1.md §4 expected the bottom
"around IQ2_XXS, with IQ1_S included only where it exists." The pinned
manifests (llama-3.1-8b-bartowski.json rev `bf5b95e9…`,
qwen2.5-7b-bartowski.json rev `8911e8a4…`) carry **no IQ2_XXS, no IQ1_S,
no IQ1_M for either model**; the lowest published rung below Q2_K is
**IQ2_M in both**. By the "lowest published" rule, **the registered
bottom rung is IQ2_M for both reference models.** The expectation named
IQ2_XXS; the manifest reality is IQ2_M; the rule, not the expectation,
governs.

**Registered reference ladder, both models (8 rows):**
F16 (local) · Q8_0 · Q6_K · Q5_K_M · Q4_K_M · Q3_K_M · Q2_K · IQ2_M

Every other manifest file (IQ3/IQ4 variants, _S/_L/_XL sizes, Q4_0
family, ARM-repacked Q4_0_x_x, f32/f16 uploads) remains
provenance-pinned by §4 but is **not** a confirmatory rung and enters no
reference table. The shootout and official-vs-community arms (§5) are
unchanged by this amendment.

## D. Cell family and multiplicity

The confirmatory reference family is **2 models × 7 quant rungs × 4
scored suites = 56 cells**, plus 32 arm-scoped cells (8 arm files × 4
suites) outside the reference matrix. The 56-cell family matches the
launch-scope anticipation this ruling itself states (no external source
is cited for the anticipation). §8's dual rule applies unchanged:
per-cell verdicts descriptive at α = 0.05; any cross-cell headline
survives Holm over the family it aggregates across.

## E. Alternatives considered and rejected (user ruling, 2026-08-30)

- **Full manifest ladder (20 rungs/model, 160 reference cells)** and
  **family ladder + IQ3/IQ4 mid-rungs (10–11 rungs, 80–88 cells)**:
  REJECTED — both inflate the Holm family well beyond the anticipated
  launch scope.
- **In-house quantization of IQ2_XXS/IQ1_S for the reference models**:
  REJECTED — in-house rungs are 1.5B-spectacle-only (IDEA-v1.md §4's
  exception "exists to protect download recommendations, and the 1.5B is
  not one"; PREREG §4 bars `bitcliff-inhouse` rungs from reference
  tables, enforced in code).

## F. What this amendment does NOT do

- Does not modify any registered rule, seed, n, grading rule, budget, or
  the §4 manifests; it fills the rung-enumeration slot §4 never carried.
- Does not change the shootout/official arms (§5) or the 1.5B ladder
  (0B′ scope).
- Does not promote IDEA-v1.md to registered status beyond the sentences
  quoted here, and cites no other design document.
