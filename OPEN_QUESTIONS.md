# OPEN QUESTIONS — overnight run 2026-08-26/27

## RESOLUTIONS (2026-08-27, user rulings — all four entries below are now closed)

1. **§1 HF gate:** resolved — user logged in, Llama-3.1 gate access approved;
   Llama-8B calibration resumed under official-repo provenance.
2. **§2 2b n/seed:** REGISTERED as n=96, seed 2024 (mirroring 2a). Tonight's
   provisional longctx measurements stand as the real calibration
   measurements. Recorded in AMENDMENT_DRAFT.md as a dedicated, dated,
   disclosed registration-gap section beyond §7's registered amendment scope.
3. **§3 alias-mapping union over M1/M2/M3:** approved — same mechanical
   output-blind rule, widened QID set with the measured rationale; disclosed
   in the amendment referencing the mapping file's _meta.
4. **§4a:** "expected" (gold passcodes) added to published longctx_retrieval
   fields; packager updated and re-run. **§4b:** config-2a refusal heuristic
   kept with an in-code revisit note; the durable corpus-pointer-in-items
   fix is a ticketed later work item (freeze-plan §10). (ticket closed 2026-08-29: manifest _run_config.corpus_sha256 + positive provenance check replaced the heuristic — see freeze-plan §10).

The original entries are preserved below as written overnight.

Decisions encountered tonight that PREREG and the work order do not cover.
Nothing below was decided; work continued past each per the order.

## 1. Llama-3.1-8B-Instruct is gated and this machine has no HF login

`meta-llama/Llama-3.1-8B-Instruct` is `gated: manual`; file downloads return
401, and `hf auth whoami` reports not logged in. The F16 reference must be a
local conversion from the official safetensors (registered provenance), so
**Llama-8B calibration could not run tonight.** Using an ungated mirror
(e.g. unsloth's or NousResearch's re-upload) for the safetensors and/or the
HF tokenizer would be an unregistered provenance substitution — not decided.

**Your decision:** log in (`hf auth login`) and accept the Meta license so
the official repo works, or approve a named mirror (recorded as a
substitution in the amendment), or defer Llama-8B calibration.

## 2. PREREG registers no n/seed for the 2b longctx_retrieval item sets

§3.2/§3.4 register n and seed for arithmetic and factual_qa; §3.1 config 2b
registers corpus, knobs, grading — but **no item count and no sampling seed**
for the 2b runs, and §7 fixes "every n" as non-amendable without ever stating
this one. The calibration's in-band measurements (and later the confirmatory
2b runs) need both.

**What was done tonight (labeled provisional everywhere):** the longctx
ladder measurements ran at **n=96, seed 2024** — mirroring the registered 2a
values — solely to produce curves for your review. The AMENDMENT_DRAFT's
item-hash lines are marked blocked on this decision.

**Your decision:** register the 2b n and seed (adopting n=96/seed 2024 makes
tonight's measurements the real ones; any other choice re-runs the
measurements, which is cheap now that the harness exists). Note this will be
a PREREG amendment beyond §7's registered amendment scope (knob settings
only), so it needs its own dated, disclosed amendment text.

## 3. Seed-2718 alias-augmentation mapping: widened to the UNION of M1/M2/M3, not one draw (task 1a)

PREREG §3.4 branch (b)'s augmentation rule was written and executed against
a single draw — `factual_qa.picked_records(seed=7411)` (uniform weights,
the only mix that existed at characterization time). Task 1a's work order
directed reproducing that same process for the confirmatory seed (2718),
"the same mechanical way." Building `calibrate_f16.py`, I measured that this
is not sufficient once the §7 mix knob is in play: `picked_records` draws
each popularity decile with `rng.sample(decile, take)` against ONE shared,
sequentially-advancing `random.Random(seed)`, and different mixes (M1
uniform / M2 / M3) apportion a different `take` per decile — which shifts
how much of the shared RNG stream each decile consumes, and therefore shifts
every later decile's draw too. At seed=2718, n=500, M1/M2/M3 each draw 500
records with ~444-450 unique object QIDs, only partially overlapping
between mixes — reproducible via
`uv run python scripts/calibrate_f16.py --print-mix-overlap` (no model
needed), which prints exact sizes/pairwise-overlap/union counts computed
by the pure `mix_qid_sets`/`mix_overlap_report` functions (currently:
sizes 444/450/444, pairwise overlaps M1&M2=120, M1&M3=142, M2&M3=347,
union=841). A mapping built for the M1 draw alone (mirroring the seed-7411
precedent literally)
would leave roughly two-thirds of M2's and M3's sampled items unaugmented —
not comparable to how augmentation was characterized (round 3, §3.4) or to
the M1 measurement within the same calibration run.

**What was done:** `scripts/calibrate_f16.py::ensure_alias_mapping` unions
`picked_records(seed=2718, weights=w)`'s object QIDs over all three
registered mixes (M1/M2/M3) before fetching — still the identical
mechanical, output-blind, no-per-item-judgment rule ("English label +
English aliases of the object entity, fetched for every sampled record"),
run and committed before any model output is read; only the SET of QIDs
handed to that rule is widened to match what this calibration actually
grades. `data/popqa_wikidata_aliases_seed2718.json`'s `_meta` records the
per-mix QID counts and this rationale.

**Your decision:** this is a mechanical widening of an already-registered
rule to a case (multiple mixes, one seed) PREREG's branch-(b) text did not
anticipate, not a change to the rule itself — but it is new scope beyond
what was literally instructed ("the seed-2718 draw," singular), so it's
logged here rather than silently assumed correct. If a narrower reading is
preferred (augment only the M1 draw, leave M2/M3 items with PopQA's
original `possible_answers` and no augmentation), the mapping file and the
`mixes` parameter to `ensure_alias_mapping` would need to change and the
factual_qa calibration would need to be re-run with the earlier chosen mix
(if a re-run is even needed — the M1-only mapping is a strict subset of the
union one already written, so nothing needs re-fetching, only re-selecting
which mapping to load per mix).

## 4. `scripts/package_dataset.py` (Task 3): two conservative choices PREREG §11 does not literally settle

**4a. Published `longctx_retrieval` records omit `expected` (the gold
match-string passcodes), not just `prompt`/`prompt_tokens`.** PREREG §11's
list of what ships for this suite is: "the generator ..., the seeds, the
corpus pointer + pinned hash ..., per-item metadata (item id, key, depths,
`n_answer_tokens`, config), model outputs, and a reconstruction recipe." It
does not mention publishing the gold passcodes, and the pilot work order's
own spec for this task ("Published instead: item id, key/depth/config
metadata IF present in items (else just id+config), outputs, grades, and a
RECONSTRUCTION.md") likewise omits them. There's no license/embargo reason
to withhold the passcodes themselves (they're freshly random per item, not
extracted from the corpus text), so this is a strict, cheap-to-relax
narrowing, not a safety-critical one. `build_longctx_metadata` in
`scripts/package_dataset.py` never reads `item["expected"]`; grades already
disclose correctness. If gold passcodes should ship too, add `"expected"` to
that function's allowlist behavior and re-run the packager (no re-grading
needed).

**4b. The packager refuses to package any `longctx_retrieval` run whose
items match PREREG §3.1 config 2a's exact signature** (variant
`multivalue2`, `target_tokens` 4096, seed 2024) — parsed heuristically from
item ids, since `items.jsonl` carries no corpus-pointer field to check
directly. §3.1 states config 2a's prompts "are never displayed on the site
and never published; outputs and statistics only," so this mirrors the
twins refusal rather than treating it as an ordinary suite exclusion.
Risk: if the user later registers the 2b item-set n/seed as 2024/4096
too (see §2 above — n=96/seed=2024 mirroring 2a is explicitly on the table
for 2b), this heuristic would incorrectly refuse a legitimate 2b package.
**Your decision:** confirm whether config-2a-shaped runs should always be
refused by this packager (current behavior), or whether the refusal should
instead key off something else (e.g., a corpus pointer recorded alongside
`items.jsonl` at generation time — not currently written by
`bitcliff_pipeline/__main__.py`) once the 2b n/seed is registered. Not a
blocker: `runs/pilot-0a` (packaged for real, task 3) has no
`longctx_retrieval` items at all, so this path was not exercised on real
data.

## 5. 7B longctx: the ENTIRE registered knob space is above band — §7 has no longctx fallback (2026-08-27)

Measured (F16, registered n=96/seed 2024, Gutenberg 2b corpus): the 7B scores
0.99-1.0 on every setting tried, including the hardest registered setting
(multivalue4 @ target_tokens 8192 = 1.00). Easier variants are implied at
ceiling a fortiori. No setting in the registered candidate space (variant
ladder x t∈{4096, 8192}) lands in [0.6, 0.85]. §7 registers a fallback for
factual_qa (M3 + disclosure) but NONE for longctx out-of-band-high — the
calibration script had no rule to apply and crashed at this terminal state
(patched to record the state gracefully instead; no measurements lost).

NOT decided. **Your options:**
(a) amendment registering the longctx analogue of the M3 rule: run the 7B at
    the hardest registered setting (multivalue4 @ 8192) with the out-of-band
    F16 value disclosed — symmetrical with factual_qa's registered fallback;
(b) amendment extending the registered candidate space (e.g. target_tokens
    16384, or N>4 variants) — a bigger change: it touches a registered
    total-order/candidate set that §7 lists as non-amendable, so it needs
    the same disclosed-gap treatment as the 2b n/seed fix;
(c) something else.
Note the same question will likely arise for Llama-8B (also an 8B-class
model); its measurements will tell.

**Update 2026-08-28:** Llama-8B measured — same terminal state (hardest
setting 0.9896); the §5 decision covers both 8B-class models.

**RESOLVED 2026-08-28 (user):** option (a) — M3-analogue longctx
fallback registered via the amendment; t=16384 extension rejected.

## 6. `configs/0b/0b-arm2-official.yaml` (P3): official Qwen GGUFs' imatrix status is undocumented

Building the 0B run configs (RUN_0B.md §2 P3), `reference-manifests/
qwen2.5-7b-official.json` (the Qwen/Qwen2.5-7B-Instruct-GGUF pin) carries no
per-file `imatrix` flag — unlike `shootout-8b.json`, whose unsloth/
mradermacher entries each record `uploader`/`imatrix` explicitly. No other
committed document (LICENSE_AUDIT.md, INHOUSE_QUANTS.md, PREREG.md,
RUN_0B.md) states how Qwen's own team quantized their official q4_k_m/
q3_k_m releases.

**What was done:** `0b-arm2-official.yaml` sets `imatrix: false` for both
official quants — the conservative choice (asserting `true` without
evidence would be an unverified provenance claim in a config PREREG's
cross-check test treats as authoritative), disclosed here rather than
silently assumed. This does not block config authoring or the cross-check
test (neither depends on the imatrix flag's value), but the manifest/
reporting layer would publish a wrong provenance flag for this arm if the
assumption is wrong.

**Your decision:** confirm `imatrix: false` for the official Qwen GGUFs
(no positive evidence either way), or provide/point to a source that
settles it (e.g. Qwen's own model card, a `quantize.imatrix.file` GGUF
metadata key readable once the file is downloaded) before Arm 2 actually
runs. Low stakes either way — this only affects a provenance label, not
which files are compared or how they're scored.

## 6. The reference-ladder rung set is not registered; registered expectation vs published files diverge (2026-08-30)

PREREG §4 registers ladders only as "pinned to the committed manifests" —
no per-rung enumeration, no subset rule. The product spec (claude/IDEA.md
§4) says the ladder runs "F16 ..., then Q8, Q6, Q5, Q4, Q3, down to the
lowest level the tracked uploaders actually publish, expected around
IQ2_XXS, with IQ1_S included only where it exists." Facts, verified against
the committed manifests (llama-3.1-8b-bartowski.json rev bf5b95e9...,
qwen2.5-7b-bartowski.json rev 8911e8a4...):

- IQ2_XXS: absent from BOTH manifests. IQ1_S: absent from both. IQ1_M:
  absent from both.
- Lowest quant bartowski publishes for both reference models: **IQ2_M**.
  Sub-3-bit files present (identical label sets): IQ2_M, Q2_K, Q2_K_L.
- The only IQ2_XXS/IQ1 rungs in the whole project are the 1.5B in-house
  spectacle rungs (PREREG §4), which are spectacle_only and barred from
  reference tables by registered rule.

So under IDEA §4's own operative clause ("lowest level the tracked
uploaders actually publish"), the reference bottom is IQ2_M — the
"expected around IQ2_XXS" expectation is unmet by the uploader's actual
catalog. That resolves the BOTTOM rung, but the MIDDLE of the ladder
(which of the 24 manifest files per model are confirmatory rungs) is
genuinely unregistered. A prior RUN_0B draft ruled a 7-rung "canonical"
ladder citing PREREG §10/§12 anchors; those anchors are 1.5B contexts
(blind-check materials, pilot rungs), not reference-ladder registrations —
the citation was overclaimed, and the RUN_0B "~60-cell multiplicity
anticipation" line has no source in PREREG or IDEA (PREREG §8's dual rule
is count-agnostic; ladder size only changes the size of the Holm family a
cross-cell headline must survive). Both statements are retracted here.

NOT decided. **Your options (per reference model; the two models' manifests
carry identical candidate label sets):**

(a) **Family ladder, one file per label family, bottom = lowest
    published:** F16 + Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K, IQ2_M
    (7 quant rungs). 2×7×4 = 56 reference cells (+32 arm cells).
    Closest to IDEA §4's family enumeration; fewest cells; the IQ-vs-K
    ~2.5 bpw comparison (Q5 exploratory candidate) is covered by
    IQ2_M vs Q2_K.
(b) **Full manifest ladder:** every runnable quant file in the pinned
    manifests — 20 rungs/model (24 minus f32/f16, which are not quants,
    minus Q4_0_4_4/4_8/8_8, ARM-repacked files the pinned llama.cpp build
    cannot load; technical exclusions, disclosed). 2×20×4 = 160 reference
    cells (+32). Maximal coverage; densest cliff localization; ~3× GPU
    cost of (a); descriptive per-cell verdicts unaffected, but any
    headline aggregating "from rung Y down" must survive a larger Holm
    family.
(c) **(a) plus the IQ mid-rungs** (IQ3_XS, IQ3_M, IQ4_XS both models;
    IQ4_NL Llama / Q4_0 Qwen are further candidates): 10-11 rungs,
    80-88 reference cells (+32). Adds the IQ-vs-K comparison at 3-4 bit.
(d) **In-house quantize IQ2_XXS (and IQ1_S) for the reference models** to
    reach IDEA's "expected" bottom. Flagged strongly: IDEA §4's in-house
    exception is explicitly spectacle/1.5B-only ("the published-files-only
    rule exists to protect download recommendations, and the 1.5B is not
    one"), and PREREG §4 bars in-house rungs from reference tables — this
    option would need its own disclosed registration amendment and is
    disfavored by the registered text.

Whichever option is chosen, the choice + rationale should be recorded as a
dated disclosure in the run documentation (it fills a gap PREREG never
registered, like §2/§5 above), and RUN_0B.md's matrix, GPU-hours, and cost
line get recomputed against it before any "proceed".

**NOT RESOLVED — awaiting your ruling.**

**RESOLVED 2026-08-30 (user): option (a)** — registered via Amendment 2
(drafted as AMENDMENT2_DRAFT.md, appended + OTS-stamped before any
confirmatory generation, on the user's "stamp it"): F16 (local) + Q8_0,
Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K + lowest-published-below-Q2_K, which
the pinned manifests resolve to IQ2_M for both reference models
(IQ2_XXS/IQ1_S absent). (b)/(c) rejected (Holm-family inflation beyond
anticipation); (d) rejected (in-house rungs are 1.5B-spectacle-only).
Note on sources, kept honest: the ruling cited an idea.md §4 file-level
enumeration and an idea.md §7 "roughly sixty confirmatory cells" sentence;
the repo's IDEA.md §4 carries the family-level ladder + lowest-published
rule (quoted in the amendment), the file-level enumeration exists as
plan.md's pilot-ladder precedent (quoted), and the "roughly sixty" sentence
was not found anywhere in the repo — flagged inside AMENDMENT2_DRAFT.md
for the user to source or strike before stamping.

## 7. PREREG §8 registers no RNG seed for the paired-bootstrap CIs (2026-09-03)

§8 registers the machinery (McNemar exact; two-sided 95% CI on Δaccuracy by
paired bootstrap, 10,000 resamples) but no seed for the bootstrap RNG —
unlike every sampling seed in §3. The CIs are Monte-Carlo estimates; an
unseeded run is irreproducible, and seed choice marginally moves CI
endpoints (hence potentially a cell's state at the margin).

**What was done (labeled provisional):** the analysis runs at **seed 8271**
(fresh; distinct from every registered seed: 42, 1301, 2024, 2718, 3141,
7411, 20260828), recorded in every output row. If you ratify it, tonight's
numbers stand; any other choice re-runs the analysis (cheap, local,
deterministic given the seed).

**Your decision:** ratify seed 8271 (or name another), recorded as a dated
disclosure in the findings document — the same disclosed-gap treatment as
§2's 2b n/seed.

**§7 RESOLVED 2026-09-04 (user): seed 8271 RATIFIED**, together with the
derived per-cell RNG rule (`8271:{run_id}:{quant_label}:{suite}`) and the
per-pair rule (`8271:pair:{name_a}:{name_b}:{suite}`, names sorted) the
driver documents in FINDINGS_0B.md. A four-seed robustness sweep (seeds
1-4) confirming state/cliff/Holm invariance is logged in FINDINGS_0B.md.

## 8. factual_qa mix-order bug: every 0B factual_qa cell ran on a NON-registered item set (2026-09-04)

Investigating the F16 cross-machine delta (user request) found its true
cause — not hardware. `configs/0b/*.yaml` copies PREREG §7's M3 weight
vector in its prose order (decile 1 = most popular), but
`factual_qa.items_from_records` indexes weights against deciles built in
ASCENDING s_pop order (index 0 = least popular). `calibrate_f16.py`
reverses the vector for this convention; the 0B configs did not, and
`tests/test_0b_configs.py` asserted the yaml against its own equally
unreversed constant. Net effect: the cloud confirmatory run sampled a
DIFFERENT 500-item factual_qa set (item_set sha256 `ac5cb282…`) than the
registered M3 set Amendment 1 records (`2e53ca0e…`). Verified by
rebuilding both sets from PopQA. Evidence pack:
`bitcliff/pipeline/analysis/0b/F16_CROSS_MACHINE.md`.

Cross-machine drift itself is negligible: on identical item sets, Mac
and cloud F16 grades agree 0/500 discordant for both models, and the Mac
regen on the registered set reproduces Amendment 1's values exactly
(0.314 / 0.192).

**Scope of contamination:** all factual_qa cells in FINDINGS_0B.md — 22
ladder/arm cells + the factual_qa shootout pairs, including both
factual_qa cliff/Holm verdicts. Internally consistent (every rung graded
the same wrong set, so paired deltas are honest measurements OF THAT
SET) but not the registered measurement. longctx (item-set hash matches
calibration), arithmetic (seed 3141), and twins (fixed 94) are
unaffected. NLL divergence is longctx-only — unaffected.

**What was fixed in code (no re-run, no silent decision):** configs
corrected to the convention-reversed vector reproducing the registered
`2e53ca0e…` set; test now cross-checks against `calibrate_f16.py`'s
constant. FINDINGS_0B.md carries a prominent caveat on every factual_qa
result pending this ruling.

NOT decided. **Your options:**
(a) **Re-run factual_qa generation only, all 26 rungs, on the registered
    set.** GPU cost ≈ 3-4 h on a fresh box from the CUDA-baked AMI
    (which already carries the hash-gated F16s; quants re-download in
    ~15 min) ≈ **$8-12**. Then re-grade, re-analyze (minutes, local),
    and the published factual_qa cells become the registered
    measurement. Disclosure: a dated note that the first factual_qa
    pass ran on a mis-apportioned mix and was repeated on the
    registered set before any publication.
(b) **Publish as-is with a deviation disclosure** — factual_qa cells
    labeled as measured on a disclosed non-registered draw (same rule,
    wrong apportionment order). Cheaper, permanently ugly for a
    pre-registered project.
(c) Anything else you rule.

**NOT RESOLVED — awaiting your ruling. No GPU spend without your go.**

**§8 RESOLVED 2026-09-05 (user ruling: option (a), executed):** factual_qa
regenerated for all 26 rungs on the registered item set (run 0b2, box
i-0ce4aad4171306a13 from the pinned CUDA AMI — same instance class,
sampling, n_ctx as the first pass; item set the only diff; boot-time
item-set gate enforced 2e53ca0e… before generation; ~55 min, ≈$2.10 of
the $25 cap). Both F16 baselines reproduce Amendment 1 §C exactly (llama
0.314, qwen 0.192 — item-count-identical), confirming machine-independence
at the grade level. Analysis re-sourced factual_qa from 0b2 (source_run_id
column); first-pass factual_qa kept as a disclosed sensitivity run in
FINDINGS_0B.md Appendix A. Cliff/Holm changes vs first pass: llama
unchanged (Q3_K_M, SURVIVES); qwen cliff IQ2_M → Q2_K (SURVIVES).
