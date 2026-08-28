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
   fix is a ticketed later work item (freeze-plan §10).

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
every later decile's draw too. Measured at seed=2718, n=500: M1, M2, and M3
each draw 500 records with ~444-450 unique object QIDs, but only
120/450 of M2's QIDs and 142/444 of M3's QIDs overlap with M1's. A mapping
built for the M1 draw alone (mirroring the seed-7411 precedent literally)
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
