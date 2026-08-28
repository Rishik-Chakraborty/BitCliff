# AMENDMENT DRAFT — PREREG §7 difficulty calibration

**Status: DRAFT, for user review. Not appended to PREREG.md. Not
committed as a stamped amendment. Not OpenTimestamps-stamped.** This file
exists so the exact amendment text can be reviewed before it is copied
into PREREG.md's `[AMENDMENT SLOT — difficulty calibration, appended and
stamped at F4+]` (§7) and stamped per §15.

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

**[USER-CONFIRM] tie-break interpretation:** both t values are in-band
for multivalue4 (0.646 at 4096, 0.750 at 8192); 8192 was chosen per §7's
"ties broken toward the larger `target_tokens`" — a defensible but
interpretive reading of "ties" (both in-band, not both equal). Confirm
this reading at review.

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

**Chosen: none. Terminal state: `above_band_everywhere`** — hardest
setting (multivalue4 @ 8192) F16 accuracy = 1.000, disclosed. §7
registers no longctx fallback for this state (unlike factual_qa's M3
rule). **Not decided here — see `OPEN_QUESTIONS.md` §5.**

### llama-3.1-8b-instruct

**factual_qa** — M1 0.248, M2 0.330, M3 0.314. None in [0.6, 0.85].
**Chosen mix: M3 — out-of-band, F16 accuracy 0.314, disclosed.**
Item-set sha256 (M3): `2e53ca0e73e9cbdd5d7bac672857ba918ff7b74d2405ddf62f37aa9a630091c6`
(identical to the other two models').

**longctx_retrieval** — t=4096: multivalue2/multivalue3/multivalue4 all
1.000, multiquery4 0.979 — too easy everywhere. Full ladder searched
again at t=8192: multivalue2 0.990, multivalue3 1.000, multiquery4
0.990, multivalue4 (hardest) 0.990 — **still nothing in-band.**

**Chosen: none. Terminal state: `above_band_everywhere`** — hardest
setting (multivalue4 @ 8192) F16 accuracy = 0.9896, disclosed. Same
unresolved state as the 7B; `OPEN_QUESTIONS.md` §5 now covers both
8B-class models and its resolution applies to both.

---

## D. Summary table

| model | factual_qa mix | factual_qa F16 acc | in-band? | longctx variant | longctx t | longctx F16 acc | in-band? |
|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b-instruct | M3 | 0.110 | no (disclosed) | multivalue4 | 8192 | 0.750 | **yes** |
| qwen2.5-7b-instruct | M3 | 0.192 | no (disclosed) | — (`above_band_everywhere`) | — | 1.000 (hardest) | no — no setting exists in-band |
| llama-3.1-8b-instruct | M3 | 0.314 | no (disclosed) | — (`above_band_everywhere`) | — | 0.9896 (hardest) | no — no setting exists in-band |

**Only the 1.5B has a fully in-band configuration for both suites** (and
only for longctx — its own factual_qa also falls back to M3
out-of-band). Both reference models (7B, 8B-class) score above the
[0.6, 0.85] band on every point in the registered longctx candidate
space, and below/near it (never above 0.33) on every registered
factual_qa mix — the opposite pattern from longctx. This asymmetry (one
suite saturates high, the other stays low, for the same larger models)
is itself worth carrying into the methodology writeup, independent of
how `OPEN_QUESTIONS.md` §5 is resolved.

---

## E. What this amendment does NOT do

- Does not modify PREREG.md.
- Is not OpenTimestamps-stamped.
- Does not resolve `OPEN_QUESTIONS.md` §5 (7B/Llama-8B longctx
  above-band-everywhere) — that requires a user decision on the options
  listed there (registering a longctx analogue of the M3
  disclose-and-run fallback; extending the registered candidate space;
  or something else) before a longctx setting can be chosen for either
  reference model.
- Does not change any registered seed, n, candidate set, or total order
  (§A above is a gap-fill of a value PREREG never stated, not a change
  to a registered one).
