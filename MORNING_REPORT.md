# MORNING REPORT — overnight run 2026-08-26 → 2026-08-28

Constraints held throughout: **zero AWS/GPU spend** (everything ran on this
machine), **nothing timestamped** (AMENDMENT_DRAFT.md waits for you),
**nothing pushed, no one messaged**. Every uncovered decision went to
OPEN_QUESTIONS.md; the four you ruled on mid-run are applied, one new one is
open.

---

## 1. What completed

**Task 1 — F16 difficulty calibration (all three models).**
- Built `scripts/calibrate_f16.py` implementing PREREG §7 exactly (registered
  ladder order, in-band rule, mix apportionment, §3.1 tokenizer-match
  assertion before first generation — passed for all three models). 33 real
  measurement runs recorded with item-set hashes and wall times; 203/203
  tests green. Four harness bugs found and TDD-fixed along the way (corpus
  CRLF, llama-cpp shutdown crash, missing t=8192 continuation when 4096 is
  all-too-easy, missing terminal state for above-band-everywhere).
- Llama-3.1-8B unblocked mid-run by your HF login: official-repo safetensors
  downloaded and converted locally (provenance clean). Qwen2.5-7B likewise.
- The seed-2718 Wikidata alias mapping (union over M1/M2/M3 draws, your
  ruling 3) was fetched output-blind and committed before any generation.

**Calibration results:**

| model | factual_qa (M1 / M2 / M3) | chosen mix | longctx result | chosen setting |
|---|---|---|---|---|
| Qwen2.5-1.5B | 0.114 / 0.112 / 0.110 | **M3** (out-of-band, disclosed per §7) | full ladder walked (non-monotone → registered linear fallback); multivalue4 in-band at both t | **multivalue4 @ t=8192** (0.646 @ 4096, 0.750 @ 8192; tie → larger t) `[USER-CONFIRM tie-break reading]` |
| Qwen2.5-7B | 0.140 / 0.196 / 0.192 | **M3** (out-of-band, disclosed) | **above band everywhere** — hardest registered setting (multivalue4 @ 8192) = 1.000 | **none — OPEN_QUESTIONS §5** |
| Llama-3.1-8B | 0.248 / 0.330 / 0.314 | **M3** (out-of-band, disclosed) | **above band everywhere** — hardest = 0.9896 | **none — OPEN_QUESTIONS §5** |

The substantive scientific picture: **the registered [0.6, 0.85] band is
mis-centered in both directions for 8B-class models.** Closed-book factual
recall is far below band for every model at every registered mix (the
popularity knob barely moves the 1.5B at all and never reaches 0.6 even for
Llama), while in-context retrieval is above band everywhere for both
8B-class models — only the 1.5B has an in-band longctx setting. factual_qa
is covered by §7's registered M3-plus-disclosure fallback; longctx is not
(no registered fallback exists) — that is the one open decision.

**AMENDMENT_DRAFT.md** — written, reviewed against every evidence file
(every accuracy, hash, and disclosure cross-checked), one factual error
found and fixed (a wrong provenance claim about `provisional_note` fields —
the corrected text directs readers to per-measurement timestamps), overlap
counts made reproducible via `calibrate_f16.py --print-mix-overlap`.
Contains: the per-model knob results and item hashes; the **2b n=96/seed
2024 registration-gap section** in your exact terms (dated, disclosed as
beyond §7's amendment scope, self-explanatory to a reader); the
**union-mapping disclosure** referencing the mapping `_meta`. PREREG is
untouched; nothing is stamped.

**Task 2 — playground UI** (`site/`): static, no backend, vanilla JS.
Ladder view with grouped picker + Random button, A/B view with a snapping
slider (default F16 vs IQ2_M), output cards with grade/truncated/loop
badges, uploader + in-house spectacle-only tags, word-level first-divergence
highlighting, hash permalinks with copy-link, exploratory-data and
browse-only banners. Fixtures: 50 pilot items × 11 rungs, retired-retrieval
suite excluded by the exporter, embargo triple-enforced. Reviewed and
approved with live in-browser verification (Playwright). Open with any
static server: `python3 -m http.server -d site`.

**Task 3 — dataset packager** (`scripts/package_dataset.py`): code-enforced
embargoes (longctx prompts structurally excluded + post-write leak scanner;
twins runs refused outright; retired retrieval excluded), deterministic
output, PREREG-hash-carrying manifest, `--verify-recipe` reconstruction
check (PASS against the registered 2a digest, offline). A review caught a
real PREREG-§11 gap — published longctx records couldn't carry
`key`/`depths`/`n_answer_tokens` — fixed by merging them from the same
verified reconstruction path. Your rulings 4a (golds published) and 4b
(heuristic kept + annotated, corpus-pointer ticketed) applied. 16/16 tests;
pilot-0a packaged to `dist/dataset-pilot-0a/`.

## 2. What's blocked / needs you

- **OPEN_QUESTIONS §5 (the one open question):** both 8B-class models are
  above the band at every registered longctx setting; §7 has no longctx
  fallback. Options written in the file: (a) register the M3-analogue
  (hardest setting + disclosure), (b) amend the candidate space (e.g.
  t=16384 — touches a non-amendable list, needs the same disclosed-gap
  treatment as the 2b fix), (c) your call. **Nothing chosen for 7B/Llama
  longctx until you rule.**
- **[USER-CONFIRM] in the amendment:** the 1.5B tie-break reading (both t
  in-band for multivalue4 → 8192 chosen per "ties toward larger
  target_tokens").
- **Amendment review + stamp:** read AMENDMENT_DRAFT.md; after your §5
  ruling is folded in, the F4+ ceremony (append to PREREG's amendment slot,
  commit, OpenTimestamps) runs on your word.
- Housekeeping when convenient: `ots upgrade freeze/freeze-commit-hash.txt.ots`
  (the freeze receipt's Bitcoin attestation should have landed by now).

## 3. OPEN_QUESTIONS.md — complete status

1. **HF gate (Llama-8B)** — RESOLVED by you (login + approval); Llama
   calibrated from official repo overnight.
2. **2b longctx n/seed unregistered** — RESOLVED by you (n=96/seed 2024);
   provisional measurements became the real ones; registration-gap section
   drafted in the amendment.
3. **Alias-mapping union over mixes** — RESOLVED by you (approved);
   disclosed in the amendment; overlap numbers now reproducible via
   `--print-mix-overlap`.
4. **Packager 4a/4b** — RESOLVED by you; both applied and reviewed.
5. **7B + Llama-8B longctx above band everywhere, no registered fallback** —
   **OPEN.** Your decision, options above. (Llama update appended
   2026-08-28: same terminal state, one decision covers both.)

## 4. Your next decision, exactly

Rule on **OPEN_QUESTIONS §5** and confirm the **tie-break reading**; then
say the word to fold both into AMENDMENT_DRAFT.md and run the stamp
ceremony. Separately and unchanged: **0B (first GPU spend) waits for your
explicit go** — nothing cloud-shaped ran or will run without it.
