# Phase 0A Pilot Notes — run `pilot-0a` (2026-08-26)

**Status: EXPLORATORY (spec §10). Every number below is thrown away; only the
parameter choices and curation decisions survive into the freeze.**

Run: 8 rungs (F16 + 7 bartowski imatrix quants) × 90 items (40 retrieval,
40 GSM8K arithmetic, 10 unscored spectacle), fully deterministic
(seed 42, temp 0, top_k 1), macOS arm64 / llama-cpp-python 0.3.35.
All 720 outputs generated, graded, and reported without pipeline errors.

## Gate verdict (spec §10 bars)

- **Signal bar: PASS.** The arithmetic curve is informative and has the
  flat-then-cliff shape: accuracy 0.775 (F16) → 0.75 (Q8/Q6) → 0.675 (Q5) →
  0.65 (Q4) → 0.60 (Q3) → collapse to 0.325 (Q2_K) / 0.40 (IQ2_M), with
  loops and truncations appearing only at the bottom two rungs. The curves
  are not noise. **The project lives.**
- **Spectacle bar: PARTIAL — in-house bottom rungs required.** At IQ2_M
  (the lowest published rung for this 1.5B) degradation is real but subtle:
  broken French/Spanish ("Bonne matin", "Bon día"), garbled German word
  order, limericks that lose meter, wrong Beatles facts, confused multi-step
  arithmetic, occasional infinite loops at Q2_K. Entertaining to a careful
  reader; not word-salad. The spec §4 exception applies as anticipated:
  quantize in-house bottom rungs (IQ1_S / IQ1_M, clearly labeled
  `uploader: bitcliff-inhouse`) for the deranged zone before launch.

## Findings that feed the freeze

1. **Length budget:** 640 tokens clipped exactly 1/90 F16 outputs — spec-010
   (pancake recipe), an unscored spectacle item. Both scored suites had zero
   F16 truncations, so the pilot's scored numbers are budget-clean. Decision:
   raise `max_tokens` to 896 in the confirmatory config so the frozen budget
   clips nothing at all; not worth re-running the throwaway pilot for.
2. **Retrieval task is too easy as instantiated.** With 8 pairs in context,
   even IQ2_M scores 0.9875 — the task measures short-range copying, which
   survives 2-bit. F16's own errors (10 partials, e.g. answering
   "9493, 8577" for expected "8974, 9493" — one code hallucinated) mean the
   low rungs sometimes "beat" full precision on noise flips. For the freeze:
   reconcile with the paper's exact multivalue2 (more pairs / distractors /
   longer range) so retrieval has headroom to be damaged.
3. **Grading precision:** spot-checked low-rung retrieval "correct" grades —
   they are genuine answers, not context echoes, so the known substring
   false-positive did not drive the numbers here. The word-boundary fix
   stays queued for the freeze regardless.
4. **Odd case to flag, never smooth (spec §5):** Q2_K scores *below* IQ2_M
   on arithmetic (0.325 vs 0.40) — a non-monotonic bottom, exactly the kind
   of row the matrix must show with a flag.
5. **Curation seeds for the launch 50:** Q2_K loop on arithmetic-1301-007
   (the fries problem — repeats "Griffin has 27 fries" forever), IQ2_M's
   mistranslations (spec-006) and broken limerick (spec-003) are the most
   shareable failures found. The truly dramatic content needs the in-house
   sub-2-bit rungs.

## Chosen parameters going forward

- `max_tokens`: 896 (was 640) for all confirmatory configs.
- Retrieval: harder instantiation at freeze (paper-reconciled multivalue2).
- Ladder: unchanged for reference models; 1.5B spectacle ladder gains
  in-house IQ1_S/IQ1_M rungs (llama.cpp `llama-quantize` + imatrix),
  labeled in-house, never a download recommendation.
