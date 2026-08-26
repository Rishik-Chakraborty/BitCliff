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

1. **Length budget:** the registered budget is **1024 tokens** and does not
   churn. The pilot ran at 640 and clipped exactly 1/90 F16 outputs —
   spec-010 (pancake recipe), an unscored spectacle item; both scored suites
   had zero F16 truncations. 640 being near-sufficient confirms 1024 is
   generous. Confirmatory configs use 1024 as registered.
2. **CAVEAT — the pilot retrieval suite is a placeholder.** It saturates
   into short-range copying (even IQ2_M scores 0.9875 with 8 pairs in
   context), so **zero retrieval conclusions are drawn from pilot data** —
   not about durability, not about asymmetry, not about noise flips. The
   real multivalue2 task arrives as an external bundle at the freeze and
   replaces this suite entirely; the freeze then recalibrates difficulty so
   F16 lands ~0.6–0.85 on every scored suite with visible headroom
   (difficulty tuning is legitimate pilot use, disclosed in PREREG).
   Observed here and noted only as pipeline behavior: F16 itself makes
   errors on this placeholder (10 partials with hallucinated codes).
3. **Grading precision:** spot-checked low-rung retrieval "correct" grades —
   they are genuine answers, not context echoes, so the known substring
   false-positive did not drive the numbers here. The word-boundary fix
   stays queued for the freeze regardless.
4. **Q2_K below IQ2_M on arithmetic (0.325 vs 0.40): expected scheme
   pattern, flag, never smooth.** IQ-quants are known to outperform K-quants
   around 2.5 bpw, so this is not an anomaly to explain away — it is a
   candidate finding for registered question 5 (k-quant transfer /
   scheme-dependence) and the matrix shows it with a flag.
5. **Curation seeds for the launch 50:** Q2_K loop on arithmetic-1301-007
   (the fries problem — repeats "Griffin has 27 fries" forever), IQ2_M's
   mistranslations (spec-006) and broken limerick (spec-003) are the most
   shareable failures found. The truly dramatic content needs the in-house
   sub-2-bit rungs.

## Chosen parameters going forward

- `max_tokens`: 1024 for all confirmatory configs — the registered value,
  confirmed generous by the pilot; registered parameters do not churn.
- Retrieval: the placeholder suite is replaced at the freeze by the real
  multivalue2 bundle (external), then difficulty-recalibrated so F16 lands
  ~0.6–0.85 per scored suite; recalibration disclosed in PREREG.
- Ladder: unchanged for reference models; 1.5B spectacle ladder gains
  in-house IQ1_S/IQ1_M (+ IQ2_XXS) rungs (llama.cpp `llama-quantize` +
  imatrix), labeled `bitcliff-inhouse`, spectacle-only, never surfaced in
  any download recommendation.
