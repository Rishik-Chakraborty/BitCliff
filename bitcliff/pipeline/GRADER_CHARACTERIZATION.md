# Alias-grader characterization — exploratory PopQA run

**Status: EXPLORATORY (binding user ruling 1b).** This document exists to characterize
the `factual_qa` alias grader (`suites/factual_qa.py:grade`) before it is frozen — not
to measure model quality. **Every accuracy/retention number below is thrown away** in
the same sense as the Phase 0A pilot (`PILOT_NOTES.md`): it is pilot-legitimate data
used only to inform calibration and grader design. The only things that survive into
the freeze are (1) the grader-FP/FN characterization and verdict, and (2) the F16
difficulty-band observation, which feeds the registered popularity-mix calibration
rule. No confirmatory claim about quant retention on `factual_qa` is made or implied
by this run.

## 1. Run configuration

- Config: `configs/qwen2.5-1.5b-factualqa-explore.yaml`
- Run id: `factualqa-explore` (`runs/factualqa-explore/`, gitignored)
- Command: `uv run python -m bitcliff_pipeline configs/qwen2.5-1.5b-factualqa-explore.yaml --run-id factualqa-explore --stage all`
- Ladder: 3 rungs — **F16**, **Q4_K_M** (bartowski, imatrix), **Q2_K** (bartowski,
  imatrix) — a subset of the pilot's 1.5B ladder, spanning full precision down to the
  bottom published quant.
- Generation: seed 42, temperature 0.0, top_k 1, `n_ctx` 4096, global `max_tokens` 640
  (unused override for this suite — see next line).
- Suite: `factual_qa: {n_items: 500, seed: 7411, max_tokens: 64}` only — PopQA
  (`akariasai/PopQA`, `test` split; MIT via canonical source, `LICENSE_AUDIT.md` §1),
  stratified by popularity decile, prompt `"Answer with just the answer: {question}"`,
  strict word-boundary alias-substring grading (`suites/factual_qa.py:grade`).
- Downloads: none — F16 and both quant `.gguf` files pre-existed under `models/`
  (verified before the run; the `download` stage wrote only the manifest).
- Result: 500 items × 3 rungs = 1500 generations, all completed, graded, and reported
  without pipeline errors. `runs/factualqa-explore/results.csv`:

  | quant_label | n | correct | wrong | truncated | loops | accuracy | retention |
  |---|---|---|---|---|---|---|---|
  | F16    | 500 | 60 | 440 | 1 | 0 | 0.120 | 1.000 (baseline) |
  | Q4_K_M | 500 | 65 | 435 | 0 | 0 | 0.130 | 1.083 |
  | Q2_K   | 500 | 48 | 452 | 1 | 1 | 0.096 | 0.800 |

## 2. F16 difficulty band

F16 accuracy on this draw is **0.120**, well outside the registered **[0.6, 0.85]**
calibration band (`freeze-plan.md`: "the popularity mix is the calibration knob for the
F16 [0.6, 0.85] band"). This is expected and not itself a problem: `n_items=500, seed=7411`
draws evenly across all ten popularity deciles including the long tail, which is exactly
where a 1.5B closed-book model is weakest, and this run makes no attempt to tune the mix.
Per the registered rule, **the popularity-mix knob (skewing the decile draw toward
higher-popularity deciles) is the documented lever that will be used to land F16 inside
the band at calibration time** — this run does not exercise that knob and draws no
conclusion beyond "the knob will need to move a fair distance toward the popular end."
This observation is the one number from this run that is meant to carry forward.

## 3. Adjudication sample — rule, seed, actual draw

**Rule (fixed, documented):** for each of the two grader outcomes (`correct`,
`wrong`), draw up to 5 records from each of the 3 rungs (target 5/5/5 = 15 per side,
30 total), using a fresh `random.Random(20260826)` per side, iterating rungs in order
`[F16, Q4_K_M, Q2_K]`, sampling from that rung's pool (sorted by `item_id` ascending)
via `rng.sample`. Script: adjudication draw reproduced from `runs/factualqa-explore/
{items,grades}.jsonl` and `runs/factualqa-explore/outputs/*.jsonl`.

**Actual draw:** exactly 5/5/5 achieved on both sides (all three rungs had ≥5
`correct` and ≥5 `wrong` records in the full 500-item run, so no top-off was needed):

- `correct` sample: F16=5, Q4_K_M=5, Q2_K=5 (15 total)
- `wrong` sample: F16=5, Q4_K_M=5, Q2_K=5 (15 total)

## 4. Adjudication table (30 items)

Verdict = my judgment of whether the grader's `state` for that record is right, based
on reading the question, the full alias list, and the raw model output. "Right" means
the grader's call matches what the output actually asserts; "grader-FP" means graded
`correct` but the output does not actually assert a valid alias as its answer (e.g. an
alias string appears only because it echoes the question's subject); "grader-FN" would
mean graded `wrong` but the output asserts a valid alias in different surface form.

| # | Item ID | Rung | Grade | Verdict | Reason | Output excerpt |
|---|---|---|---|---|---|---|
| 1 | factual_qa-7411-0297 | F16 | correct | Right | Clean single-token answer, exact alias match ("football") | `football` |
| 2 | factual_qa-7411-0363 | F16 | correct | **grader-FP** | Alias "Asti" matched only because it echoes the question's subject noun ("Asti is the capital of..."); the model's actual claimed answer is "Piedmont, Italy", which matches no alias (expected: Province of Asti) | `Asti is the capital of Piedmont, Italy.` |
| 3 | factual_qa-7411-0417 | F16 | correct | Right | Clean answer, exact alias match | `Alternative rock` |
| 4 | factual_qa-7411-0008 | F16 | correct | Right | Clean answer, exact alias match | `Field hockey` |
| 5 | factual_qa-7411-0470 | F16 | correct | Right | Clean answer, exact alias match | `Brussels` |
| 6 | factual_qa-7411-0063 | Q4_K_M | correct | Right | Clean answer, exact alias match | `United States` |
| 7 | factual_qa-7411-0417 | Q4_K_M | correct | Right | Clean answer, exact alias match | `Alternative rock` |
| 8 | factual_qa-7411-0144 | Q4_K_M | correct | Right | Clean answer, exact alias match | `India` |
| 9 | factual_qa-7411-0223 | Q4_K_M | correct | Right | Clean answer, exact alias match | `United States` |
| 10 | factual_qa-7411-0116 | Q4_K_M | correct | Right | Clean answer, exact alias match | `Australia` |
| 11 | factual_qa-7411-0036 | Q2_K | correct | Right | Clean answer, exact alias match | `France` |
| 12 | factual_qa-7411-0243 | Q2_K | correct | Right | Clean answer, exact alias match | `Soccer` |
| 13 | factual_qa-7411-0020 | Q2_K | correct | Right | Clean answer, exact alias match | `Soccer` |
| 14 | factual_qa-7411-0057 | Q2_K | correct | Right | Clean answer, exact alias match | `Kaunas` |
| 15 | factual_qa-7411-0034 | Q2_K | correct | Right | Clean answer, exact alias match | `Canada` |
| 16 | factual_qa-7411-0271 | F16 | wrong | Right | Different city named, no alias present ("Nanjing" vs. expected "Shangqiu") | `Nanjing` |
| 17 | factual_qa-7411-0304 | F16 | wrong | Right | Different person named, hedged full sentence, no alias present | `The screenwriter for "Travelers" was David S. Goyer.` |
| 18 | factual_qa-7411-0343 | F16 | wrong | Right | Different person named, no alias present (and the sentence is self-contradictory — see note) | `The Prize was the screenwriter for the film "The Prize" by John Logan.` |
| 19 | factual_qa-7411-0027 | F16 | wrong | Right | Different (well-known but unrelated) author named, no alias present | `The author of "Science-Fantasy Quintette" is Liu Cixin.` |
| 20 | factual_qa-7411-0479 | F16 | wrong | Right (borderline) | Hypernym, not an alternate phrasing: "Christianity" is a broad category, not the specific denomination ("Anglicanism") the alias list asks for — a real information loss, not a paraphrase, so this is not a grader-FN | `Christianity` |
| 21 | factual_qa-7411-0046 | Q4_K_M | wrong | Right | Different person named, no alias present | `Thea von Harbou` |
| 22 | factual_qa-7411-0353 | Q4_K_M | wrong | Right | Different country named (confuses Victoria, Seychelles with Victoria, Australia), no alias present | `Australia` |
| 23 | factual_qa-7411-0181 | Q4_K_M | wrong | Right | Different sport named, no alias present | `Basketball` |
| 24 | factual_qa-7411-0091 | Q4_K_M | wrong | Right | Different occupation named, no alias present | `Pathologist` |
| 25 | factual_qa-7411-0291 | Q4_K_M | wrong | Right | Different author named, no alias present | `The author of "A Good Year" is John Irving.` |
| 26 | factual_qa-7411-0121 | Q2_K | wrong | Right | Different person named, no alias present | `Charles Lycett` |
| 27 | factual_qa-7411-0077 | Q2_K | wrong | Right | Different person named, no alias present | `John Flynn` |
| 28 | factual_qa-7411-0072 | Q2_K | wrong | Right | Different occupation named, no alias present | `Journalist` |
| 29 | factual_qa-7411-0227 | Q2_K | wrong | Right | Different person named, no alias present | `Edmund Griffin` |
| 30 | factual_qa-7411-0046 | Q2_K | wrong | Right | Different (well-known but unrelated) author named, no alias present | `J.R. R. Tolkien` |

## 5. Counts and verdict

- **Grader-FP: 1/15** (item #2, `factual_qa-7411-0363`) — 6.7%.
- **Grader-FN: 0/15** — 0%.
- **Proposed bar:** PASS iff FP = 0/15 **and** FN ≤ 2/15.
- **Result: FAILS the bar** — FN is comfortably within tolerance (0 ≤ 2), but FP = 1/15
  is nonzero, which violates the FP = 0/15 requirement.

### Failure mode identified

The one FP is a real, reproducible failure mode of the substring grader, not sampling
noise: **when the question's subject-entity name is itself a valid alias for the
expected object** (here, the comune "Asti" and its object "Province of Asti", whose
alias list includes the bare string "Asti"), a model output that merely echoes the
question's subject as the grammatical subject of its sentence — while giving a
different, wrong entity as the actual claimed answer — gets scored `correct` purely
because the subject echo happens to contain the alias string. The grader has no notion
of "this alias occurred in the answer position," only "this alias occurred anywhere in
the normalized output." This is exactly the "alias matched inside an off-topic
sentence" failure mode named in the task brief, observed once in a 30-item sample.

## 6. BLOCKED — bar failure, fallback is not this task's call

Per the task brief: **the characterization bar fails** (FP = 1/15 ≠ 0/15). Per binding
instruction, this is reported **BLOCKED** with the table above. Whether to trigger the
named TriviaQA fallback (`LICENSE_AUDIT.md`, `freeze-plan.md`: "fallback if the audit
fails: TriviaQA"), to tighten the grader (e.g. require the alias to occur outside a
leading subject-echo span, or drop leading-clause self-reference before matching), or
to accept the observed FP rate as a released caveat is a **controller/user decision**,
not made here. Note that the FP mechanism found (subject-name-equals-alias) is a
property of PopQA's own aliasing (an entity being its own object-alias), so it is not
obviously avoided by switching datasets — TriviaQA's alias lists could exhibit an
analogous case; this is a data point for that decision, not a resolution of it.
