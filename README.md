# BitCliff

**Watch an LLM lose its mind as you drag the bits out — then get per-capability numbers on what each GGUF file actually deletes, so you know which one to download.**

BitCliff measures what GGUF quantization destroys, capability by capability, under a **public pre-registration**. For each capability on each model, **the cliff is the highest-precision file at which that capability is measurably damaged beyond the registered margin.** "Retrieval cliff: Q3_K_M" is a sentence, a badge, a table row, and the thing people will argue about.

## Why

GGUF selection runs on folklore. A model page shows fifteen files and their sizes; the only quality signals are perplexity and corpus KLD — both token-averaged, neither capability-aware. People pick by file size and Reddit memory. BitCliff replaces that with two things on one site:

1. **Spectacle** — a slider that compresses a model in front of you until it forgets facts, loops, and finally speaks salad.
2. **Reference** — per-capability retention tables with pre-registered statistics, where every cell is one of four honest states: *damaged*, *small real loss within margin*, *equivalent*, or *indeterminate* (never rounded up to "free").

The slider buys traffic. The tables keep it.

## The rigor

Every confirmatory rule was registered **before any confirmatory data exists**, and the registration is tamper-evident:

- **`PREREG.md`** registers every question, suite, parameter, seed, grading rule, statistical rule, budget, and embargo. Nothing confirmatory deviates from it; exploratory work is declared exploratory and its numbers are discarded.
- The freeze commit (`5e6882b`) is stamped with **OpenTimestamps and Bitcoin-attested** — receipt in `freeze/freeze-commit-hash.txt.ots`. "The rules were decided after seeing the data" is impossible by construction. (The draft-status banner at the top of `PREREG.md` predates the freeze; the receipts in `freeze/` are the durable record.)
- Changes only happen through a **dated, stamped amendment ceremony** filling registered slots. Amendment 1 (difficulty calibration, 2026-08-29) is likewise Bitcoin-attested (`freeze/amendment1-commit-hash.txt.ots`).
- Any decision PREREG doesn't cover is never decided silently — it goes to `OPEN_QUESTIONS.md` with the eventual ruling recorded.
- Model files are pinned by sha256 (`bitcliff/pipeline/reference-manifests/`); a quant level is a **file, not a label** — bartowski's, unsloth's, and mradermacher's Q4_K_M are different files with different calibration, and results are pinned to hashes.

## Models and suites

| Role | Model | Notes |
|---|---|---|
| Spectacle | Qwen2.5-1.5B-Instruct | degrades dramatically; the only model with a live prompt box |
| Reference | Llama-3.1-8B-Instruct | bartowski imatrix ladder |
| Reference | Qwen2.5-7B-Instruct | bartowski imatrix ladder; plus official-vs-community and uploader-shootout arms |

Ladders run from a locally-converted F16 reference down through Q8…Q3 into the deranged zone (IQ2/IQ1 where published; in-house "spectacle-only" rungs for the 1.5B are labeled as such and never appear in download recommendations).

Suites: **long-context retrieval** (multivalue needle task), **arithmetic** (GSM8K + an embargoed contamination-twin set), and **factual QA** (PopQA with alias-augmented, subject-echo-guarded grading). All confirmatory generation is deterministic (greedy, temp 0, seed 42) and paired per item against F16.

## Repository layout

| Path | What it is |
|---|---|
| `PREREG.md` | The registered protocol; amendments appended at the end |
| `OPEN_QUESTIONS.md` | Every decision PREREG didn't cover, plus rulings |
| `HANDOFF.md` | Session handoff / index of project state |
| `freeze/` | OpenTimestamps receipts for the freeze and Amendment 1 |
| `bitcliff/pipeline/` | The measurement pipeline (Python, `uv`): configs, scorers, graders, calibration results, manifests, dataset packager |
| `bitcliff/pipeline/PILOT_NOTES.md` | Phase 0A pilot verdicts (exploratory; numbers discarded) |
| `site/` | Static playground UI (ladder view, A/B slider) over pilot fixtures |
| `claude/IDEA.md` | The full product spec |
| `plan*.md`, `freeze-plan.md` | Executed build plans (history) |

## Running things

```bash
# pipeline tests
cd bitcliff/pipeline && uv run pytest -q

# static playground
python3 -m http.server -d site
```

Confirmatory GPU runs (Phase 0B) are configured under `bitcliff/pipeline/configs/0b/` and gated behind the pre-registration; see `RUN_0B.md`.

## Embargoes

Some materials are deliberately withheld to keep the measurements meaningful: long-context prompts are never published or displayed, the arithmetic contamination-twin set is embargoed (`private/`, refused by the dataset packager), and the 2a comparability corpus publishes outputs/stats only. The packager (`scripts/package_dataset.py`) enforces these in code.

## Status (2026-08-30)

- ✅ Phase 0A pilot complete (exploratory; gate passed, numbers discarded)
- ✅ PREREG frozen and Bitcoin-attested; Amendment 1 (difficulty calibration) registered and attested
- ⏳ Phase 0B confirmatory runs: configs ready, pending go (one open question on the reference-ladder rung set — `OPEN_QUESTIONS.md` §6)
