# HANDOFF — BitCliff session handoff (written 2026-08-29)

For a fresh Claude Code session with no memory. This is an index, not an
archive: every claim below is verifiable in the named files or git history.

## 1. Project state

BitCliff measures what GGUF quantization deletes, per capability, under a
public pre-registration. Product spec: `claude/IDEA.md`.

**Done:**
- **Phase 0A pilot** (exploratory, numbers discarded): 1.5B ladder incl.
  in-house deranged rungs; gate PASSED. See `bitcliff/pipeline/PILOT_NOTES.md`.
- **The freeze (F4):** PREREG.md committed and OpenTimestamps-stamped.
  - Freeze commit: `5e6882b7a10c5e4670052855380e8646911db5f3`
  - Receipt: `freeze/freeze-commit-hash.txt.ots` — **Bitcoin-attested**
    (blocks 964240/964307; upgraded receipt committed).
- **Difficulty calibration + Amendment 1** (F16-only, all three models,
  local): registered via PREREG's "Amendment 1 (2026-08-29)".
  - Amendment commit: `00a1228ca6df39ddb5971e87920d4d0b78913cf5`
  - Receipt: `freeze/amendment1-commit-hash.txt.ots` — **Bitcoin-attested** (blocks 964530/964545/964549; upgraded receipt committed, `62243b2`).
- Also built and reviewed: static playground UI (`site/`, pilot fixtures),
  dataset packager with code-enforced embargoes
  (`bitcliff/pipeline/scripts/package_dataset.py`).

**Next: Phase 0B confirmatory runs** — the first GPU spend. Waits for the
user's explicit go. See §3.

## 2. Binding rules (non-negotiable for any new session)

- **PREREG.md governs everything.** Registered rules, seeds, n's, grading,
  budgets, and total orders may not change. Amendments only fill registered
  slots via the documented dated-and-stamped ceremony; they never alter
  registered rules silently.
- Registered parameters live in **config files**, never hardcoded.
- Nothing confirmatory deviates from PREREG. Exploratory work is declared
  exploratory and its numbers are discarded.
- A decision PREREG doesn't cover is **never decided silently** — it goes to
  `OPEN_QUESTIONS.md` and work moves on until the user rules.
- **No AWS/GPU spend, no timestamping, no pushes, no messages to anyone**
  without the user's explicit go. AWS conventions (when authorized):
  `bitcliff/.claude/skills/aws-ops/SKILL.MD` (us-east-1 only, g6e for GPU,
  tag instances, sync artifacts to S3 then TERMINATE, stop-and-ask over
  $500 credits / $50 cash).
- Build work runs **subagent-driven with reviews** (superpowers SDD skill:
  fresh implementer per task, independent review, fix loops, final review).

## 3. The 0B task, as registered

All settings below are registered in PREREG.md (+ Amendment 1). Confirmatory
= quant ladders vs F16, paired per item, deterministic (greedy, temp 0,
top_k 1, seed 42; global budget 1024 tokens; per-suite: longctx 32,
factual_qa 64).

**Models & ladders** (files pinned by sha256 in
`bitcliff/pipeline/reference-manifests/*.json`):
- Llama-3.1-8B-Instruct + Qwen2.5-7B-Instruct: bartowski imatrix ladders
  (reference pair). Qwen2.5-1.5B: existing local ladder (confirmatory rerun
  on the serving machine is 0B′ per `claude/IDEA.md` §10).
- Shootout (8B only, Q4_K_M+Q3_K_M): unsloth + mradermacher static/i1
  (`shootout-8b.json`). Official-vs-community: Qwen/Qwen2.5-7B-Instruct-GGUF
  vs bartowski (`qwen2.5-7b-official.json`).
- F16 references: converted locally, already on disk under
  `bitcliff/pipeline/models/f16/` (official-repo provenance; Llama needed
  the user's HF login — it exists, don't re-download).

**Suites & registered item sets:**
- `longctx_retrieval` 2b (site/dataset): PG-1184 corpus (hash in
  CORPUS_MANIFEST §1), **n=96, seed 2024** (Amendment 1 §A). Settings fixed
  by Amendment 1 §C: **all three models run multivalue4 @ t=8192** — 1.5B
  in-band (0.750); 7B (F16 1.000) and Llama-8B (0.9896) under the §A2
  out-of-band fallback, disclosed. Prompts NEVER published/displayed.
- `longctx_retrieval` 2a (Q5 comparability, 1.5B only): PG-essays corpus,
  seed 2024, n=96, paper-exact; outputs/stats only, never published.
- `arithmetic`: GSM8K, n=500, seed 3141. `arithmetic_twins`: all 47 pairs,
  seed 1301, EMBARGOED (private/, refused by the packager).
- `factual_qa`: PopQA, n=500, seed 2718, **mix M3** (out-of-band, disclosed,
  all models), augmented aliases (`data/popqa_wikidata_aliases_seed2718.json`),
  subject-echo-guarded grading.
- Tokenizer-match assertion (§3.1) runs before each confirmatory generation.

**Deliverables:** per-cell stats per PREREG §8 (M=3pp, CI-inclusion
equivalence, four cell states, dual Holm rule for cross-cell headlines),
divergence measurements (10-token answer span primary, digits-only
sensitivity), cliff tables, packaged dataset (embargoes enforced by
`package_dataset.py`), all runs' manifests hash-pinned.

## 4. Key files (one line each)

- `PREREG.md` — the registered protocol; Amendment 1 at the end; §7
  calibration rule, §8 statistics, §11 dataset scope, §15 ceremony.
- `OPEN_QUESTIONS.md` — every uncovered decision + the user's rulings
  (§1–§5 all resolved as of 2026-08-29).
- `MORNING_REPORT.md` — overnight run summary: calibration results table,
  what completed, band-mis-centering finding.
- `bitcliff/pipeline/PILOT_NOTES.md` — pilot verdicts + freeze/amendment
  commit cross-references.
- `bitcliff/pipeline/calibration/<model>/summary.json` + `measurements.jsonl`
  — every calibration measurement with item-set hashes.
- `bitcliff/pipeline/CORPUS_MANIFEST.md` / `LICENSE_AUDIT.md` /
  `INHOUSE_QUANTS.md` / `GRADER_CHARACTERIZATION.md` — corpus hashes,
  license verdicts, in-house quant provenance, 3-round grader evidence.
- `bitcliff/pipeline/reference-manifests/*.json` — GGUF file lists with
  sha256 (from HF API lfs.oid; no downloads).
- `bitcliff/pipeline/vendor/bundle-docs/` — the six vendored multivalue2
  docs (MECHANISM/TASK/GRADING/ANSWER_TOKENS/PAPER_CONFIG/PROVENANCE),
  sha-enumerated in PREREG §15.
- `plan.md`, `plan-task2.md`, `plan-freeze.md`, `freeze-plan.md` — executed
  plans (history; freeze-plan §10 holds the pre-0B tickets).
- `site/` — static playground (open via `python3 -m http.server -d site`).
- Pipeline: `bitcliff/pipeline/` (uv; `uv run pytest -q` → 203 passing).

## 5. Gotchas from this session (each cost real time)

- **Monitors that only watch for success lines miss crashes.** Two
  calibration stalls went unnoticed because the watcher had no
  Traceback/Error/Killed patterns. Always include failure signatures; when
  a subagent says "waiting on my monitor," verify the process is actually
  alive (`ps`).
- **PopQA's `obj_id` is NOT the Wikidata QID.** The QID is encoded in
  `o_uri`. Anything touching alias augmentation must derive from `o_uri`.
- **The factual_qa grader has a subject-echo guard** (aliases occurring in
  the question are ineligible; empty effective set → wrong). It exists
  because round-1 characterization failed on subject-echo FPs. Grading rule
  verbatim in `GRADER_CHARACTERIZATION.md` §7.1 — code governs over prose.
- **`spectacle_only` rungs** (bitcliff-inhouse IQ1_S/IQ1_M/IQ2_XXS, 1.5B
  only) must never appear in reference tables or download recommendations —
  the flag is threaded config→manifest→report rows; exclusion is enforced
  in reporting code, keep it that way.
- **The packager refuses runs matching config 2a's signature** (multivalue2,
  seed 2024, t=4096, n=96) as a leak guard — a heuristic, since runs don't
  yet record corpus provenance. Pre-0B ticket in `freeze-plan.md` §10:
  record corpus_sha256 in run manifests, then replace the heuristic. Also
  pre-0B: verify llama-cpp `create_completion` sets finish_reason="length"
  on truncation (the token path's truncated flag depends on it).
- Two mix draws with different weights consume the shared RNG stream
  differently — item sets differ per mix even at the same seed (why the
  alias mapping unions M1/M2/M3 QIDs; `calibrate_f16.py --print-mix-overlap`
  reproduces the numbers).

## 6. Lessons appended during 0B (2026-08-31)

- **A transfer monitor must watch part/byte progress, not process liveness
  or log-line presence.** The take-3 S3 upload of the Llama F16 sat at ZERO
  completed parts for 2+ hours while looking alive: the process ran, the
  retry loop ticked, heartbeat lines printed — and the monitor's filter
  excluded the heartbeat lines and had no "progress hasn't advanced" alarm.
  `--no-progress` made the stall invisible in the log too. Rule going
  forward (user, standing): any transfer over 1 GB must report rate and ETA
  within 60 seconds and STOP AND ASK if the projection exceeds 30 minutes;
  monitors on transfers alert on stalled byte/part counts, not on the
  process being alive.
