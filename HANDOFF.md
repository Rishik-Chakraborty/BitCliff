# HANDOFF — BitCliff session handoff (rewritten 2026-09-07)

For a fresh Claude Code session with no memory. This is an index, not an
archive: every claim below is verifiable in the named files, git history,
or S3. Read this first; PREREG.md governs; nothing spends, stamps, pushes,
or messages anyone without the user's explicit go.

## 1. Repo state (as of this handoff's commit)

Local main and origin/main are IN SYNC at the commit preceding this
handoff commit: local HEAD `47e8462c5b8be7b0b029ee415c153088881b9252` ==
origin/main (verified 2026-09-07; this handoff commit itself is pushed as
part of the same instruction). Branch list: `main` only (work branches
`pre-0b-tickets`, `0b-analysis`, `factualqa-rerun` were merged no-ff and
deleted). Test suite: 420 passing offline (`uv run pytest -q
--ignore=tests/test_registered.py` in `bitcliff/pipeline/`; the ignored
file's ~20 tests rebuild item sets from live HF and also pass, slowly).

## 2. 0B final state

**The 0B confirmatory phase is COMPLETE and analyzed on registered data.**

- All 26 generation rungs (Llama-8B ladder 8, shootout arm 7, Qwen-7B
  ladder 8, official arm 3) × 4 suites; 26 NLL divergence rungs × 96
  items. factual_qa cells come from the **0b2 rerun on the registered
  item set** (see §3); everything else from the original 0b runs.
- `bitcliff/pipeline/analysis/0b/FINDINGS_0B.md` is FINAL: 88 cells with
  states/CIs/McNemar/flip-directions/truncations, cliff + Holm tables,
  the accuracy-inversion section, shootout-trigger verdict, seed-sweep
  disclosure, Appendix A sensitivity run. `cells.csv` carries
  `source_run_id` (0b2 substitution audit) and
  `acc_inversion`/`inversion_above` (PREREG §8 accuracy-level flag).
- Bootstrap seed **8271 RATIFIED** (OPEN_QUESTIONS §7) with per-cell rule
  `8271:{run_id}:{quant_label}:{suite}` and per-pair rule
  `8271:pair:{a}:{b}:{suite}`; 4-seed sweep on the final data: all
  cliffs/Holm/trigger invariant; two boundary cells disclosed in the
  sweep section (ratified seed governs, per user ruling).
- Headline results: cliffs — longctx IQ2_M/IQ2_M, arithmetic Q2_K/Q2_K,
  twins Q3_K_M/(none), factual_qa Q3_K_M (Llama) / Q2_K (Qwen); Holm
  verdicts are in the FINDINGS tables (llama-twins and qwen-longctx
  candidates are descriptive-only per the dual rule — read the tables,
  not this line). **PREREG §5 shootout trigger FIRED** → Qwen-7B
  uploader shootout is a REGISTERED follow-up measurement (see §5).
- Both GPU boxes TERMINATED (i-0d645775964749bbd for 0B,
  i-0ce4aad4171306a13 for 0b2), EIP `eipalloc-04a06d3cbff03d123`
  RELEASED, zero EC2 instances remain (verified at each teardown).
  **Spend ≈ $100.5 of the $250 cap** (0B ≈ $98, 0b2 ≈ $2.10).

## 3. Disclosure ledger — do not summarize this away; it is the record

1. **Freeze (F4):** PREREG commit `5e6882b7a10c5e4670052855380e8646911db5f3`,
   receipt `freeze/freeze-commit-hash.txt.ots`, Bitcoin-attested (blocks
   964240/964307).
2. **Amendment 1** (difficulty calibration + §A/§A2/§B gap fixes): commit
   `00a1228ca6df39ddb5971e87920d4d0b78913cf5`, receipt
   `freeze/amendment1-commit-hash.txt.ots`, Bitcoin-attested (blocks
   964530/964545/964549).
3. **Amendment 2** (the registered reference ladder — 7 rungs/model,
   bottom = lowest-published = IQ2_M, sourced from IDEA-v1.md §4 +
   plan.md pilot precedent; options (b)/(c)/(d) rejected): commit
   `22fbaba04694b3b3ef9701c00802540208ed18f2`, follow-up `12f882c`,
   receipt `freeze/amendment2-commit-hash.txt.ots` **upgraded 2026-09-04,
   Bitcoin-attested (3 block attestations embedded)**. Note: idea.md v2
   (user's rewrite, committed 2026-08-30 at `claude/IDEA.md`, v1
   preserved at `claude/IDEA-v1.md`) is context, NOT a registration
   source, and Amendment 2 deliberately cites only v1.
4. **The item-set decile bug (OPEN_QUESTIONS §8, RESOLVED):** the first
   0B pass's configs applied PREREG §7's M3 weight vector in prose order,
   but the sampler indexes deciles ascending by popularity — every
   factual_qa cell was measured on a NON-registered item set
   (`ac5cb282…` instead of the registered `2e53ca0e…`). Found while
   investigating an F16 cross-machine delta that turned out NOT to be
   hardware: on identical item sets, Mac and cloud F16 grades agree
   0/500 discordant for both models (evidence:
   `bitcliff/pipeline/analysis/0b/F16_CROSS_MACHINE.md`). User ruled
   option (a): the **0b2 rerun** (2026-09-05, ~55 min, ≈$2.10)
   regenerated factual_qa for all 26 rungs on the registered set —
   both F16 baselines reproduce Amendment 1 §C exactly (Llama 0.314,
   Qwen 0.192). First-pass factual_qa is retained as a **disclosed
   sensitivity run in FINDINGS_0B.md Appendix A** (cliff comparison:
   Llama unchanged; Qwen cliff IQ2_M → Q2_K on the registered set).
   Structural fix: `registered.py` + boot-time item-set hash gates
   (below) make the bug class impossible to repeat silently.
5. **The flag-definition gap (OPEN_QUESTIONS §9, RESOLVED):** PREREG §8's
   "non-monotonic rungs are flagged" was implemented (through two
   analysis passes) as a STATE-sequence flag only; the registered
   sentence's accuracy-level reading (its own IQ-vs-K parenthetical) was
   ruled correct 2026-09-06 and implemented: per-cell
   `acc_inversion`/`inversion_above`, a full FINDINGS listing — **22
   inversions (14 ladder + 8 arm), correcting the pre-implementation
   hand count of 15** (seven arm-Q4_K_M-above-F16 pairs had been
   missed) — with the IQ-vs-K ~2.5 bpw note attached specifically to
   the two IQ2_M-above-Q2_K rows (llama twins, llama factual_qa).
   Verified: zero cell states, cliffs, or Holm verdicts changed.
6. Older resolved items (§1–§7) live in OPEN_QUESTIONS.md with their
   rulings: HF gate, 2b n/seed registration, alias-mapping union,
   packager 4a/4b (+ later positive corpus-provenance check), 8B-class
   longctx out-of-band fallback, ladder rung set (→ Amendment 2),
   bootstrap seed ratification.

## 4. Standing rules (binding for any new session)

- **PREREG.md governs everything. No confirmatory work outside PREREG.**
  Amendments only fill registered slots via the dated-and-stamped
  ceremony; uncovered decisions go to OPEN_QUESTIONS.md and wait for the
  user — never decided silently.
- **Registered constants only from `registered.py`**
  (`bitcliff/pipeline/src/bitcliff_pipeline/registered.py`) — never
  hand-copied. Configs carry literals only because YAML can't import;
  tests cross-check them and the boot-time item-set hash gate refuses
  generation on any non-registered draw (`exploratory: true` skips the
  gate loudly, smoke configs only).
- **Spend gated on explicit approval.** No AWS/GPU spend, no
  timestamping, no pushes, no messages to anyone without the user's
  explicit go. AWS conventions: `bitcliff/.claude/skills/aws-ops/`
  (us-east-1, g6e for GPU, tag transient, sync-to-S3-then-terminate,
  stop-and-ask thresholds). Standing transfer rule: any transfer > 1 GB
  reports rate + ETA within 60 s and stops to ask if projected > 30 min;
  transfer monitors alert on stalled bytes/parts, never process liveness.
- **"stamp it" and "proceed" are distinct triggers.** "stamp it" runs an
  amendment ceremony (append → commit → OTS → follow-up commit);
  "proceed" authorizes a planned spend. Neither implies the other.
- **Credentials never in chat.** HF tokens enter via the user's own SSH
  session only; AWS keys live in the local profile (`bitcliff-agent`);
  never echo, read back, or log either.
- **No eyeball-narration before registered machinery.** Raw accuracies
  are not "findings" until the §8 machinery (states, CIs, Holm) has run;
  never report a curve's story ahead of the registered statistics, and
  label anything exploratory as exploratory.
- Build work runs subagent-driven with reviews (superpowers SDD).

## 5. Open items

**Registered questions Q1–Q5 — analysis status, stated plainly:**
- **Q1 (corpus KLD ranks files?): NOT run.** No corpus-KLD measurements
  exist yet; collecting them is unstarted work.
- **Q2 (at-answer divergence predicts item flips?): data collected,
  analysis NOT run.** All 26 rungs' teacher-forced answer-span NLL
  records exist (`runs-cloud/.../nll/*.jsonl`, 96 items each); the
  registered item-level prediction analysis has not been executed.
- **Q3 (F16 predicts its own fragile answers?): data collected, analysis
  NOT run** (F16 NLL + flip records exist; screening analysis unbuilt).
- **Q4 (contamination direction via twins): partially.** Twin cells are
  in the §8 matrix as ordinary cells, but the REGISTERED within-pair
  original-vs-twin comparison (restricted to pairs where F16 solves
  both, PREREG §3.3) has NOT been run.
- **Q5 (asymmetry transfer to k-quants): NOT run.** Requires the 1.5B
  config-2a comparability run (0B′ scope) plus the cross-suite
  asymmetry analysis; neither exists.
- Also NOT run: the §9 durability comparison page-analysis, and 0C
  (blind check) is unstarted with its `[TO BE FILLED before 0C]` slots
  open in PREREG §10.
- **Qwen-7B uploader shootout: MANDATED by the fired §5 trigger**
  (registered follow-up measurement, "run after the launch analyses").
  Not yet scheduled; needs its own config/manifest work + user go for
  spend.
- **0B′ parked** (1.5B confirmatory rerun + curated-50 regeneration on
  the serving machine, per PREREG; explicitly not started, per user).
- **Site/launch work not started.** `site/` still serves PILOT fixtures
  only (exploratory data, browse-only banners, 50 items × 11 rungs from
  pilot-0a); no confirmatory fixtures exported, no cliff tables,
  methodology page, or share cards. The 0B/0b2 runs have NOT been
  through `package_dataset.py` (only pilot-0a ever was); packaging them
  is open work (note the packager's longctx corpus-provenance gate now
  reads `manifest.json:_run_config`).

## 6. Where everything lives

- **S3** (`bitcliff-artifacts-048568674517`, us-east-1, public access
  blocked, 30-day IA lifecycle): `0b/runs/0b-final-runs.tgz` (sha256
  `98688d3a…`, the complete 0B artifact set), `0b/runs/live/` +
  `0b2/runs/{live,final}` (per-rung syncs), `0b/meta/`, `0b/code/`,
  `0b/tokenizers/` (small assets; the F16 S3 upload was abandoned in
  favor of cloud-side reconstruction).
- **Repo:** run records `bitcliff/pipeline/runs-cloud/pipeline/runs/`
  ({0b,0b2}-*, incl. grades, outputs, nll/, manifests) + fingerprint
  (`runs-cloud/fingerprint.txt` = THE 0B configuration);
  `src/bitcliff_pipeline/registered.py` (registered constants);
  `analysis/0b/cells.csv`, `analysis/0b/FINDINGS_0B.md`,
  `analysis/0b/F16_CROSS_MACHINE.md`,
  `analysis/0b/sweep/cells-seed{1..4}.csv`; `configs/0b/`,
  `configs/0b2/`, `configs/smoke/`; `OPEN_QUESTIONS.md` (repo root —
  §1–§9 all resolved); `PREREG.md` + `freeze/*.ots`; `RUN_0B.md`
  (executed plan); `claude/IDEA.md` (v2, context) + `claude/IDEA-v1.md`
  (freeze-era design doc, Amendment 2's citation source).
- **AWS kept deliberately (user ruling):** AMIs
  `ami-036a0dda7513b8f1c` (staged pre-CUDA) and
  `ami-0dc0c90fcfca46f7c` (CUDA-baked, hash-gated F16s included — the
  fast relaunch path for the shootout follow-up), snapshot
  `snap-037f608f41bbda8b7`, plus each AMI's backing snapshots (small
  monthly storage cost). Key pair `bitcliff-0b`
  (`~/.ssh/bitcliff-0b.pem`); security group `sg-0c62b8cfd39512d55`
  (us-east-1, SSH pinned to the operator's home IP — **the IP rotates;
  update the SG rule on the next session's first SSH failure**).
  Cleanup candidates: unused key pairs + SGs in us-east-2/us-west-2
  (created during the capacity hunt; both regions have 0 G-quota).

## 7. Gotchas that cost real time (carry-forward + new)

- **pkill/pgrep -f self-match:** a pattern that appears in the invoking
  ssh/bash command line matches itself — three separate incidents
  (killed own SSH session twice, false "driver alive" once). Use
  `p[a]ttern` bracket tricks AND keep the target word out of the rest of
  the command, or split into two SSH calls.
- **Transfer/stall monitors watch progress, not liveness** (standing
  rule above). A 100%-CPU NLL phase looks like a stall to a files+GPU
  check (add python CPU%); a 0-part multipart upload looks alive to a
  process check.
- **macOS tarballs:** always `COPYFILE_DISABLE=1`; AppleDouble `._*`
  files broke 45 tests on the cloud box once.
- **uv on the box:** the venv carries a hand-built CUDA llama-cpp-python
  (0.3.35, `CMAKE_ARGS=-DGGML_CUDA=on`, built --no-cache — the cached
  CPU wheel bit once, `gpu_offload_supported` is the tell); never let
  `uv sync`/`uv pip` touch it casually; concurrent `uv run` + `uv pip
  install` deadlock on the project env; use `uv run --no-sync` on the
  box.
- **EBS from AMI lazy-restores:** first read of the baked F16s pulls
  blocks from S3 — the first sha256 pass takes minutes; that's warmup,
  not a hang.
- **HF stale `.lock` files** under `~/.cache/huggingface` can wedge
  dataset loads at 0% CPU; delete and retry.
- **g6e capacity is weather:** stopped instances re-roll capacity on
  start and can strand (lost one box to it). The CUDA-baked AMI +
  try-both-types-across-AZs relaunch loop is the proven recovery; an
  EIP kept the address stable while it existed.
- **PopQA `obj_id` is NOT the Wikidata QID** (QID is in `o_uri`); the
  factual_qa grader has a subject-echo guard (code governs,
  `GRADER_CHARACTERIZATION.md` §7.1); `spectacle_only` rungs never enter
  reference tables (enforced in reporting code); mix draws share one RNG
  stream so item sets differ per mix at the same seed
  (`--print-mix-overlap` reproduces).
