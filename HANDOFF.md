Push status (2026-10-03): HEAD = origin/main = the "handoff: rotation before 0B-prime" commit (parent `87092f0`), ahead/behind 0/0 — verify with `git rev-list --left-right --count main...origin/main`.
# HANDOFF — BitCliff session handoff (rewritten 2026-09-07; updated 2026-10-03)

For a fresh Claude Code session with no memory. This is an index, not an
archive: every claim below is verifiable in the named files, git history,
or S3. Read this first; PREREG.md governs; nothing spends, stamps, pushes,
or messages anyone without the user's explicit go.

## 1. Repo state (as of this handoff's commit)

Branch list: `main` only (work branches `pre-0b-tickets`, `0b-analysis`,
`factualqa-rerun`, `code-turn-0b-prime`, `code-turn-nll-driver` were
merged no-ff and deleted). State is current through `87092f0` (merge of
`code-turn-nll-driver`, 2026-09-28), pushed; nothing local-only. Always
check `git rev-list --left-right --count main...origin/main` first. Test
suite: 592 passing (`uv run pytest -q` in `bitcliff/pipeline/`, ~14 s; run
with `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1` —
`tests/test_registered.py` and the 0B-prime config tests rebuild item sets
from the local HF cache and the HF tokenizers under `models/hf/`, so a
cold machine needs network or a warm cache).

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
   964240/964295/964307).
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
6. **Amendment 3 (§5 shootout trigger fired):** the trigger fired
   2026-09-03 (`be3b07d`/`b2f29c1`), held on registered data and all sweep
   seeds; PREREG §15.5 makes a firing an amendment. Appended 2026-09-08 at
   commit `9f94913d322b009b214e2bfde80f1217e9d1da17`, receipt
   `freeze/amendment3-commit-hash.txt.ots`, Bitcoin-attested (block
   headers 966021/966024/966041; upgraded 2026-09-10). §C quotes §5's consequence
   clause byte-identically and commits the Qwen-7B shootout file manifest to
   its own stamped section before any run. OPEN_QUESTIONS §12 inventories
   what §5 fixes by inheritance and what that manifest section must fix
   (uploader set, files/hashes/enumeration date, machine, bartowski reuse,
   Holm membership).
7. **Amendment 4 (the 1.5B confirmatory ladder):** the slot Amendment 2 §F
   reserved. Amendment 2 §C's rule applied to
   `bartowski/Qwen2.5-1.5B-Instruct-GGUF` at revision `9eadc661…`: F16
   (local, `954b4492…`) + Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K, IQ2_M
   (lowest published below Q2_K; no IQ2_XXS/IQ1 published). Manifest
   `reference-manifests/qwen2.5-1.5b-bartowski.json`; in-house spectacle
   files pinned separately in `qwen2.5-1.5b-inhouse.json`, excluded.
   Appended 2026-09-12 at `3d793e591612082d92fa55f4c64b71416dc33b3a`,
   receipt `freeze/amendment4-commit-hash.txt.ots`, Bitcoin-attested
   (966743/966744/966764; upgraded 2026-09-14). Gated append: rule quote
   byte-identical, every hash re-read from HF at stamp time, content grep
   for run-plan terms outside §F returned zero.
8. **Post-freeze PREREG body edits (OPEN_QUESTIONS §11):** exactly two
   body lines differ from the freeze commit — line 3 (status header,
   `0285fcd`, 2026-09-07) and line 570 (§7 slot pointer, made in `00a1228`).
   Every append is gated on `git diff 5e6882b7 HEAD -- PREREG.md` showing
   only those two body hunks plus the receipt/appendix hunk at 790.
9. **Official-Qwen imatrix label (OPEN_QUESTIONS §10, RESOLVED
   2026-09-08):** the config's `imatrix: false` for Arm 2 was asserted
   without evidence at run time; the GGUF headers of both pinned official
   files (range-fetched, KV block parsed) carry no `quantize.imatrix.*` key.
   Evidence `reference-manifests/evidence/qwen2.5-7b-official-gguf-headers.md`;
   footnote in FINDINGS_0B.md's Arm 2 section (inside the generated body —
   re-add after any regeneration). The seven bartowski 1.5B rungs were
   checked the same way and all carry the keys
   (`…/evidence/qwen2.5-1.5b-bartowski-gguf-headers.md`).
10. Older resolved items (§1–§7) live in OPEN_QUESTIONS.md with their
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
- **Qwen-7B uploader shootout: MANDATED by the fired §5 trigger**, now
  recorded as Amendment 3 (stamped). Before it can run: a stamped
  `shootout-7b.json` manifest section fixing the items OPEN_QUESTIONS §12
  lists, then a user go for spend. "Run after the launch analyses."
- **0B′ (1.5B confirmatory run): ladder registered (Amendment 4), run
  plan drafted (`RUN_0B_PRIME.md`), NOT started.** Plan readings confirmed
  2026-09-27 (OPEN_QUESTIONS §13): twins run for the 1.5B; the 2a NLL pass
  is registered Q2 data, not exploratory. §8's rate corrected to
  $2.2421/h; cap set to $50. Of RUN_0B_PRIME.md §9's pre-"proceed" list,
  done this code turn: 2a item-set hash in `registered.py` (`05537fe`);
  `configs/0b-prime/` ladder + smoke configs and cross-check tests
  (`fca1a25`); the runner's per-item timing field (`1b6f418`, legacy-load
  fix `9546a93`); the reproduction-gate script (`7e9db50`); the item-set
  hash gate's n-mismatch fix (`bf6ac1f`). The NLL driver that produced
  0B's `nll/*.jsonl` records was not recoverable — the S3 `0b/code/`
  prefix holds only a 2026-08-30 snapshot predating the NLL runs, and no
  driver exists in git history (OPEN_QUESTIONS §16) — so a new one was
  **written** (`scripts/nll_driver.py`, commit `ba146e7`, against
  `nll_scorer.score_records`) **and reviewed; pending the box NLL
  reproduction gate** (RUN_0B_PRIME.md §4, second gate) before the §3 NLL
  passes can run. Machine ruling: the 0B
  fingerprint / CUDA AMI.
  **Code prerequisites complete through `87092f0`; the box procedure lives
  in `RUN_0B_PRIME.md`:** §1 pre-launch statement; §3 the single fixed
  1.5B NLL invocation (after the generate stage, same `--run-id`; the
  driver refuses a non-exploratory run without that run's
  `manifest.json`); §4 gate 1 (`scripts/repro_gate.py`, Qwen-7B Q4_K_M,
  first 25 longctx items, empty diff) and gate 2 (`scripts/nll_driver.py`
  `--allow-no-manifest`, Qwen-7B and Llama-8B Q4_K_M on 2b, `cmp` exact
  match vs committed `nll/Q4_K_M.jsonl`; any mismatch → stop, report, and
  the ruling is regenerating all 26 NLL passes); the per-run
  `fingerprint.txt` step before sync-back (required by `analyze_0b.py` for
  non-0B run sets); every box command is `uv run --no-sync`; §8 cost
  table at $2.2421/h (≈ $12–19 without benchmark, ≈ $16–25 with), cap $50.
  **Open: the two-rung NLL reproduction gate on the box** (nothing has
  run; no "proceed" given). Risk to weigh before "proceed": the Llama leg
  may mismatch — the pre-fix Llama/arm1 n_ctx and all unlogged llama-cpp
  constructor args are unrecoverable (OPEN_QUESTIONS §16; the NLLCFG_*
  log lines carry no settings) — and a full 26-pass regeneration would
  exceed the $50 0B′ cap.
- **Parked items (ruled, deferred; none blocks the box run):**
  no committed box command captures `llama_cpp_commit` for
  `fingerprint.txt` (operator records it from the build); the Llama-8B
  Q4_K_M gate download is identified by its pinned sha256 only
  (`ensure_quants` passes no HF revision); `nll_driver.py` has no
  GGUF-vs-HF tokenizer-match gate of its own (covered for the 1.5B by the
  same-`--run-id` manifest requirement after generation; for the gate
  rungs by the `model_sha256` field match), a small resume crash window
  (jsonl written before its manifest entry), and a manifest that records
  module-constant `logits_all`/`n_gpu_layers` even under an injected
  factory; `analyze_0b.py` hardcodes "7 ladder cells" in the Holm
  preamble (correct for every current ladder), supports shootout arms for
  the 0B run set only (the 7B shootout needs its own code turn), checks a
  missing sweep dir only after `run_analysis`, and keeps byte-identical
  0B prose whose wording is stale (the two "re-append after any FINDINGS
  regeneration" HTML comments; "all 8 cliff rungs" counts families);
  `scripts/repro_gate.py`'s `_read_output_dicts` is now redundant with
  `generate.read_records`; the 1.5B FINDINGS fingerprint prose has no
  committed target yet.
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
  §1–§10 resolved; §11 and §12 are disclosures; note the file has two
  sections numbered §6, the imatrix one was renumbered §10);
  `PREREG.md` (freeze + Amendments 1–4) + `freeze/*.ots` (four receipts,
  all Bitcoin-attested; `.ots.bak` files are the pre-upgrade receipts);
  `AMENDMENT{,2,3,4}_DRAFT.md` (reviewed drafts, superseded banners);
  `reference-manifests/` (8B ladders, official Qwen, shootout-8b,
  `qwen2.5-1.5b-bartowski.json`, `qwen2.5-1.5b-inhouse.json`) and
  `reference-manifests/evidence/` (GGUF-header imatrix evidence);
  `RUN_0B.md` (executed plan); `RUN_0B_PRIME.md` (0B′ plan, working
  doc); `docs/superpowers/plans/` (historical plans, audit trail);
  `claude/IDEA.md` (v2, context) + `claude/IDEA-v1.md` (freeze-era design
  doc, Amendment 2's citation source).
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
