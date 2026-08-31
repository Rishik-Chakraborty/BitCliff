# RUN_0B — Phase 0B confirmatory run plan

**Status: DRAFT FOR USER REVIEW. Nothing in this document runs until the
user says "proceed." No AWS action of any kind has been taken.**

Written 2026-08-29 under the user's 0B leash (one instance ≤ g6e.xlarge,
hard budget $250, stop-and-ask at $100/$175, failure-aware monitors,
instance STOPPED when idle / TERMINATED at completion, sync-back per run,
smoke test first, PREREG governs, no timestamping/pushes/messages).

---

## 1. Scope

**In:** the confirmatory runs PREREG registers for 0B — both reference
ladders (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct), the uploader
shootout (Arm 1), official-vs-community (Arm 2), all four scored suites
per model, and the longctx answer-span divergence measurements.

**Out:** the 1.5B confirmatory rerun (0B′, serving machine, later per
PREREG / user leash); spectacle prompts (1.5B-only); 0C blind check;
anything cross-cell analytical (runs locally afterward from synced data).

## 2. Local prerequisites (no spend) — must complete BEFORE launch

The pre-0B tickets (corpus_sha256 provenance + truncation preflight) are
closed on branch `pre-0b-tickets`. Three build items remain that 0B cannot
run without; all local, $0, subagent-driven with reviews like the tickets:

- **P1 — longctx confirmatory wiring.** `bitcliff_pipeline/__main__.py`'s
  `build_items` does not build `longctx_retrieval` items (the suite's
  builder needs a real tokenizer + verified corpus). Wire a
  `longctx_retrieval` config block (variant, target_tokens, seed, n,
  corpus_path, corpus_sha256, tokenizer path) through `build_items`, with
  the §3.1 tokenizer-match assertion executed before each model's first
  generation (same gate `calibrate_f16.py` already implements — reuse, not
  reimplement). Registered values come from Amendment 1 §C
  (multivalue4 @ t=8192, n=96, seed 2024, PG-1184 corpus) via config files,
  never hardcoded.
- **P2 — answer-span divergence scorer.** The vendored generator already
  emits per-item `nll_input_ids` + `nll_target_mask` (teacher-forcing
  sequence + answer mask). Missing: the scorer that feeds those through
  llama-cpp (logits path), computes per-position answer-span NLL per
  PREREG §3.1's answer-token spec (position p scored from state p−1;
  primary = full 10-token span mean; digits-only 8-position mean as the
  registered sensitivity), and writes per-item records per rung. Needed
  for Q2/Q3 deliverables; must exist before 0B because it runs on the
  same instance + pin as generation.
- **P3 — 0B run configs.** One YAML per run unit (see §5 matrix) pinning
  quant files to the reference manifests' sha256s, registered seeds/n's,
  per-suite budgets, and generation settings (greedy, temp 0, top_k 1,
  seed 42, global 1024). Configs are the registered-parameter carriers;
  a config-vs-PREREG cross-check test accompanies them.

Estimated effort: roughly one working session, SDD-driven. **These need
your go too** — they were not in the Phase-1 ticket list, but 0B cannot
execute without them, so approving this plan approves building them.

## 3. Instance, region, storage, software pins

| item | value |
|---|---|
| Instance | **g6e.xlarge** (1× NVIDIA L40S 48GB, 4 vCPU, 32 GiB RAM) — the leash's ceiling, and sufficient: largest file is a 15GB F16, plus 8k context KV |
| Region | us-east-1 (aws-ops fixed decision) |
| AMI | AWS Deep Learning Base GPU AMI (Ubuntu 22.04), latest in us-east-1 at launch; **exact AMI ID recorded in every run manifest** |
| Storage | 250 GB gp3 EBS (canonical models ~125 GB + headroom); deleted at termination |
| Tags | `project=bitcliff`, `lifecycle=transient`, `phase=0b` (aws-ops rule) |
| Python | project-pinned via uv (same `uv.lock` as local; lockfile synced to instance) |
| llama-cpp-python | **0.3.35** (local pin), built from sdist with `CMAKE_ARGS="-DGGML_CUDA=on"`; the bundled llama.cpp commit is read off the installed build at setup and recorded in every manifest |
| Fingerprint | every output record already carries `platform.platform()` / arch / llama-cpp-python version (PREREG §6); the run manifest additionally records AMI ID, driver + CUDA version, GPU name, llama.cpp commit, and compile flags |

Determinism scope per PREREG §6: all 0B confirmatory generation for the
7B and 8B models happens on this one instance and software pin. If the
instance dies irrecoverably mid-0B, completed per-rung outputs remain
valid (paired per item within rung and model on identical hw/sw) — but
any model whose ladder is split across two boxes is a deviation:
STOP, log to OPEN_QUESTIONS.md, ask.

**Access/authorization state (updated 2026-08-31):** profile
`bitcliff-agent` verified (account 048568674517), `aws` CLI installed and
primary. **F16 conversion pin (added under the §4 revision):** llama.cpp
commit `bf942164697d2d62c2237a17b677dc2c017ea8e7` (the same commit
INHOUSE_QUANTS.md registers), command per PILOT_RUNBOOK.md; cloned at
that exact commit on the instance.

## 4. Data movement (REVISED 2026-08-31 — cloud-side F16 reconstruction)

The original plan (upload the 29 GB of local F16 references to S3 from
home) is DEAD: home bandwidth stalled a multipart upload at zero parts
for 2+ hours (HANDOFF §6 lesson). Replaced by user ruling 2026-08-31:

1. **Small assets via S3 only** — pipeline code tarball, 2b corpus
   (inside it), tokenizer bundles, and the registered F16 hash file.
   All four verified landed in
   `s3://bitcliff-artifacts-048568674517/0b/` (size + MD5/composite-MD5
   match against local, 2026-08-31).
2. **F16 references reconstructed ON the instance:** download the
   official safetensors (`meta-llama/Llama-3.1-8B-Instruct`,
   `Qwen/Qwen2.5-7B-Instruct`; resolved HF revision recorded in the run
   manifest at download time), convert with the pinned llama.cpp commit
   `bf942164697d2d62c2237a17b677dc2c017ea8e7` and the recorded command
   (`convert_hf_to_gguf.py <hf-dir> --outtype f16 --outfile <out>`,
   PILOT_RUNBOOK.md). **Gate: the converted files' sha256 MUST equal the
   registered local hashes — Llama `139c255d857940bc945b2a3242fbbb2641
   b59191d79413bcad9b74fa1784c7b0`, Qwen7B `970ccec3ad83bb62aa25ce585bed
   4ebd297963257442afe697d350465933f2c5` (committed to S3 as
   `0b/meta/f16-sha256.txt`). Any mismatch: STOP the instance, log to
   OPEN_QUESTIONS, ask — never substitute.**
3. **Gated Llama download:** needs the user's HF token, entered by the
   user directly over SSH (never through chat/agent logs): SSH to the
   instance and run `hf auth login` (token stored in
   `~/.cache/huggingface/token`, mode 600). Agent-run commands rely on
   that ambient stored token and never read, echo, or copy it.
4. Quant ladders (~95 GB, registered rungs + arm files only) download
   HF→instance directly, each file sha256-verified against the committed
   reference manifests before use — a mismatch is a hard stop (uploader
   re-quantized → "newer file exists"; STOP instance, log, ask).

**Standing transfer rule (user, 2026-08-31, binding for every transfer
> 1 GB anywhere in 0B):** report observed rate and projected ETA within
60 seconds of starting; if the projection exceeds 30 minutes, STOP the
transfer and ask. Transfer monitors alert on stalled byte/part progress,
never on mere process liveness.

## 5. Run matrix

Registered settings for every cell: greedy, temp 0.0, top_k 1, seed 42;
global 1024-token budget; per-suite: longctx 32, factual_qa 64. Suites
per model (identical item sets across that model's rungs, paired per
item): `longctx_retrieval` 2b (multivalue4 @ t=8192, n=96, seed 2024,
PG-1184), `arithmetic` (GSM8K n=500 seed 3141), `arithmetic_twins` (47
pairs = 94 items, seed 1301, embargoed outputs), `factual_qa` (PopQA
n=500 seed 2718, mix M3, augmented aliases). Truncation preflight +
tokenizer-match assertion run before each model's generation.
**Config 2a (PG-essays, 1.5B-only) is NOT part of 0B** — it belongs to
the 1.5B work.

| run unit | files | suites |
|---|---|---|
| Llama-8B F16 baseline | local conversion (15 GB) | all 4 |
| Llama-8B bartowski ladder | 7 rungs (canonical, see ruling below) | all 4 |
| Shootout Arm 1 (Llama-8B) | 6 files: unsloth Q4_K_M+Q3_K_M, mradermacher static + i1 Q4_K_M+Q3_K_M (bartowski's two come from its ladder run) | all 4 |
| Qwen-7B F16 baseline | local conversion (14 GB) | all 4 |
| Qwen-7B bartowski ladder | 7 rungs (canonical) | all 4 |
| Arm 2 official (Qwen-7B) | Qwen official q4_k_m (2 shards) + q3_k_m | all 4 |
| Divergence pass | every rung above incl. F16 | longctx answer-span NLL (P2 scorer) |

**Ladder rung lists — RULED 2026-08-30 (OPEN_QUESTIONS §6, option (a)),
registered via Amendment 2, PENDING STAMP.** The registered rule: per
reference model, F16 (local baseline) + Q8_0, Q6_K, Q5_K_M, Q4_K_M,
Q3_K_M, Q2_K + the lowest rung the tracked uploader publishes below
Q2_K — which the pinned bartowski manifests resolve to **IQ2_M for both
models** (IQ2_XXS/IQ1_S absent from both; idea.md §4's "expected around
IQ2_XXS" unmet by the actual catalogs; the lowest-published rule, not the
expectation, governs). Full text, sources quoted verbatim, and rejected
alternatives: `AMENDMENT2_DRAFT.md`. **Amendment 2 is appended to PREREG
and OTS-stamped BEFORE any confirmatory generation** — the ladder is the
row-definition of every reference table.

**Cell count (registered ladder):** 2 models × 7 quant rungs × 4 scored
suites = **56 confirmatory reference cells**, + 32 arm-scoped cells
(8 arm files × 4 suites) outside the reference matrix. PREREG §8's dual
rule applies unchanged (per-cell descriptive at α=0.05; Holm over any
cross-cell headline's family). Options (b)/(c) (160 / 80–88 cells) and
(d) (in-house bottom rungs) rejected per the ruling.

**Totals: 22 quant-rung generation runs (14 ladder + 6 shootout + 2
official) + 2 F16 baselines + 24 divergence passes.** ~1,190 scored items
per rung.

## 6. Order of runs

1. **Setup** (~1–2 h): launch, mount, pull code + S3 assets, build
   llama-cpp-python CUDA, record fingerprint, verify F16 sha256s.
2. **Smoke test (leash rule):** 20 items, one rung (Llama-8B Q4_K_M), one
   suite (factual_qa), end-to-end: generation → grading → manifest
   (with `_run_config` + corpus provenance) → truncation preflight →
   tokenizer-match assertion. Verify outputs by eye + tests. **Report
   smoke result to you before the full ladders start** (cheap checkpoint,
   instance STOPPED while waiting unless you pre-authorize continuing).
3. **Llama-8B:** F16 baseline → ladder top-down (Q8_0 → Q6_K → Q5_K_M →
   Q4_K_M → Q3_K_M → Q2_K → IQ2_M) →
   shootout files → divergence passes (per rung, immediately after its
   generation, while the file is loaded).
4. **Qwen-7B:** F16 baseline → ladder → official arm → divergence.
5. Each rung ends with `s3 sync` of its outputs + manifests (sync-back is
   part of the run, per leash). Grading/reports can run on-instance
   (cheap CPU) or locally after sync — grading locally is preferred; only
   generation and NLL scoring need the GPU.
6. **Teardown:** final S3 sync, pull everything to this machine, verify
   counts + hashes, commit locally, then TERMINATE (never stop-and-forget).

Budget checkpoints: at **$100** and **$175** cumulative estimated spend I
stop, report spend + remaining work, and wait for your ack (leash).
Projected: $100 lands around the end of the Llama ladder — a natural
pause point (instance STOPPED while waiting).

## 7. Monitoring (the 7B-stall lesson)

Every run executes under a failure-aware monitor watching for **both**
progress lines and failure signatures (`Traceback|Error|CUDA|OOM|Killed|
assert|core dumped`), plus a liveness check (process alive via `ps`, new
output bytes within 30 min). Any stall > 30 min with no new output: kill
the job, snapshot state (S3 sync + notes), **STOP the instance**, report
to you. Any surprise or PREREG ambiguity: same — stop the meter first,
then log to OPEN_QUESTIONS.md, then ask. The instance is STOPPED whenever
no job is executing (incl. while waiting for your budget acks).

## 8. Time and cost estimate

Timing anchors: local-machine F16 calibration walls (longctx mv4@8192
n=96 ≈ 37 min; factual_qa 500 ≈ 4–7 min); arithmetic estimated from
answer lengths (~250 tok avg × 500 items). L40S single-stream assumed
≈ 1–2.5× local Mac for decode, faster for prefill — estimates stay at the
conservative end.

| block | estimate |
|---|---|
| Setup + CUDA build + smoke | 2–3 h |
| F16 reconstruction on-instance (safetensors ~30 GB download + 2 conversions + hash gate) | 1–1.5 h |
| Downloads (~95 GB HF→EC2, canonical rungs + arms only) | ~1 h (overlaps setup) |
| 22 quant rungs × ~40–60 min | 15–22 h |
| 2 F16 baselines × ~1–1.5 h | 2–3 h |
| 24 divergence passes × ~3–5 min | 1.5–2 h |
| **Total GPU-instance hours** | **≈ 22–32 h** (midpoint ~27 h) |

**COST ESTIMATE (revised 2026-08-31 for cloud-side F16 reconstruction;
price verified from AWS's published on-demand map): ~27 GPU-hours ×
$1.8610/h (g6e.xlarge us-east-1 on-demand) ≈ $50; range $41–60 for
22–32 h; plus ~$5 EBS+S3 (S3 now carries only ~110 MB of small assets) ⇒
≈ $46–65 total, against the $250 hard cap.**
Consistent with IDEA.md §12's ~$150 allocation for confirmatory runs.
If the realized per-rung time trends above the range after the first
three rungs, I extrapolate, report, and wait at the $100 checkpoint
rather than discovering it at $175.

## 9. Decisions — RULED 2026-08-30

1. **Ladder rung lists (§5): RULED 2026-08-30 — OPEN_QUESTIONS §6 option
   (a), registered via Amendment 2 (AMENDMENT2_DRAFT.md), which must be
   appended + OTS-stamped before any confirmatory generation.** §5's
   matrix, §8's hours, and the cost line are final against this ladder.
2. **Local prerequisites (§2): GO** — P1–P3, local, $0, SDD with
   reviews; the P2 NLL scorer implements PREREG §3.1's registered Q2
   definition (teacher-forced on the full-precision trajectory) with
   tests asserting exactly that.
3. **Smoke-test checkpoint (§6.2): CONFIRMED** — stop and report after
   the smoke test.
4. **AWS access: DONE** — profile `bitcliff-agent` verified via
   `aws sts get-caller-identity` (account 048568674517); CLI now
   installed and working, used as primary (leash applies identically).
5. **The cost line (§8): revised to ≈ $45–65** under the canonical
   ladder. Nothing launches until the explicit "proceed" — which per the
   2026-08-30 ruling comes after: this ladder ruling (done) + revised
   cost (done) + P1–P3 complete + credentials confirmation (done).
