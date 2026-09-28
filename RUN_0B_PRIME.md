# RUN_0B′ — 1.5B confirmatory run plan (working doc, 2026-09-08)

Working document, not registration. PREREG.md governs; Amendment 4 pins the
1.5B ladder — stamped 2026-09-12 (commit `3d793e5`), Bitcoin-attested
2026-09-14 (blocks 966743/966744/966764). Every ruling below is labeled with its PREREG
anchor where one exists and **exploratory** where none does. Nothing here
authorizes spend; a box is launched only on the user's explicit "proceed".

## 1. Machine

**Ruling:** 0B′ runs on the 0B fingerprint — `runs-cloud/fingerprint.txt`
(g6e.2xlarge, NVIDIA L40S, CUDA 13.2, llama.cpp commit `bf942164…`,
llama-cpp-python 0.3.35 CUDA build, Python 3.11.16), launched from the
pinned CUDA-baked AMI `ami-0dc0c90fcfca46f7c` (HANDOFF §6; base AMI
`ami-036a0dda7513b8f1c`), us-east-1, per `bitcliff/.claude/skills/aws-ops/`.

**Pre-launch statement (aws-ops rule):** before any launch, the instance
type, the $/h rate, and the expected duration are stated to the user.

**Anchor:** PREREG §6 lines 499–504 scope determinism to a fixed hardware
and software configuration and require every record to carry the
fingerprint. The 1.5B F16 calibration values (Amendment 1 §C, lines
968–1003) were measured on the Mac; the F16 cross-machine check
(`analysis/0b/F16_CROSS_MACHINE.md`) found 0/500 discordant grades between
Mac and cloud on identical item sets for the 7B and 8B, so a Mac-vs-cloud
F16 delta is not expected but is checked by the gate in §4.

**§6's serving-machine sentence** ("The curated playground prompts are
regenerated on the exact serving machine", line 503) applies to the curated
playground prompts and is the Phase 2 reproduction gate for the live box.
It does not bind 0B′, which produces reference measurements, not the
playground fixtures.

## 2. Scope

**Ladder:** the Amendment 4 ladder — F16 (local, sha256 `954b4492…`) +
Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K, IQ2_M from
`reference-manifests/qwen2.5-1.5b-bartowski.json`. **In-house rungs
(IQ2_XXS, IQ1_M, IQ1_S): not run.** They are `spectacle_only` (PREREG §4
lines 441–448) and enter no reference statistic; regenerating them belongs
to the playground-fixture work, not to 0B′.

**Suites the registered text assigns to the 1.5B:**

| suite | registered for the 1.5B? | anchor |
|---|---|---|
| longctx_retrieval, configuration 2a | yes — "Q5 comparability run (Qwen2.5-1.5B-Instruct only)": PG-essays corpus sha `b6135331…`, multivalue2, target_tokens 4096, depths (0.1, 0.5, 0.9), seed 2024, n=96 | §3.1 lines 122–142 |
| longctx_retrieval, configuration 2b | yes — calibrated for the 1.5B: multivalue4 @ 8192, F16 0.750 in-band, item set `ed18db13…` | §3.1 lines 144–163; Amendment 1 §C lines 968, 979–1003 |
| arithmetic | yes — n=500, seed 3141, no model restriction | §3.2 lines 204–217 (207–208) |
| arithmetic_twins | registered generically: §3.3 (lines 219–266) registers the suite and its 94 pair items with no model restriction, so nothing in the text excludes the 1.5B; no 1.5B-specific registration exists because the suite has no calibration knob. **Reading: it runs — confirmed by the user 2026-09-27** (OPEN_QUESTIONS §13). | §3.3 lines 247–262 |
| factual_qa | yes — calibrated for the 1.5B: mix M3, F16 0.110 out-of-band disclosed, item set `2e53ca0e…` | §3.4 lines 268–298; Amendment 1 §C lines 968–977 |

Twins and factual_qa are therefore registered for the 1.5B and run.

**Machinery:** §8 per-cell inference (margin 0.03, McNemar exact, paired
bootstrap 10,000; lines 583–612); boot-time item-set hash gates from
`registered.py` (`2e53ca0e…` factual_qa, `ed18db13…` longctx 2b for the
1.5B, derived gates for arithmetic and twins; 2a's item-set hash
`5404e813…` is pinned in `registered.py` (commit `05537fe`; first-20 digest
verified)); bootstrap seed
8271 with the ratified per-cell and per-pair string rules (OPEN_QUESTIONS
§7). The 2a tokenizer-match assertion (§3.1 lines 111–120) and the
first-20-item digest `9220589b…` (§3.1 lines 130–135) must pass before the
first 2a generation.

**Item count per model load:** 2a 96 + 2b 96 + arithmetic 500 + twins 94 +
factual_qa 500 = 1286 scored items; 8 loads (F16 + 7 rungs).

## 3. NLL passes (registered Q2 data for the 1.5B)

Run the teacher-forced answer-span NLL scorer (`nll_scorer.py`) over the 2a
**and** 2b items for all 8 loads, producing both registered aggregations —
the full-span primary mean and the digits-only sensitivity mean — from the
one forward pass per item (PREREG §3.1 lines 186–195). **Label: registered
Q2 data for the 1.5B** (user ruling 2026-09-27, OPEN_QUESTIONS §13): the
§3.1 spec carries no model restriction. Q2's analysis population is fixed
in the Q2 pre-specification before any Q2 analysis runs.

Note: §3.1's answer-token spec is written against the Qwen2.5 tokenizer's
10-token `"v1, v2"` span, which is the 2a/paper configuration. An earlier
draft of this plan labeled the pass exploratory; the ruling above
supersedes that label.

## 4. Reproduction gate at box start

Before any 0B′ item is generated: re-run one 0B Qwen-7B rung on its first
25 longctx items with the recorded settings and diff against the committed
run. Concretely:

- Rung: `0b-qwen-7b-ladder` / Q4_K_M (file sha256 per
  `qwen2.5-7b-bartowski.json`; the F16 is baked into the AMI, the quant
  re-downloads).
- Items: `longctx_retrieval-multivalue4-t8192-s2024-0000` … `-0024`, built
  by the same generator from the pinned PG-1184 corpus (sha `0a21a138…`),
  identical `items.jsonl` ids asserted.
- Settings: from `runs-cloud/pipeline/runs/0b-qwen-7b-ladder/manifest.json`
  `_run_config` (greedy, temperature 0.0, top_k 1, seed 42, max_tokens 32,
  n_ctx 16384, multivalue4 @ 8192).
- Compare: `outputs/Q4_K_M.jsonl` `text` and `finish_reason`, and
  `grades.jsonl` `state`/`truncated`, for those 25 ids, against the
  committed files. **The diff must be empty.** Any difference: stop, sync
  the 25 records to S3, report, do not start 0B′.

**Second gate — NLL reproduction, before any 1.5B NLL pass (§3):**
reproduce 0B's teacher-forced NLL records for two rungs on their 2b items,
with `scripts/nll_driver.py` (commit `ba146e7`):

```
uv run python scripts/nll_driver.py configs/0b/0b-qwen-7b-ladder.yaml \
    --run-id 0b-qwen-7b-ladder-nllgate --runs-dir runs --models-dir models \
    --suites longctx_retrieval --labels Q4_K_M
uv run python scripts/nll_driver.py configs/0b/0b-llama-8b-ladder.yaml \
    --run-id 0b-llama-8b-ladder-nllgate --runs-dir runs --models-dir models \
    --suites longctx_retrieval --labels Q4_K_M
```

(Qwen-7B Q4_K_M is the rung already on the box for the first gate above;
Llama-8B Q4_K_M re-downloads.) Write into scratch run dirs
(`0b-qwen-7b-ladder-nllgate`, `0b-llama-8b-ladder-nllgate`) — never over the
committed `runs-cloud/` records. Compare per item, field for field, against
the committed `runs-cloud/pipeline/runs/0b-qwen-7b-ladder/nll/Q4_K_M.jsonl`
and `runs-cloud/pipeline/runs/0b-llama-8b-ladder/nll/Q4_K_M.jsonl`. **Exact
match required** on every `NLLRecord` field (`item_id`, `suite`,
`quant_label`, `model_sha256`, `per_token_nll`, `span_mean_nll`,
`digits_only_mean_nll`, `n_answer_tokens`, `machine`) except `machine`,
which is identical only if the box fingerprint matches the 0B fingerprint
(§1); a `machine` difference alone means the box is not the 0B fingerprint
— stop, that is a fingerprint mismatch, not an NLL-gate failure. Any other
field mismatch: stop, report; the ruling is regenerating all 26 NLL passes.

Provenance caveat (OPEN_QUESTIONS §16): the committed Llama-8B ladder NLL
records were produced by the pre-fix provider (before commit `7364672`,
"NLL provider reads llama scores array, not eval_logits") with an
n_ctx unrecoverable from the 0B log; the committed Qwen-7B ladder records
were produced by the post-fix provider with n_ctx 8448 (`ceil(8263/256)*256`,
`runs-cloud/0b-nll.log`). `nll_driver.py` always uses the post-fix
`llm.scores` path and leaves every other llama-cpp-python constructor
argument (n_batch/n_ubatch, flash-attention, ...) at the 0.3.35 default, so
the Llama-8B leg of this gate is the only test of whether that provider
path reproduces the pre-fix numbers; a pass on Qwen-7B alone does not cover
it.

## 5. Timing

No run to date has recorded generation timing: neither pilot-0a, 0b, nor 0b2
output records carry a per-item duration or tokens-per-second field (checked
2026-09-07; the only timing evidence is output-file mtimes). Before 0B′ the
runner gains a per-item timing field (wall seconds and generated-token
count per record, plus a per-rung load time) — a separate prompt, code
change, tests. Every 0B′ record carries it. Until then the cost table below
is an estimate from mtimes.

## 6. Optional: throughput benchmark (product data, not science)

Both 8B ladders (Llama-3.1-8B, Qwen2.5-7B: 14 quant files + 2 F16), one
fixed prompt set (e.g. the 25 gate items plus 25 fixed arithmetic items),
tokens per second per file on the fingerprint, greedy, same n_ctx as 0B.
Output: one CSV per model, `speed_tok_s` per file, for the Numbers page's
speed column (idea.md §5). Enters no registered statistic. Runs only if the
user includes it in the "proceed".

## 7. Placeholder: Q1 corpus KLD

Nothing runs here until a pre-specification exists and is stamped: the
generic-text corpus and its pin, token count, the KLD definition and
aggregation, the ranking statistic, and the capability-damage ordering it
is compared against (PREREG §2 lines 46–48 register the question only).
Filled only from that stamped section.

## 8. Cost table

Rate: **$2.2421/h**, the g6e.2xlarge on-demand rate (us-east-1), the
instance type 0B actually ran on (`runs-cloud/fingerprint.txt`); corrected
from an earlier `$2.24/h` rounding (user-confirmed 2026-09-27). The dollar
columns below are left as computed at $2.24/h — the difference is at most
2.1 cents on any row, at the 10 h total. (An earlier draft used RUN_0B.md
§8's $1.8610/h, which names g6e.xlarge.) GPU-hour figures derive from 0B
output-file mtimes (8B:
16–35 min per 4-suite rung) scaled 3–4× for a 1.5B, HANDOFF's ~55 min for
the 26-rung 0b2 factual_qa pass, and the 0B NLL log timestamps
(`runs-cloud/0b-full.log`). NLL timing note: 0B's 2b NLL passes ran ≈ 100
min per load on Llama-8B before the provider fix (`7364672`, reads
`llm.scores`) and ≈ 6.8 min per load on Qwen-7B after it (8 loads in 54.3
min, 19:21:59–20:16:16); the 2a row keeps its pre-fix derivation and is
likely conservative. Rough.

| item | GPU-hours | $ at 2.24/h |
|---|---|---|
| Setup: launch from CUDA AMI, 7 quant downloads (~7 GB), F16 obtain/convert + hash gates, smoke | 1.0–1.5 | 2.24–3.36 |
| Reproduction gate (§4): Qwen-7B Q4_K_M download + 25 longctx items + diff | 0.25–0.5 | 0.56–1.12 |
| NLL reproduction gate (§4, second gate): Llama-8B Q4_K_M download (~4.9 GB) + two gate NLL passes (Qwen-7B, Llama-8B; 2b items; post-fix 0B timing ≈ 6.8 min per 7B-class load at t=8192) | 0.25–0.5 | 0.56–1.12 |
| Q5 suites only (2a longctx + arithmetic, 8 loads, 596 items each) | 0.5–0.7 | 1.12–1.57 |
| Full registered scope (§2: 1286 items × 8 loads) | 1.0–1.5 | 2.24–3.36 |
| NLL passes on 2a (§3: 8 loads × 96 items at t=4096, logits_all) | 2.0–3.5 | 4.48–7.84 |
| NLL passes on 2b (§3: 8 loads × 96 items at t=8192; 0B's post-fix 6.8 min/load for the 7B scaled 3–4× down, plus load overhead) | 0.25–0.5 | 0.56–1.12 |
| Optional benchmark (§6: 16 files, ~1 h re-downloads + ~5 min each) | 2.0–2.5 | 4.48–5.60 |
| EBS (AMI-backed gp3 volume for the run's hours) + S3 puts | — | ≈ 1.00 |
| **Total, full scope + NLL (2a + 2b), no benchmark** | **4.75–8.0** | **≈ 12–19** |
| **Total, with benchmark** | **6.75–10.5** | **≈ 16–25** |

(The Q5-suites row is a subset of the full-scope row and is not added into
the totals.)

Cap: **$50 hard stop, benchmark included. Burn reports at 50/80/100 of cap.
Set by user 2026-09-27.**

## 9. Before "proceed" — what must be committed

1. Amendment 4 appended and stamped (the 1.5B ladder pins). **Done:**
   stamped 2026-09-12 (commit `3d793e5`), Bitcoin-attested 2026-09-14
   (blocks 966743/966744/966764).
2. `registered.py`: 2a item-set hash for the 1.5B added and gate-tested.
   **Done:** commit `05537fe` (registered.py through Amendment 4 — 1.5B
   pins, 2a item-set hash `5404e813…`, first-20 digest `9220589b…`
   verified).
3. `configs/0b-prime/qwen2.5-1.5b-ladder.yaml` (+ a smoke config), with
   `tests/test_0b_prime_configs.py` cross-checking every filename and
   sha256 against `qwen2.5-1.5b-bartowski.json` and every seed/n against
   `registered.py`; M3 weights in convention order. **Done:** commit
   `fca1a25` (ladder + smoke configs; 2a as suite key
   `longctx_retrieval_2a`; declarative `nll:` block).
4. Runner timing field (§5) implemented and tested. **Done:** commit
   `1b6f418` (per-item `gen_wall_seconds`/`gen_tokens`), plus fix
   `9546a93` (timing fields default `None` so legacy output records load).
5. The reproduction-gate script (§4) with its expected-output fixture.
   **Done:** commit `7e9db50` (`scripts/repro_gate.py` + 0B Qwen-7B
   Q4_K_M longctx fixture).
6. This document's rulings confirmed. **Done 2026-09-27:** §2's twins
   reading (runs), §3's label (registered Q2 data, 2a and 2b), §8's rate
   and cap.
7. The NLL driver recovered: the script that produced 0B's `nll/*.jsonl`
   records is not in the repo (OPEN_QUESTIONS §16); recover it from the S3
   `0b/code/` snapshot, record its hash, and commit it before the §3 passes.
   **NOT done.** The S3 `0b/code/` prefix
   (`s3://bitcliff-artifacts-048568674517/0b/code/`) holds exactly one
   object, `pipeline-code.tgz` (105.5 MiB, uploaded 2026-08-30) — a
   snapshot predating the NLL runs, so no driver is in it; none exists in
   git history either. A new driver must be written against
   `nll_scorer.score_records`, with its own review, not recovered.
8. Item-set hash gate n-mismatch fix: `assert_item_sets_match_registered`
   (`src/bitcliff_pipeline/__main__.py` line 65) skips any suite whose item
   count differs from its registered n, so a mistyped n goes ungated. A
   confirmatory config must fail loudly on that instead of skipping.
   **Done:** commit `bf6ac1f` (item-set gate fails loudly on
   non-registered n).
