# RUN_0B′ — 1.5B confirmatory run plan (working doc, 2026-09-08)

Working document, not registration. PREREG.md governs; Amendment 4 (draft,
`AMENDMENT4_DRAFT.md`) pins the 1.5B ladder and must be stamped before any
confirmatory 1.5B generation. Every ruling below is labeled with its PREREG
anchor where one exists and **exploratory** where none does. Nothing here
authorizes spend; a box is launched only on the user's explicit "proceed".

## 1. Machine

**Ruling:** 0B′ runs on the 0B fingerprint — `runs-cloud/fingerprint.txt`
(g6e.2xlarge, NVIDIA L40S, CUDA 13.2, llama.cpp commit `bf942164…`,
llama-cpp-python 0.3.35 CUDA build, Python 3.11.16), launched from the
pinned CUDA-baked AMI `ami-0dc0c90fcfca46f7c` (HANDOFF §6; base AMI
`ami-036a0dda7513b8f1c`), us-east-1, per `bitcliff/.claude/skills/aws-ops/`.

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
| arithmetic_twins | registered generically: §3.3 (lines 219–266) registers the suite and its 94 pair items with no model restriction, so nothing in the text excludes the 1.5B; no 1.5B-specific registration exists because the suite has no calibration knob. **Reading: it runs.** Flagged for the user to confirm, since the twin comparison's own registered analysis (both-solved filter, two-sided) is model-agnostic. | §3.3 lines 247–262 |
| factual_qa | yes — calibrated for the 1.5B: mix M3, F16 0.110 out-of-band disclosed, item set `2e53ca0e…` | §3.4 lines 268–298; Amendment 1 §C lines 968–977 |

Twins and factual_qa are therefore registered for the 1.5B and run. If the
user rules the twins reading wrong, that suite is dropped; nothing else
changes.

**Machinery:** §8 per-cell inference (margin 0.03, McNemar exact, paired
bootstrap 10,000; lines 583–612); boot-time item-set hash gates from
`registered.py` (`2e53ca0e…` factual_qa, `ed18db13…` longctx 2b for the
1.5B, derived gates for arithmetic and twins; 2a's item-set hash is not yet
pinned in `registered.py` and must be added before the run); bootstrap seed
8271 with the ratified per-cell and per-pair string rules (OPEN_QUESTIONS
§7). The 2a tokenizer-match assertion (§3.1 lines 111–120) and the
first-20-item digest `9220589b…` (§3.1 lines 130–135) must pass before the
first 2a generation.

**Item count per model load:** 2a 96 + 2b 96 + arithmetic 500 + twins 94 +
factual_qa 500 = 1286 scored items; 8 loads (F16 + 7 rungs).

## 3. NLL passes (exploratory for the 1.5B)

Run the teacher-forced answer-span NLL scorer (`nll_scorer.py`) over the 2a
items for all 8 loads, producing both registered aggregations — the
full-span primary mean and the digits-only sensitivity mean — from the one
forward pass per item (PREREG §3.1 lines 186–195). **Label: exploratory for
the 1.5B.** Note for the user: §3.1's answer-token spec is written against
the Qwen2.5 tokenizer's 10-token `"v1, v2"` span, which is the 2a/paper
configuration, so a ruling that this pass is the registered Q2 measurement
for the 1.5B rather than exploratory would be defensible; this plan keeps
the label the user set.

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

Rate: **$1.8610/h**, the RUN_0B.md §8 line 229 figure. Caveat: that line
names g6e.xlarge, while 0B actually ran on g6e.2xlarge (fingerprint); if the
2xlarge on-demand rate applies, multiply the dollar column by its ratio.
GPU-hour figures derive from 0B output-file mtimes (8B: 16–35 min per
4-suite rung; NLL ≈ 100 min per rung at t=8192) scaled 3–4× for a 1.5B, and
HANDOFF's ~55 min for the 26-rung 0b2 factual_qa pass. Rough.

| item | GPU-hours | $ at 1.8610/h |
|---|---|---|
| Setup: launch from CUDA AMI, 7 quant downloads (~7 GB), F16 obtain/convert + hash gates, smoke | 1.0–1.5 | 1.86–2.79 |
| Reproduction gate (§4): Qwen-7B Q4_K_M download + 25 longctx items + diff | 0.25–0.5 | 0.47–0.93 |
| Q5 suites only (2a longctx + arithmetic, 8 loads, 596 items each) | 0.5–0.7 | 0.93–1.30 |
| Full registered scope (§2: 1286 items × 8 loads) | 1.0–1.5 | 1.86–2.79 |
| NLL passes on 2a (§3: 8 loads × 96 items at t=4096, logits_all) | 2.0–3.5 | 3.72–6.51 |
| Optional benchmark (§6: 16 files, ~1 h re-downloads + ~5 min each) | 2.0–2.5 | 3.72–4.65 |
| EBS (AMI-backed gp3 volume for the run's hours) + S3 puts | — | ≈ 1.00 |
| **Total, full scope + NLL, no benchmark** | **4.5–7.0** | **≈ 9–14** |
| **Total, with benchmark** | **6.5–9.5** | **≈ 13–19** |

Cap: ____________ (user sets before "proceed").

## 9. Before "proceed" — what must be committed

1. Amendment 4 appended and stamped (the 1.5B ladder pins).
2. `registered.py`: 2a item-set hash for the 1.5B added and gate-tested.
3. `configs/0b-prime/qwen2.5-1.5b-ladder.yaml` (+ a smoke config), with
   `tests/test_0b_prime_configs.py` cross-checking every filename and
   sha256 against `qwen2.5-1.5b-bartowski.json` and every seed/n against
   `registered.py`; M3 weights in convention order.
4. Runner timing field (§5) implemented and tested.
5. The reproduction-gate script (§4) with its expected-output fixture.
6. This document's rulings confirmed, in particular §2's twins reading and
   §3's exploratory label.
