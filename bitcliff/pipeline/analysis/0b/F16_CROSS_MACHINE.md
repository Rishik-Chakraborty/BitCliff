# F16 cross-machine discrepancy: Mac calibration vs 0B cloud run

Evidence pack for a hand-check of the 9.4pp (llama) / 4.8pp (qwen) gap between
the Mac F16 `factual_qa` calibration (Amendment 1 §C: 0.314 llama-3.1-8b,
0.192 qwen2.5-7b, mix M3, n=500, seed 2718) and the 0B cloud confirmatory run
(0.220 llama, 0.144 qwen, same registered config label). PREREG §6 scopes
determinism to a fixed hw/sw configuration, so *some* cross-machine drift was
expected going in — but the actual mechanism found below is not
hardware/backend drift. **It is a wiring bug: the cloud confirmatory configs
graded a materially different, harder factual_qa item set than Mac
calibration measured, because the `weights` vector in `configs/0b/*.yaml`
was copied from PREREG §7's prose in the wrong index order.** Once that is
corrected for (comparing Mac and cloud on the *same* item set), the two
machines agree exactly at the aggregate level and item-for-item: 0/500
discordant grades for both models.

Regeneration wall time (Apple M5 Pro, Metal): ~192–239s per (model, item-set)
cell, 4 cells run, ~14 min total generation time (see §2 for the log lines).

---

## 1. Config diff table

| dimension | Mac calibration (`scripts/calibrate_f16.py`) | 0B cloud run (`runs-cloud/`, `configs/0b/*.yaml`) | identical? |
|---|---|---|---|
| llama-cpp-python version | `0.3.35` (this Mac, checked live: `import llama_cpp; llama_cpp.__version__`) | `0.3.35` (`runs-cloud/fingerprint.txt`) | **yes**, same version |
| llama-cpp-python backend | Metal (Apple M5 Pro GPU, `n_gpu_layers=-1`; `ggml_metal_device_init: GPU name: MTL0 (Apple M5 Pro)`) | CUDA (`fingerprint.txt`: `gpu=NVIDIA L40S, 595.91.07`, `cuda=release 13.2`, `llama_cpp_python=0.3.35 CUDA build, gpu_offload_supported=True`) | **no** — different backend/kernels (both installed from the identical PyPI sdist `llama_cpp_python-0.3.35.tar.gz`, pinned by hash in `uv.lock`, so the vendored llama.cpp source commit should be the same on both machines — `fingerprint.txt` records `llama_cpp_commit=bf942164697d2d62c2237a17b677dc2c017ea8e7` for the cloud build; this repo has no independent way to print that same string on the Mac build, so it is not independently confirmed here, only inferred from the shared sdist) |
| llama.cpp bundled commit | not independently printable from this Mac's installed wheel | `bf942164697d2d62c2237a17b677dc2c017ea8e7` (`fingerprint.txt`) | not directly determinable on Mac side; inferred **likely identical** (same sdist) |
| sampling params (factual_qa) | greedy, `temperature=0.0`, `top_k=1`, `seed=42` (`GEN_SEED`), 64-token answer budget (`FACTUAL_QA_ANSWER_BUDGET`), `n_ctx=2048` (`FACTUAL_QA_N_CTX`) | greedy, `temperature=0.0`, `top_k=1`, `seed=42`, 64-token budget (`suites.factual_qa.max_tokens: 64`), **`n_ctx=16384`** (`generation.n_ctx`, shared across all suites because longctx's t=8192 prompts need the headroom) | **no** — `n_ctx` differs (2048 vs 16384); seed/temp/top_k/budget identical |
| prompt construction | `factual_qa.items_from_records` → `PROMPT_PREFIX + question` (`"Answer with just the answer: "`), `create_chat_completion(messages=[{"role":"user","content":prompt}])` via `generate.run_items` | same function (`factual_qa.load_popqa_items` → `items_from_records`), same `generate.run_items`, called from `__main__.py:240` | **yes**, byte-identical code path (calibrate_f16.py imports and calls the same `bitcliff_pipeline.generate.run_items` / `bitcliff_pipeline.suites.factual_qa` the pipeline uses — not a reimplementation) |
| grader | `factual_qa.grade` (imported directly, called at `calibrate_f16.py:849`) | `grading.py`'s suite dispatch table maps `"factual_qa": factual_qa.grade` (same function, `grading.py:8-`) | **yes**, identical function |
| alias augmentation file | `data/popqa_wikidata_aliases_seed2718.json` | `data/popqa_wikidata_aliases_seed2718.json` (`configs/0b/*.yaml` `alias_augmentation_path`) | **yes** |
| **item set (factual_qa, mix "M3", n=500, seed=2718)** | `item_set_sha256 = 2e53ca0e73e9cbdd5d7bac672857ba918ff7b74d2405ddf62f37aa9a630091c6` (PREREG's registered M3 hash for both llama-3.1-8b and qwen2.5-7b; item sets are model-independent) | `item_set_sha256 = ac5cb2820d6693ad58c365e5c3cfcf6962b52c833d6cde16d9edcb52ae9fa081` (recomputed from `runs-cloud/pipeline/runs/*/items.jsonl`, all 4 cloud run dirs agree) | **NO — different item sets.** Root cause identified below. |
| model weights (F16 gguf) | `models/f16/Llama-3.1-8B-Instruct-f16.gguf` sha256 `139c255d857940bc945b2a3242fbbb2641b59191d79413bcad9b74fa1784c7b0`; `models/f16/Qwen2.5-7B-Instruct-f16.gguf` sha256 `970ccec3ad83bb62aa25ce585bed4ebd297963257442afe697d350465933f2c5` | manifest.json `F16.sha256` identical for both (`0b-llama-8b-ladder/manifest.json`, `0b-qwen-7b-ladder/manifest.json`) | **yes**, byte-identical weights |

### Root cause of the item-set mismatch

PREREG §7 writes the M3 weight vector in prose as **"decile 1 = most
popular .. decile 10 = least popular"** order:
`(0.16, 0.16, 0.16, 0.16, 0.16, 0.04, 0.04, 0.04, 0.04, 0.04)`.

But `suites/factual_qa.py::items_from_records`'s `weights[i]` parameter is
applied to `deciles[i]` **as that function builds them** — sorted by
`s_pop` **ascending**, so `deciles[0]` is the *least* popular decile and
`deciles[9]` is the *most* popular one (`factual_qa.py` module docstring,
step 1-3).

- `scripts/calibrate_f16.py`'s `M3_WEIGHTS` constant **correctly reverses**
  PREREG's listed vector into this ascending-index convention:
  `(0.04,)*5 + (0.16,)*5` — 80% of the mass on the popular half, 20% floor
  on the tail, matching PREREG's own description ("popular-weighted with a
  fixed 20% tail floor"). This is what Mac calibration measured and is what
  produces item-set hash `2e53ca0e...`.
- `configs/0b/*.yaml`'s `suites.factual_qa.weights` field instead copies
  PREREG's **literal, unreversed** decile-1-first vector verbatim:
  `[0.16, 0.16, 0.16, 0.16, 0.16, 0.04, 0.04, 0.04, 0.04, 0.04]`. Since
  `__main__.py:66` forwards this straight to `items_from_records` with no
  reversal (`weights = tuple(s["weights"])`), the cloud run applied 80% of
  the mass to `deciles[0..4]` — the **least popular (tail) half** — and
  only 20% to the popular head. This is the mirror image of the intended
  M3 mix: **more tail-heavy than M1 (uniform)**, not "popular-weighted."
  `tests/test_0b_configs.py:29` independently defines its own
  `M3_WEIGHTS = (0.16,)*5 + (0.04,)*5` and asserts the yaml matches *that*
  — so the test is self-consistent with the bug and never caught it; it
  never cross-checks against `calibrate_f16.py`'s (correct) constant or
  against `items_from_records`'s actual indexing convention.
- Confirmed empirically: rebuilding the item set from the real
  `akariasai/PopQA` test split with `calibrate_f16.py`'s `M3_WEIGHTS`
  reproduces `2e53ca0e...` exactly; rebuilding with the cloud yaml's literal
  vector reproduces `ac5cb282...` exactly (both recomputed live with
  `uv run python3` in this investigation). Only 20/500 item ids happen to
  carry the same question text across the two sets (decile 0's early draws
  before the seeded RNG streams diverge); the other 480/500 ids are
  different PopQA questions entirely, so a naive "compare by item_id"
  discordance count between the Mac calibration's own M3 set and the cloud
  outputs would be comparing different questions under the same id label
  — not a valid signal, which is why §2 below uses a regenerated, same-item-set
  comparison instead.

This is a real, disclosable wiring bug in the confirmatory-run configs, not
a hardware/software drift phenomenon. It was not previously flagged in
PREREG, Amendment 1, RUN_0B.md, or OPEN_QUESTIONS.md (grepped, no hits) —
this file does not edit any of those per the read-only constraint on
committed artifacts; the maintainer should decide whether/how to disclose
it there.

---

## 2. Discordant items

Per-item Mac outputs were not stored by `calibrate_f16.py` (only
per-mix `measurements.jsonl` summaries) — regenerated locally, same code
path (`bitcliff_pipeline.generate.make_llm` / `run_items`,
`suites.factual_qa.items_from_records` / `grade`), M3 weights, alias file
`data/popqa_wikidata_aliases_seed2718.json`, 64-token budget, seed 42 /
temp 0.0 / top_k 1, `n_ctx=2048`. Two item sets were regenerated per model,
since §1 established the Mac and cloud M3 mixes are not the same item set:

| file | weights variant | item_set_sha256 | wall time |
|---|---|---|---|
| `analysis/0b/f16-mac-regen-llama-3.1-8b-instruct.jsonl` | `calibrate_f16.py` M3 (reversed — Mac calibration's own mix) | `2e53ca0e73e9cbdd5d7bac672857ba918ff7b74d2405ddf62f37aa9a630091c6` | 208.9s |
| `analysis/0b/f16-mac-regen-qwen2.5-7b-instruct.jsonl` | same | same | 191.3s |
| `analysis/0b/f16-mac-regen-cloudset-llama-3.1-8b-instruct.jsonl` | cloud yaml's literal M3 (the mix actually graded on the cloud) | `ac5cb2820d6693ad58c365e5c3cfcf6962b52c833d6cde16d9edcb52ae9fa081` | 232.7s |
| `analysis/0b/f16-mac-regen-cloudset-qwen2.5-7b-instruct.jsonl` | same | same | 238.0s |

**Sanity check (Mac calibration reproducibility):** the first two rows
reproduce PREREG's registered Amendment 1 values exactly —
llama 0.3140 (157/500), qwen 0.1920 (96/500), both matching
`calibration/<model>/measurements.jsonl`'s stored M3 accuracy and
`item_set_sha256` bit for bit. This validates the regeneration
methodology before using it for the real comparison below.

**Real comparison (same item set: Mac regenerated on the cloud's actual
`ac5cb282...` M3 set, machine and backend held apart):**

| model | n items | text byte-identical | grade (correct/wrong) matches | n_discordant | Mac acc | cloud acc |
|---|---|---|---|---|---|---|
| llama-3.1-8b-instruct | 500 | 487/500 (97.4%) | **500/500** | **0** | 0.2200 (110/500) | 0.2200 (110/500) |
| qwen2.5-7b-instruct | 500 | 492/500 (98.4%) | **500/500** | **0** | 0.1440 (72/500) | 0.1440 (72/500) |

**n_discordant = 0 for both models.** Direction split: n/a (no discordant
items). The Mac-regenerated aggregate accuracy on the cloud's item set is
an *exact* match to the cloud's own registered F16 factual_qa accuracy for
both models — not just close, bit-for-bit on `n_correct`.

The 13/500 (llama) and 8/500 (qwen) items with non-byte-identical but
equally-*graded* text are genuine cross-machine (Metal vs CUDA) decode
divergences — sampled examples:

- `factual_qa-2718-0184` (llama): Mac `"Haruki Murakami"` vs cloud
  `"I couldn't find any information about a book called \"The Smile\"."`
  — both wrong, but a full early-token divergence, not just a trailing
  variation.
- `factual_qa-2718-0200` (qwen): Mac `"William R. Kirkpatrick"` vs cloud
  `"William Alland"` — same pattern, both wrong.
- Several others are single-word/punctuation-level (`"Catholicism"` vs
  `"Catholicism."`) or paraphrase-level (`"I couldn't find any information
  on the book..."` vs `"...about the book..."`), consistent with a
  near-tied top-1 logit at some early decode step that different
  backends' floating-point kernels break differently, occasionally
  cascading into a fully different continuation for the rest of the
  64-token budget. None of the observed divergences flipped a grade in
  this run (all pre- and post-divergence continuations graded the same
  verdict), but the mechanism is real and not zero-probability of
  eventually flipping an item at larger n.

Because the item sets differ by construction (§1), a per-`item_id`
discordance count between the Mac calibration's *own* M3 set
(`2e53ca0e...`) and the cloud's outputs was not computed as a headline
number — 480/500 ids would be comparing different PopQA questions under
the same id label, which is not a meaningful "discordant item," only an
artifact of the weights bug already characterized in §1.

---

## 3. Band check (PREREG §7, [0.6, 0.85])

| model | machine | factual_qa F16 accuracy | vs [0.6, 0.85] band |
|---|---|---|---|
| llama-3.1-8b-instruct | Mac calibration | 0.314 | below band |
| llama-3.1-8b-instruct | cloud confirmatory | 0.220 | below band |
| qwen2.5-7b-instruct | Mac calibration | 0.192 | below band |
| qwen2.5-7b-instruct | cloud confirmatory | 0.144 | below band |

Both cloud F16 accuracies (0.220, 0.144) remain **far below** the
registered [0.6, 0.85] band, on the same side as the Mac calibration
values that originally triggered the Amendment 1 §7 out-of-band M3
fallback (no candidate mix landed in-band; M3 selected with disclosure).
**The registered out-of-band fallback disclosure is unaffected in kind** —
nothing here changes which mix was chosen or that the fallback applies.
**The disclosed out-of-band VALUE differs by machine** (0.314 vs 0.220
llama, 0.192 vs 0.144 qwen), and per §1/§2 above, that difference is now
understood to be driven almost entirely by the item-set-mismatch bug
rather than by genuine measurement noise — i.e., the cloud run's
out-of-band value is a measurement of a *different, harder* item set than
the one PREREG's Amendment 1 text describes and hashes.

---

## 4. Most likely mechanism (summary)

The 9.4pp (llama) and 4.8pp (qwen) gaps between Mac F16 calibration and the
cloud F16 confirmatory run are **not** primarily a cross-machine
determinism artifact. `configs/0b/*.yaml`'s `factual_qa.weights` field
copied PREREG §7's M3 vector verbatim in its prose ordering
("decile 1 = most popular .. decile 10 = least popular") without reversing
it for `items_from_records`' actual ascending-`s_pop`-index convention —
a bug independently reproduced by `tests/test_0b_configs.py`'s own
(equally unreversed) `M3_WEIGHTS` constant, so no test caught it. The
result is that the cloud confirmatory run graded a genuinely different
500-item factual_qa set (`ac5cb282...`) than the one Mac calibration
measured and PREREG's Amendment 1 registered (`2e53ca0e...`) — one that
weights the *unpopular* (harder) decile half at 80% instead of 20%, i.e.
closer to "inverse-M3" than to the intended "popular-weighted, 20% tail
floor" mix — which mechanically produces a lower F16 accuracy regardless
of machine. Isolating the machine variable by regenerating the Mac F16 run
on the cloud's *actual* item set shows the two machines agree exactly:
0/500 discordant grades and bit-identical aggregate accuracy for both
models. The only residual cross-machine signal is a small number of
non-byte-identical outputs (13/500 llama, 8/500 qwen) consistent with
ordinary greedy-decode tie-breaking divergence between Metal and CUDA
floating-point kernels on items where the model's top token choice is
near-tied (open-ended, low-confidence factual_qa answers) — this is the
kind of drift the freeform 64-token factual_qa task is naturally more
exposed to than longctx's 32-token, tightly-separated substring-match
task (which PREREG already reports as matching across machines to 4
decimals) — but at n=500 it produced zero grade flips in this run, not the
multi-point-percentage effect the raw calibration-vs-cloud numbers
suggested before the item-set bug was isolated.

---

## Files

- `analysis/0b/f16-mac-regen-llama-3.1-8b-instruct.jsonl` — Mac F16 regen, Mac's own M3 item set (`2e53ca0e...`)
- `analysis/0b/f16-mac-regen-qwen2.5-7b-instruct.jsonl` — same, qwen2.5-7b
- `analysis/0b/f16-mac-regen-cloudset-llama-3.1-8b-instruct.jsonl` — Mac F16 regen, cloud's actual (buggy-weights) item set (`ac5cb282...`)
- `analysis/0b/f16-mac-regen-cloudset-qwen2.5-7b-instruct.jsonl` — same, qwen2.5-7b
- `analysis/0b/F16_CROSS_MACHINE.md` — this file

Each `.jsonl` file's first line is a `_meta` record (model_id, weights
variant, item_set_sha256, model_sha256, n_items, n_correct, accuracy,
wall_time_s, gen params); subsequent lines are per-item records
(`item_id`, `question`, `expected`, `output`, `finish_reason`, `grade`,
`machine`).
