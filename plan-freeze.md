# Freeze Execution Plan (F1–F3 of freeze-plan.md)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Executes freeze-plan.md up to (not including) the F4 timestamped commit — the run STOPS when PREREG.md is drafted, for user review.

**Goal:** Build the freeze machinery (longctx_retrieval suite, closed-book QA suite, twins builder+verifier, reporting flag), run the audits and verified file lists, characterize the alias grader, and assemble PREREG.md — then stop for user sign-off before any timestamping.

**Spec:** `freeze-plan.md` (authority), `claude/IDEA.md`, the bundle at `/Users/rishikchakraborty/Documents/Coding/quantization/multivalue2-bundle/` (TASK.md, MECHANISM.md, PAPER_CONFIG.md, GRADING.md, ANSWER_TOKENS.md, PROVENANCE.md, `generate_multivalue2.py`), and the user rulings of 2026-08-26 (below, binding).

## Global Constraints (user rulings verbatim-binding)

1. **Closed-book QA:** PopQA, n=500 stratified by entity popularity, word-boundary-anchored alias-substring grading. (a) License audit runs and is recorded BEFORE any prompts are built. (b) A manual spot-check of 30 graded items (15 correct / 15 incorrect, sampled across rungs) characterizes the grader's FP/FN rate before PREREG freezes it. TriviaQA is the named fallback if audit or spot-check fails.
2. **Corpus 2b:** The Count of Monte Cristo (PG #1184). Record exact Gutenberg file ID, encoding, and SHA256 in the manifest.
3. **Margins:** M = 3pp per suite, equivalence by CI-inclusion, per-cell verdicts with no multiplicity correction — BUT any headline claim aggregating across cells is stated only if it holds under Holm. The dual rule is written into PREREG explicitly.
4. **Timestamp:** OpenTimestamps on the PREREG commit; the commit hash is also recorded in PILOT_NOTES.md so the artifacts cross-reference. (Timestamping itself is F4 — NOT in this plan; PREREG must document the mechanism.)
5. **Shootout:** 8B only, Q4+Q3. If the uploader effect is visible there, extending to the 7B is a registered follow-up, never scope creep.
- Naming: the capability is **long-context retrieval** (`longctx_retrieval`); bare "retrieval" never appears in new code or docs. The placeholder retrieval suite is deleted, not renamed.
- Grading for longctx_retrieval is the paper rule VERBATIM (conjunctive plain substring, order-insensitive; GRADING.md is authority). Answer-span spec per ANSWER_TOKENS.md.
- Zero cloud spend; everything local. Exploratory runs (grader characterization) are declared exploratory. Nothing confirmatory runs in this plan.
- `max_tokens` for confirmatory configs is 1024 (registered); longctx_retrieval generation uses the paper's 32-new-tokens budget as a per-suite override.
- Tooling: uv + pytest, TDD for code tasks, commit per task. Work happens on branch `freeze-prep` off main.

## File Structure (under bitcliff/pipeline unless noted)

- Create: `src/bitcliff_pipeline/suites/longctx_retrieval.py` (+ `vendor/generate_multivalue2.py`), `src/bitcliff_pipeline/suites/factual_qa.py`, `src/bitcliff_pipeline/twins/` (`templates.py`, `builder.py`, `verifier.py`), `LICENSE_AUDIT.md`, `CORPUS_MANIFEST.md`, `reference-manifests/*.json`, `GRADER_CHARACTERIZATION.md`, `PREREG.md` (repo root)
- Modify: `src/bitcliff_pipeline/items.py` (prompt_tokens), `generate.py` (token-id path + per-suite max_tokens), `grading.py` (GRADERS), `__main__.py` (build_items), `report.py` (spectacle_only threading), `config.py` (per-suite overrides), configs
- Delete: `src/bitcliff_pipeline/suites/retrieval.py`, `tests/test_retrieval.py`

---

### Task 1: longctx_retrieval suite (vendor + adapter + verbatim grading; delete placeholder)

**Files:** Create `src/bitcliff_pipeline/vendor/__init__.py`, `src/bitcliff_pipeline/vendor/generate_multivalue2.py` (copied VERBATIM from `/Users/rishikchakraborty/Documents/Coding/quantization/multivalue2-bundle/generate_multivalue2.py` — read it first; logic must not change), `src/bitcliff_pipeline/suites/longctx_retrieval.py`, `tests/test_longctx_retrieval.py`. Delete `src/bitcliff_pipeline/suites/retrieval.py`, `tests/test_retrieval.py`. Modify `src/bitcliff_pipeline/items.py`, `src/bitcliff_pipeline/grading.py` (GRADERS dict), `src/bitcliff_pipeline/__main__.py` (build_items: drop "retrieval" branch; longctx items are NOT built by build_items in this task — the suite exposes its own builder for later run configs), `configs/qwen2.5-1.5b-pilot.yaml` (remove the retrieval suite block), plus fix any test fixtures that referenced the deleted suite (test_cli's config builder uses retrieval — switch it to a longctx-free shape using spectacle only, or arithmetic-record injection).

**Interfaces:**
- `items.EvalItem` gains `prompt_tokens: tuple[int, ...] | None = None` (last field, default None — all existing constructions keep working). For longctx items: `prompt` holds a short human-readable descriptor (never the document; the document is only in tokens), `expected` = the match_strings tuple, `prompt_tokens` = full token-id prompt.
- `longctx_retrieval.build_items(tokenizer, corpus_text: str, corpus_sha256: str, n_items: int, seed: int, variant: str = "multivalue2", target_tokens: int = 4096) -> list[EvalItem]` — wraps the vendored generator; verifies sha256 of corpus_text against corpus_sha256 (raises on mismatch); item ids `longctx_retrieval-{variant}-t{target_tokens}-s{seed}-{i:04d}`.
- `longctx_retrieval.grade(item, text) -> str` — paper rule verbatim: `"correct" if all(s in text for s in item.expected) else "wrong"` (no partial; unanchored substring; order-insensitive by construction).
- `GRADERS = {"longctx_retrieval": ..., "arithmetic": ..., "spectacle": ..., "factual_qa": ...}` — factual_qa added in Task 3; this task leaves a working dict without "retrieval".

**Steps:** (TDD) 1) Read the bundle generator + GRADING.md. 2) Write failing tests: grade rule (both present → correct, one → wrong, reversed order → correct, gold embedded in longer digit-run → correct [the paper's documented laxity — assert it to lock the rule], none → wrong); EvalItem back-compat (4-arg and 5-arg construction); build_items with a STUB tokenizer (deterministic word→id map with encode/decode/apply_chat_template minimal surface the generator needs — inspect the generator to build the stub) and a tiny synthetic corpus whose sha you compute in the test: assert n items, exact-token-length property if the generator exposes it, determinism across two calls, corpus-hash mismatch raises. NO network in tests. 3) Implement. 4) Full suite green (note: deleting test_retrieval.py drops the count; test_cli fixtures updated). 5) Commit `feat(pipeline): longctx_retrieval suite (vendored multivalue2), placeholder retrieval deleted`.

---

### Task 2: token-id generation path + per-suite max_tokens

**Files:** Modify `src/bitcliff_pipeline/generate.py`, `src/bitcliff_pipeline/config.py`, `tests/test_generate.py`, `tests/test_config.py`.

**Interfaces:**
- `generate.run_items(llm, items, quant_label, model_sha256, gen, max_tokens_by_suite: dict[str, int] | None = None)` — for an item with `prompt_tokens is not None`, call `llm.create_completion(prompt=list(item.prompt_tokens), max_tokens=<suite override or gen.max_tokens>, temperature=gen.temperature, top_k=gen.top_k, seed=gen.seed)`; response text at `out["choices"][0]["text"]`, finish_reason same key. Items without tokens keep the existing chat path. The OutputRecord's `gen_settings` dict gains `"max_tokens_effective"` per record.
- Config: `suites` block may carry `longctx_retrieval: {..., max_tokens: 32}`; `__main__` passes `{suite: cfg["max_tokens"] for suite,cfg in config.suites.items() if "max_tokens" in cfg}` into run_items.

**Steps:** TDD with a FakeLlm gaining `create_completion(prompt, ...)` recording calls: assert token-id items route to create_completion with the exact token list and the 32-token override while chat items still hit create_chat_completion with the global budget; `max_tokens_effective` recorded on both. Full suite green. Commit `feat(pipeline): token-id generation path with per-suite budget`.

---

### Task 3: closed-book factual QA suite (license audit FIRST)

**Files:** Create `LICENSE_AUDIT.md` (bitcliff/pipeline/), `src/bitcliff_pipeline/suites/factual_qa.py`, `tests/test_factual_qa.py`. Modify `grading.py` (GRADERS gains "factual_qa").

**Ordering (ruling 1a):** Step 1 of this task, before ANY suite code: query the HF API for `akariasai/PopQA` (`https://huggingface.co/api/datasets/akariasai/PopQA`) — record license tag, verification date, and the raw JSON snippet in LICENSE_AUDIT.md; do the same for the fallback `mandarjoshi/trivia_qa`. Also record (from freeze-plan §4): GSM8K (openai/gsm8k), sgoel9/paul_graham_essays (unresolved → quarantined to never-published 2a), Qwen2.5 (Apache-2.0), Llama-3.1 (fetch the community-license output-publication clause verbatim from the HF model page/LICENSE file and quote it), quantization repo (Apache-2.0, commit 0d1885e). If PopQA's license is missing/restrictive: STOP and report BLOCKED (TriviaQA switch is a controller decision, not yours).

**Interfaces:**
- `factual_qa.items_from_records(records, n_items, seed) -> list[EvalItem]` — records are PopQA-shaped dicts (`question`, `possible_answers` as a JSON-encoded list string or list, `s_pop`/popularity field — inspect the real schema via the datasets viewer/API and document which field you used); stratified sample: sort by popularity, split into 10 deciles, draw n_items/10 per decile with `random.Random(seed)`; prompt = `"Answer with just the answer: {question}"`; `expected` = tuple of aliases; ids `factual_qa-{seed}-{i:04d}`.
- `factual_qa.load_popqa_items(n_items, seed)` — thin datasets wrapper (untested).
- `factual_qa.grade(item, text) -> str` — "correct" iff ANY alias matches under: lowercase both; normalize whitespace; alias matched as a word-boundary-anchored substring (`re.search(rf"(?<!\w){re.escape(alias_norm)}(?!\w)", text_norm)`); else "wrong". No partial.

**Steps:** license audit (Step 1, committed evidence) → TDD the pure functions (stratification determinism + decile coverage; grading: alias hit → correct, case-insensitive → correct, alias inside a longer word → wrong [the boundary anchor], punctuation-adjacent alias → correct, no alias → wrong) → implement → suite green → commit `feat(pipeline): closed-book factual QA suite (PopQA) with recorded license audit`.

---

### Task 4: GSM8K twins — templates, builder, round-trip verifier

**Files:** Create `src/bitcliff_pipeline/twins/__init__.py`, `twins/templates.py` (the template data), `twins/builder.py`, `twins/verifier.py`, `tests/test_twins.py`. Modify `bitcliff/pipeline/.gitignore` (add `private/`).

**Interfaces:**
- `TwinTemplate(gsm8k_index: int, text_template: str, param_names: tuple[str, ...], solve_src: str, constraints_src: str, original_values: dict, original_answer: str)` — `solve_src` defines `def solve(**params) -> float|int`; `constraints_src` defines `def valid(**params) -> bool`.
- `verifier.verify_template(t, gsm8k_question: str, gsm8k_answer: str) -> None` — raises unless (a) `t.text_template.format(**t.original_values) == gsm8k_question` EXACTLY and `solve(**original numeric params)` equals the extracted `####` answer, and (c) every numeric param value appears in the formatted text exactly where the template says.
- `builder.build_twin(t, seed) -> dict` — resamples numeric params (name params from a fixed name pool) until `valid(...)`, returns `{"question": ..., "answer": str(solve(...)), "template_index": ..., "seed": ...}`; deterministic per (template, seed).
- `builder.build_twin_set(templates, gsm8k_records, seed, out_path) -> None` — verifies every template against its GSM8K record, builds one twin each, writes JSONL to `private/twins/` (gitignored), prints the file's sha256 (recorded at F4).
- **Template authoring target: ≥30 verified templates** from the first 60 GSM8K test items; untemplatable items are skipped with a one-line reason in a module docstring list. Every committed template MUST pass `verify_template` in the test suite against the real GSM8K text — tests may load GSM8K from the datasets cache (it is already downloaded on this machine from the pilot; if truly unavailable offline, vendor the needed 60 records into `tests/data/gsm8k_first60.jsonl` with a note that it is MIT-licensed test data).

**Steps:** TDD the verifier with a hand-made fake record first (mismatch text → raises; wrong solve → raises); then author templates in batches of 10, running the verifier tests each batch; implement builder + determinism test + constraint-respecting resample test; suite green; commit `feat(pipeline): GSM8K twin templates, builder, round-trip verifier (N templates)`.

---

### Task 5: corpora + verified reference file lists (no big downloads)

**Files:** Create `bitcliff/pipeline/CORPUS_MANIFEST.md`, `bitcliff/pipeline/reference-manifests/` (`llama-3.1-8b-bartowski.json`, `qwen2.5-7b-bartowski.json`, `shootout-8b.json`, `qwen2.5-7b-official.json`), `bitcliff/pipeline/scripts/enumerate_reference_files.py`.

**Steps:**
1. **Gutenberg 2b corpus (ruling 2):** download PG #1184 plain-text UTF-8 (record the exact file URL/ID, e.g. `https://www.gutenberg.org/cache/epub/1184/pg1184.txt`, and its encoding); strip header/footer by the documented rule (text between the `*** START OF THE PROJECT GUTENBERG EBOOK ...` and `*** END OF ...` markers, exclusive); save to `corpora/pg1184-monte-cristo.txt` (gitignored — add `corpora/` to .gitignore); record file ID, source URL, encoding, byte length, and sha256 of BOTH raw download and stripped text in CORPUS_MANIFEST.md.
2. **2a corpus verification:** load `sgoel9/paul_graham_essays` via datasets, join per PROVENANCE.md (`"\n\n"`), sha256 must equal `b6135331a3132d08cb84262870ae8f9d9acb6bae4cd7f0278926a64c38f9329e`; record PASS/FAIL in CORPUS_MANIFEST.md (FAIL → report BLOCKED).
3. **Bundle digest verification:** with the REAL Qwen2.5-1.5B-Instruct tokenizer (transformers; HF cache already has the model dir from the pilot) and the 2a corpus, build the first 20 multivalue2 items via `longctx_retrieval.build_items` / the vendored generator and compute the bundle's digest recipe; must equal `9220589bd8607bd0ff3be5bdcfecd23df07cac82d354d992468b15b60f398972` (the generator/PAPER_CONFIG.md documents the digest construction). Record PASS/FAIL. FAIL → BLOCKED (vendoring broke item construction).
4. **Reference file lists via HF API (no downloads):** `scripts/enumerate_reference_files.py` hits `https://huggingface.co/api/models/<repo>/tree/main` and records, per `.gguf` file: filename, size, and `lfs.oid` (the sha256) into the four manifests with retrieval date + repo revision (the API's commit sha). Repos: `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`, `bartowski/Qwen2.5-7B-Instruct-GGUF`, shootout-8B (Q4_K_M+Q3_K_M only, ruling 5): `unsloth/Meta-Llama-3.1-8B-Instruct-GGUF` and mradermacher's paired repos (`mradermacher/Meta-Llama-3.1-8B-Instruct-GGUF` static + `...-i1-GGUF` imatrix), and `Qwen/Qwen2.5-7B-Instruct-GGUF` (official). If a repo name 404s, find the correct name via the HF search API and record what you actually used.
5. Commit `feat(pipeline): corpus manifest and verified reference file lists`.

---

### Task 6: thread spectacle_only through reporting

**Files:** Modify `src/bitcliff_pipeline/report.py`, `src/bitcliff_pipeline/__main__.py`, `tests/test_report.py`, `tests/test_cli.py`.

**Interfaces:** `aggregate(grades, spectacle_only_labels: frozenset[str] = frozenset())` adds `"spectacle_only": bool` to each row; `add_retention` passes it through; `write_csv/json` unchanged (extra column flows); `plot_retention` draws spectacle_only rungs with dashed lines and a `(spectacle-only)` legend suffix; `__main__` report stage derives the label set from the config (`{q.label for q in config.quants if q.spectacle_only}`). Reference-table consumers (future site) exclude on this column — enforced by code, per freeze-plan §10 F1.

**Steps:** TDD (row carries the flag; csv has the column; plot smoke test with a mixed ladder) → implement → green → commit `feat(pipeline): spectacle_only threaded through report rows and plot`.

---

### Task 7: exploratory grader characterization (ruling 1b)

**Files:** Create `bitcliff/pipeline/GRADER_CHARACTERIZATION.md`, `configs/qwen2.5-1.5b-factualqa-explore.yaml`.

**Steps:**
1. Config: 1.5B ladder subset — F16, Q4_K_M, Q2_K only (three rungs spanning the range), suites: `factual_qa: {n_items: 500, seed: 7411}` only; `max_tokens: 64` for this suite (short answers), global gen settings as pilot. Run `--run-id factualqa-explore --stage all` locally (EXPLORATORY — say so in the doc). Expect ~1500 generations of short outputs.
2. Sample for the spot-check with a fixed seed: 15 graded-correct and 15 graded-incorrect items, drawn across the three rungs (5/5/5 each side, or as close as counts allow — document the actual draw).
3. Adjudicate each of the 30 by reading question, alias list, and the model output: is the grade RIGHT? Classify disagreements as grader-FP (graded correct, answer actually wrong) or grader-FN (graded wrong, answer actually right — e.g. an unlisted-but-valid alias). Record per-item verdicts in a table with the output excerpt, plus the FP/FN counts and rates.
4. Report F16 accuracy from results.csv (informative for the difficulty band; note if outside [0.6, 0.85] the popularity-mix knob per the registered rule will address it at calibration).
5. Verdict line: does the grader pass characterization (proposed bar: FP rate = 0/15 and FN ≤ 2/15 in the sample), or does the TriviaQA fallback trigger? If it FAILS: report BLOCKED with the table — the fallback switch is a controller/user decision.
6. Commit `docs(pipeline): alias-grader characterization (exploratory PopQA run)`.

---

### Task 8: PREREG.md assembly (STOP after this task)

**Files:** Create `PREREG.md` (repo root). Modify `bitcliff/pipeline/PILOT_NOTES.md` (add a "Freeze cross-reference" placeholder line noting the PREREG commit hash will be recorded here at F4).

**Content:** the full registration per freeze-plan §1, §6, §7, §8 with every user ruling inlined:
- The five questions (Q4 mechanism-(i) void, two-sided, closed-book QA as the direct mechanism-(i) measurement; Q5 with the IQ-vs-K pilot observation as exploratory-candidate only).
- Four suites; longctx_retrieval naming rule; both multivalue2 configurations (2a paper-exact seed-2024 n=96 lineage + never-published prompts; 2b PG#1184 with the CORPUS_MANIFEST hashes); PopQA spec + grading rule verbatim + the 30-item characterization protocol AND its measured result (from Task 7); twins construction + embargo + restriction rule + two-sided test.
- Fixed parameters (1024 global budget; 32-token longctx budget; deterministic settings; determinism scope).
- Difficulty-calibration RULE ([0.6, 0.85] F16 band, knobs per suite, amendment mechanics).
- Statistics: M = 3pp, CI-inclusion equivalence, four cell states, **the dual multiplicity rule verbatim: per-cell verdicts uncorrected; any cross-cell headline claim must survive Holm — stated as registered, not post-hoc**.
- Answer-token spec (10-token primary, digits-only sensitivity); grading symmetry note; tokenizer-bound task-level-comparison rule; blind-check protocol; dataset scope + reconstruction recipe + embargoes; citation hygiene (n=96 only, n=24 never); shootout scope (8B-only Q4+Q3, 7B extension pre-registered as a follow-up trigger, per ruling 5); timestamp mechanism (OpenTimestamps on the PREREG commit; hash cross-recorded in PILOT_NOTES.md); exploratory-pilot declaration covering pilot-0a, the in-house rungs, and the factualqa-explore run.
- Commit `docs: PREREG draft for user review`. **Then the plan ENDS — the controller stops and presents PREREG.md to the user. No timestamping, no F4.**

## Out of scope
F4 (user-edited PREREG, freeze commit, OpenTimestamps stamp), difficulty calibration runs, 0B confirmatory, site work.
