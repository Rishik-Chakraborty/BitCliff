# Pre-0B Tickets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two pre-0B tickets from `freeze-plan.md` §10: (1) record corpus provenance + suite config in each run's `manifest.json` and replace `package_dataset.py`'s config-2a seed-signature heuristic with a positive provenance check; (2) verify llama-cpp `create_completion` populates `finish_reason="length"` on truncation, as a preflight that runs before confirmatory generation.

**Architecture:** A reserved `_run_config` key (underscore prefix = "not a rung") is written into each run's `manifest.json` at the download stage, carrying `model_id` and the full `suites` config verbatim — including, for any future `longctx_retrieval` run config, its `corpus_sha256`. `hashing.verify_manifest` and the packager learn to treat underscore-prefixed keys as metadata, not rungs. The packager's 2a heuristic is deleted and replaced by a positive check on the recorded corpus hash. Separately, `generate.py` gains a truncation preflight (`assert_truncation_finish_reason`) run once per loaded model in the generate stage, plus a real-model integration test proving actual llama-cpp behavior.

**Tech Stack:** Python 3.12, uv, pytest, llama-cpp-python. All work under `bitcliff/pipeline/`.

**Spec:** `freeze-plan.md` §10 (the two ticket lines), `OPEN_QUESTIONS.md` §4b (the heuristic's history and the ruled replacement direction), PREREG §3.1 (2a embargo: "never displayed on the site and never published"), PREREG §6 (truncation is its own verdict; `truncated` derives from `finish_reason == "length"` in `grading.py:49`).

## Global Constraints

- PREREG governs. No registered rule, seed, n, or grading rule changes. These tickets are engineering plumbing, explicitly ticketed pre-0B in `freeze-plan.md` §10.
- Working directory for all commands: `bitcliff/pipeline/` (run tests as `uv run pytest -q`).
- The full suite (203 tests at baseline) must stay green after every task; new tests add to that count.
- Registered corpus hashes (copy verbatim, never retype from memory — copy from `scripts/package_dataset.py` and PREREG §3.1):
  - 2b (PG-1184 stripped): `0a21a13834b5215876bd4019af8fbc436abbfbb61b2826db62223eb990071443` (already in packager as `CORPUS_2B_STRIPPED_SHA256`)
  - 2a (PG-essays): `b6135331a3132d08cb84262870ae8f9d9acb6bae4cd7f0278926a64c38f9329e` (PREREG §3.1 config 2a)
- No pushes, no timestamping, no network beyond what existing tests already do. Commit per task.

---

### Task 1: Record `_run_config` (model_id + suites config) in run manifest.json

**Files:**
- Modify: `src/bitcliff_pipeline/hashing.py` (verify_manifest)
- Modify: `src/bitcliff_pipeline/__main__.py` (run_pipeline download stage)
- Test: `tests/test_hashing.py`, `tests/test_cli.py`

**Interfaces:**
- Produces: `manifest.json` may contain a reserved top-level key `"_run_config"` = `{"model_id": <str>, "suites": <the config's suites dict, verbatim>}`. Keys starting with `_` are metadata: `verify_manifest` skips them; rung iteration elsewhere must not treat them as quant labels. Task 2 consumes this key.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hashing.py`:

```python
def test_verify_manifest_skips_underscore_metadata_keys(tmp_path):
    from bitcliff_pipeline.hashing import build_manifest, verify_manifest

    p = tmp_path / "m.gguf"
    p.write_bytes(b"model bytes")
    manifest = build_manifest({"Q4_K_M": p})
    manifest["_run_config"] = {"model_id": "test-model", "suites": {}}
    # must not KeyError on the metadata key; must still verify the real rung
    verify_manifest(manifest, {"Q4_K_M": p})


def test_verify_manifest_still_catches_mismatch_with_metadata_present(tmp_path):
    import pytest
    from bitcliff_pipeline.hashing import ManifestMismatch, build_manifest, verify_manifest

    p = tmp_path / "m.gguf"
    p.write_bytes(b"model bytes")
    manifest = build_manifest({"Q4_K_M": p})
    manifest["_run_config"] = {"model_id": "test-model", "suites": {}}
    p.write_bytes(b"tampered")
    with pytest.raises(ManifestMismatch):
        verify_manifest(manifest, {"Q4_K_M": p})
```

Append to `tests/test_cli.py` (uses the existing `make_config`/`FakeLlm` helpers already in that file):

```python
def test_manifest_records_run_config(tmp_path):
    """Pre-0B ticket (freeze-plan §10): manifest.json carries the run's
    suite config so the packager can positively identify corpus provenance
    (e.g. a longctx run's corpus_sha256) instead of the seed heuristic."""
    cfg = make_config(tmp_path)
    cfg = dataclasses.replace(
        cfg,
        suites={
            **cfg.suites,
            "longctx_retrieval": {
                "variant": "multivalue4",
                "target_tokens": 8192,
                "seed": 2024,
                "n_items": 96,
                "corpus_sha256": "0a21a13834b5215876bd4019af8fbc436abbfbb61b2826db62223eb990071443",
                "max_tokens": 32,
            },
        },
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "m-Q4_K_M.gguf").write_bytes(b"quant bytes")
    cfg.f16_path.write_bytes(b"f16 bytes")
    runs_dir = tmp_path / "runs"

    run_pipeline(
        cfg, run_id="runconfig-test", models_dir=models_dir, runs_dir=runs_dir,
        stage="download", llm_factory=lambda path, gen: FakeLlm(), base_dir=tmp_path,
    )

    manifest = json.loads((runs_dir / "runconfig-test" / "manifest.json").read_text())
    rc = manifest["_run_config"]
    assert rc["model_id"] == "test-model"
    assert rc["suites"]["longctx_retrieval"]["corpus_sha256"] == (
        "0a21a13834b5215876bd4019af8fbc436abbfbb61b2826db62223eb990071443"
    )
    # metadata key must not look like a rung to downstream consumers
    assert set(manifest) - {"_run_config"} == {"F16", "Q4_K_M"}
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_hashing.py tests/test_cli.py -q`
Expected: the two hashing tests FAIL with `KeyError: '_run_config'` (verify_manifest indexes `files[label]`); the CLI test FAILS with `KeyError: '_run_config'` (key never written).

- [ ] **Step 3: Implement**

In `src/bitcliff_pipeline/hashing.py`, change `verify_manifest`:

```python
def verify_manifest(manifest: dict, files: dict[str, Path]) -> None:
    for label, entry in manifest.items():
        if label.startswith("_"):
            # Reserved metadata (e.g. "_run_config"), not a rung entry.
            continue
        actual = sha256_file(files[label])
        if actual != entry["sha256"]:
            raise ManifestMismatch(
                f"{label}: expected {entry['sha256'][:12]}..., got {actual[:12]}..."
            )
```

In `src/bitcliff_pipeline/__main__.py`, in the `if stage in ("download", "all"):` block, after the `for label, entry in manifest.items(): entry.update(meta[label])` loop and before `write_manifest(...)`, add:

```python
        # Pre-0B ticket (freeze-plan §10): record the run's config in the
        # manifest so dataset packaging can positively identify corpus
        # provenance (2a vs 2b) instead of the retired seed heuristic.
        manifest["_run_config"] = {
            "model_id": config.model_id,
            "suites": config.suites,
        }
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass (206 = 203 baseline + 3 new). If any pre-existing test fails on the new `_run_config` key (e.g. an exact-manifest-keys assertion), fix the production code path only if the failure reveals a real consumer treating `_run_config` as a rung; otherwise update the test's expectation and say so in the commit message.

- [ ] **Step 5: Commit**

```bash
git add src/bitcliff_pipeline/hashing.py src/bitcliff_pipeline/__main__.py tests/test_hashing.py tests/test_cli.py
git commit -m "feat: record _run_config (model_id + suites) in run manifest.json (pre-0B ticket, freeze-plan §10)"
```

---

### Task 2: Replace the packager's 2a seed-signature heuristic with a positive corpus-provenance check

**Files:**
- Modify: `scripts/package_dataset.py` (delete `_check_2a_signature` + `CONFIG_2A_VARIANT/SEED/TARGET_TOKENS`; add `CORPUS_2A_SHA256` and `_check_longctx_corpus_provenance`; `RunData` gains `run_config`; `load_run` pops `_run_config`)
- Test: `tests/test_package_dataset.py`

**Interfaces:**
- Consumes: Task 1's `manifest.json` `"_run_config"` key (`{"model_id", "suites"}`).
- Produces: `RunData.run_config: dict | None`; `_check_longctx_corpus_provenance(run: RunData, run_dir: Path) -> None` raising `EmbargoViolation`. Rule, exactly: if the run has `longctx_retrieval` items, read `run.run_config["suites"]["longctx_retrieval"]["corpus_sha256"]` (each level via `.get`). Missing/None → refuse (no recorded provenance). Equal to `CORPUS_2A_SHA256` → refuse (2a embargo, PREREG §3.1). Not equal to `CORPUS_2B_STRIPPED_SHA256` → refuse (unrecognized corpus). Equal to `CORPUS_2B_STRIPPED_SHA256` → pass.

- [ ] **Step 1: Write the failing tests**

In `tests/test_package_dataset.py`, first find the module-level `MANIFEST` dict and the `_build_run_dir` helper. Add a helper right after `_build_run_dir`:

```python
LONGCTX_2B_ITEMS = [
    {
        "id": "arithmetic-3141-000",
        "suite": "arithmetic",
        "prompt": "2+2?",
        "expected": ["4"],
    },
    {
        "id": "longctx_retrieval-multivalue4-t8192-s2024-0000",
        "suite": "longctx_retrieval",
        "prompt": "What is the secret passcode?",
        "expected": ["9999"],
        "prompt_tokens": [1, 2, 3],
    },
]


def _build_longctx_run_dir(tmp_path, name, corpus_sha256=..., items=None):
    """Run dir whose manifest carries a _run_config block. corpus_sha256:
    Ellipsis sentinel = omit the longctx suite config entirely (legacy,
    no provenance); a str = record it."""
    run_dir = _build_run_dir(tmp_path, name, items or LONGCTX_2B_ITEMS)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    longctx_cfg = {"variant": "multivalue4", "target_tokens": 8192, "seed": 2024}
    if corpus_sha256 is not ...:
        longctx_cfg["corpus_sha256"] = corpus_sha256
    manifest["_run_config"] = {
        "model_id": "test-model",
        "suites": {"longctx_retrieval": longctx_cfg},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return run_dir
```

Replace `test_2a_signature_run_is_refused` with these five tests (delete the old test — the heuristic it exercises is being deleted):

```python
def test_longctx_run_without_recorded_provenance_is_refused(tmp_path):
    run_dir = _build_longctx_run_dir(tmp_path, "no-provenance")  # omits corpus_sha256
    out_dir = tmp_path / "dist" / "dataset-no-provenance"
    with pytest.raises(pkg.EmbargoViolation, match="no corpus provenance"):
        pkg.package(run_dir, out_dir)
    assert not out_dir.exists()


def test_longctx_run_without_run_config_at_all_is_refused(tmp_path):
    # A legacy manifest with no _run_config key: same refusal path.
    run_dir = _build_run_dir(tmp_path, "legacy", LONGCTX_2B_ITEMS)
    out_dir = tmp_path / "dist" / "dataset-legacy"
    with pytest.raises(pkg.EmbargoViolation, match="no corpus provenance"):
        pkg.package(run_dir, out_dir)
    assert not out_dir.exists()


def test_longctx_run_with_2a_corpus_hash_is_refused(tmp_path):
    run_dir = _build_longctx_run_dir(
        tmp_path, "twoa-corpus", corpus_sha256=pkg.CORPUS_2A_SHA256
    )
    out_dir = tmp_path / "dist" / "dataset-twoa-corpus"
    with pytest.raises(pkg.EmbargoViolation, match="config 2a"):
        pkg.package(run_dir, out_dir)
    assert not out_dir.exists()


def test_longctx_run_with_unknown_corpus_hash_is_refused(tmp_path):
    run_dir = _build_longctx_run_dir(
        tmp_path, "unknown-corpus", corpus_sha256="ab" * 32
    )
    out_dir = tmp_path / "dist" / "dataset-unknown-corpus"
    with pytest.raises(pkg.EmbargoViolation, match="unrecognized corpus"):
        pkg.package(run_dir, out_dir)
    assert not out_dir.exists()


def test_2a_shaped_ids_with_registered_2b_corpus_hash_are_not_refused(tmp_path):
    """The exact case the old heuristic got wrong (OPEN_QUESTIONS §4b): 2b
    now shares 2a's registered n=96/seed=2024 (PREREG Amendment 1 §A), so a
    2b run's item ids can look 2a-shaped. Positive provenance must let it
    through the provenance gate (the call may still fail LATER for missing
    tokenizer/corpus paths — assert on the error type to prove the
    provenance check itself passed)."""
    items = [
        dict(LONGCTX_2B_ITEMS[0]),
        {
            **LONGCTX_2B_ITEMS[1],
            "id": "longctx_retrieval-multivalue2-t4096-s2024-0000",
        },
    ]
    run_dir = _build_longctx_run_dir(
        tmp_path, "twob-provenance",
        corpus_sha256=pkg.CORPUS_2B_STRIPPED_SHA256, items=items,
    )
    out_dir = tmp_path / "dist" / "dataset-twob-provenance"
    # Provenance gate passes; the next gate (tokenizer/corpus paths required
    # for longctx reconstruction) raises ValueError, NOT EmbargoViolation.
    with pytest.raises(ValueError, match="tokenizer-path"):
        pkg.package(run_dir, out_dir)
```

Also grep the test file for other uses of the deleted names (`CONFIG_2A_VARIANT`, `_check_2a_signature`) and update or remove them.

- [ ] **Step 2: Run the packager tests to verify the new ones fail**

Run: `uv run pytest tests/test_package_dataset.py -q`
Expected: the five new tests FAIL (`AttributeError: ... has no attribute 'CORPUS_2A_SHA256'` / no "no corpus provenance" match); existing tests still pass.

- [ ] **Step 3: Implement in `scripts/package_dataset.py`**

Delete the `CONFIG_2A_VARIANT`, `CONFIG_2A_SEED`, `CONFIG_2A_TARGET_TOKENS` constants and their heuristic comment block. In their place:

```python
# Registered corpus hashes for the two longctx configurations (PREREG §3.1).
# 2a (PG-essays): prompts are "never displayed on the site and never
# published; outputs and statistics only" — any run recording this corpus is
# refused. 2b (PG-1184 stripped) is the only publishable longctx corpus.
CORPUS_2A_SHA256 = (
    "b6135331a3132d08cb84262870ae8f9d9acb6bae4cd7f0278926a64c38f9329e"
)
```

(`CORPUS_2B_STRIPPED_SHA256` already exists — keep it.)

In `RunData`, add a field: `run_config: dict | None = None`.

In `load_run`, after reading the manifest: `run_config = manifest.pop("_run_config", None)` and pass `run_config=run_config` to the `RunData(...)` constructor. (Popping keeps every downstream `manifest.keys()` iteration rung-only.)

Replace `_check_2a_signature` with:

```python
def _check_longctx_corpus_provenance(run: RunData, run_dir: Path) -> None:
    """Positive provenance check (pre-0B ticket, freeze-plan §10; replaces
    the retired seed-signature heuristic of OPEN_QUESTIONS §4b — that
    heuristic became wrong the moment PREREG Amendment 1 §A registered 2b at
    the same n=96/seed=2024 as 2a). The run's manifest must positively
    record which corpus built its longctx items; only the registered 2b
    corpus (PG-1184) is publishable."""
    recorded = (
        ((run.run_config or {}).get("suites", {}) or {})
        .get(LONGCTX_SUITE, {})
        .get("corpus_sha256")
    )
    if not recorded:
        raise EmbargoViolation(
            f"refusing to package {run_dir}: run contains longctx_retrieval "
            f"items but its manifest records no corpus provenance "
            f"(manifest.json _run_config.suites.longctx_retrieval."
            f"corpus_sha256 is missing). Only runs positively recorded "
            f"against the registered 2b corpus "
            f"({CORPUS_2B_STRIPPED_SHA256}) are publishable (PREREG §3.1, "
            f"§11); regenerate the run with a pipeline that records "
            f"_run_config, or amend the manifest from the run's own "
            f"provenance records."
        )
    if recorded == CORPUS_2A_SHA256:
        raise EmbargoViolation(
            f"refusing to package {run_dir}: manifest records PREREG §3.1 "
            f"config 2a's corpus ({CORPUS_2A_SHA256}) — 2a's prompts 'are "
            f"never displayed on the site and never published; outputs and "
            f"statistics only' (PREREG §3.1)."
        )
    if recorded != CORPUS_2B_STRIPPED_SHA256:
        raise EmbargoViolation(
            f"refusing to package {run_dir}: manifest records an "
            f"unrecognized corpus sha256 ({recorded}); the only publishable "
            f"longctx corpus is the registered 2b corpus "
            f"({CORPUS_2B_STRIPPED_SHA256}) (PREREG §3.1, §11)."
        )
```

At the call site in `package(...)`, replace `_check_2a_signature(longctx_combos, run_dir)` with `_check_longctx_corpus_provenance(run, run_dir)` — keep it exactly where the old call sits (before anything is written, before the tokenizer/corpus-path requirement check).

Also update the module docstring's heuristic description (the "heuristic pending provenance" paragraph) to describe the positive check.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass. Also re-run the real packaged pilot as a smoke check (pilot-0a has no longctx items, so it must package exactly as before):
`uv run python scripts/package_dataset.py runs/pilot-0a --out /tmp/claude-dataset-smoke && rm -rf /tmp/claude-dataset-smoke`
Expected: completes without error.

- [ ] **Step 5: Update the OPEN_QUESTIONS cross-reference**

In `OPEN_QUESTIONS.md`, find the §4b resolution line ("config-2a refusal heuristic kept with an in-code revisit note; the durable corpus-pointer-in-items fix is a ticketed later work item") and append to that line: `(ticket closed 2026-08-29: manifest _run_config.corpus_sha256 + positive provenance check replaced the heuristic — see freeze-plan §10)`. Do not alter the original text, only append.

- [ ] **Step 6: Commit**

```bash
git add scripts/package_dataset.py tests/test_package_dataset.py ../../OPEN_QUESTIONS.md
git commit -m "feat: positive longctx corpus-provenance check replaces 2a seed heuristic (pre-0B ticket)"
```

---

### Task 3: Truncation preflight — assert create_completion reports finish_reason="length"

**Files:**
- Modify: `src/bitcliff_pipeline/generate.py` (new constants + `assert_truncation_finish_reason`)
- Modify: `src/bitcliff_pipeline/__main__.py` (call preflight per loaded model in generate stage)
- Test: `tests/test_generate.py`, `tests/test_cli.py` (fakes updated)

**Interfaces:**
- Produces: in `generate.py`: `TRUNCATION_PREFLIGHT_PROMPT = "Count upward forever: 1, 2, 3, 4, 5, 6, 7,"`, `TRUNCATION_PREFLIGHT_MAX_TOKENS = 8`, and `assert_truncation_finish_reason(llm) -> None` (raises `RuntimeError` unless a deliberately-truncated completion reports `finish_reason == "length"`). Requires `llm.tokenize(bytes, add_bos=..., special=...)` and `llm.create_completion(prompt=list[int], ...)` — the real `llama_cpp.Llama` API. Task 4 reuses this function against a real model.
- Consumes: nothing from Tasks 1–2 (independent).

- [ ] **Step 1: Write the failing unit tests**

Append to `tests/test_generate.py`:

```python
import pytest

from bitcliff_pipeline.generate import (
    TRUNCATION_PREFLIGHT_MAX_TOKENS,
    assert_truncation_finish_reason,
)


class PreflightFakeLlm:
    def __init__(self, finish_reason):
        self._finish_reason = finish_reason
        self.completion_kwargs = None

    def tokenize(self, text, add_bos=True, special=False):
        return list(range(len(text.split())))

    def create_completion(self, prompt, **kwargs):
        self.completion_kwargs = {"prompt": prompt, **kwargs}
        return {"choices": [{"text": "1, 2, 3", "finish_reason": self._finish_reason}]}


def test_truncation_preflight_passes_on_length():
    llm = PreflightFakeLlm("length")
    assert_truncation_finish_reason(llm)  # must not raise
    # the probe really was a deliberately-truncated token-path completion
    assert llm.completion_kwargs["max_tokens"] == TRUNCATION_PREFLIGHT_MAX_TOKENS
    assert isinstance(llm.completion_kwargs["prompt"], list)
    # deterministic decode settings, PREREG §6
    assert llm.completion_kwargs["temperature"] == 0.0
    assert llm.completion_kwargs["top_k"] == 1


@pytest.mark.parametrize("bad", ["stop", None])
def test_truncation_preflight_raises_on_wrong_finish_reason(bad):
    with pytest.raises(RuntimeError, match="finish_reason"):
        assert_truncation_finish_reason(PreflightFakeLlm(bad))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_generate.py -q`
Expected: FAIL with `ImportError: cannot import name 'TRUNCATION_PREFLIGHT_MAX_TOKENS'`.

- [ ] **Step 3: Implement in `src/bitcliff_pipeline/generate.py`**

```python
# Pre-0B ticket (freeze-plan §10): grading.py derives every record's
# `truncated` flag from `finish_reason == "length"`. Verify, on the actual
# llm instance about to generate, that a deliberately-truncated
# create_completion (the token path confirmatory longctx runs use) really
# reports "length" — if a llama-cpp-python version ever stops populating
# it, every truncation would silently grade as a clean stop.
TRUNCATION_PREFLIGHT_PROMPT = "Count upward forever: 1, 2, 3, 4, 5, 6, 7,"
TRUNCATION_PREFLIGHT_MAX_TOKENS = 8


def assert_truncation_finish_reason(llm) -> None:
    tokens = llm.tokenize(
        TRUNCATION_PREFLIGHT_PROMPT.encode("utf-8"), add_bos=True, special=False
    )
    out = llm.create_completion(
        prompt=list(tokens),
        max_tokens=TRUNCATION_PREFLIGHT_MAX_TOKENS,
        temperature=0.0,
        top_k=1,
        seed=42,
    )
    finish_reason = out["choices"][0].get("finish_reason")
    if finish_reason != "length":
        raise RuntimeError(
            f"truncation preflight failed: a create_completion capped at "
            f"{TRUNCATION_PREFLIGHT_MAX_TOKENS} tokens reported "
            f"finish_reason={finish_reason!r}, not 'length' — the truncated "
            f"flag (grading.py) cannot be trusted on this "
            f"llama-cpp-python build; aborting before any confirmatory "
            f"generation (pre-0B ticket, freeze-plan §10)"
        )
```

In `src/bitcliff_pipeline/__main__.py`, in the generate stage, right after `llm = llm_factory(path, config.generation)`, add:

```python
            gen_mod.assert_truncation_finish_reason(llm)
```

- [ ] **Step 4: Update the CLI test fakes (they now receive the preflight call)**

In `tests/test_cli.py`, add to `FakeLlm`:

```python
    def tokenize(self, text, add_bos=True, special=False):
        return list(range(len(text.split())))

    def create_completion(self, prompt, max_tokens, **kwargs):
        # the truncation preflight (generate stage) probes via the token
        # path with a small budget and expects an honest "length"
        return {"choices": [{"text": "1, 2, 3", "finish_reason": "length"}]}
```

`TokenAwareFakeLlm.create_completion` overrides that with `finish_reason: "stop"` for real items; make it answer the preflight honestly by branching on the exported budget:

```python
class TokenAwareFakeLlm(FakeLlm):
    """Extends FakeLlm with create_completion for token-id items. Records
    the exact token list it was called with. Answers the truncation
    preflight (identified by its registered probe budget) honestly."""

    def create_completion(self, prompt, max_tokens=None, **kwargs):
        if max_tokens == gen_mod.TRUNCATION_PREFLIGHT_MAX_TOKENS:
            return super().create_completion(prompt, max_tokens, **kwargs)
        return {"choices": [{"text": "The passcode is 42.", "finish_reason": "stop"}]}
```

with `from bitcliff_pipeline import generate as gen_mod` added to the test file's imports. Also update `tests/test_generate.py`'s existing `FakeLlm` (the one in that file) if `run_items` tests break — they call `run_items` directly, which does NOT run the preflight, so they should not need changes; verify.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass (Task 1–2 counts + 3 new).

- [ ] **Step 6: Commit**

```bash
git add src/bitcliff_pipeline/generate.py src/bitcliff_pipeline/__main__.py tests/test_generate.py tests/test_cli.py
git commit -m "feat: truncation preflight — assert finish_reason='length' before generation (pre-0B ticket)"
```

---

### Task 4: Real-model integration verification of the truncation preflight

**Files:**
- Create: `tests/test_truncation_integration.py`

**Interfaces:**
- Consumes: Task 3's `assert_truncation_finish_reason`, `TRUNCATION_PREFLIGHT_MAX_TOKENS`.
- Produces: an integration test, auto-skipped when the local model file is absent, that proves the ACTUAL llama-cpp-python build populates `finish_reason="length"` on truncation via the token path. This is the ticket's "verify" — the FakeLlm tests alone prove nothing about llama-cpp.

- [ ] **Step 1: Write the integration test**

Create `tests/test_truncation_integration.py`:

```python
"""Pre-0B ticket (freeze-plan §10): verify the REAL llama-cpp-python build
populates finish_reason='length' on truncation via the token path
(create_completion with a token-id prompt) — grading.py's `truncated` flag
depends on it. Runs against the smallest local GGUF; skipped when the model
file is absent (e.g. CI without model assets). The same
assert_truncation_finish_reason preflight runs on the 0B cloud instance
before every confirmatory generation, so this check re-executes on the
exact serving build too.
"""

from pathlib import Path

import pytest

from bitcliff_pipeline.generate import (
    TRUNCATION_PREFLIGHT_MAX_TOKENS,
    assert_truncation_finish_reason,
)

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "Qwen2.5-1.5B-Instruct-IQ1_S-bitcliff-inhouse.gguf"
)

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"local model not present: {MODEL_PATH}",
)


@pytest.fixture(scope="module")
def llm():
    from llama_cpp import Llama

    return Llama(model_path=str(MODEL_PATH), n_ctx=512, seed=42, verbose=False)


def test_real_llama_cpp_reports_length_on_truncation(llm):
    assert_truncation_finish_reason(llm)  # raises if finish_reason != "length"


def test_real_llama_cpp_token_path_truncation_direct(llm):
    """Belt-and-braces: same assertion without going through the preflight
    helper, so a bug in the helper cannot mask a llama-cpp regression."""
    tokens = llm.tokenize(b"Count upward forever: 1, 2, 3, 4, 5, 6, 7,", add_bos=True)
    out = llm.create_completion(
        prompt=list(tokens),
        max_tokens=TRUNCATION_PREFLIGHT_MAX_TOKENS,
        temperature=0.0,
        top_k=1,
        seed=42,
    )
    assert out["choices"][0]["finish_reason"] == "length"
```

- [ ] **Step 2: Run the integration test for real (the model file exists locally)**

Run: `uv run pytest tests/test_truncation_integration.py -v`
Expected: 2 PASSED (not skipped — the IQ1_S file is on disk). This execution IS the ticket's verification; record the llama-cpp-python version in the commit message:
`uv run python -c "import importlib.metadata as m; print(m.version('llama-cpp-python'))"`

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass, including the two integration tests.

- [ ] **Step 4: Commit**

```bash
git add tests/test_truncation_integration.py
git commit -m "test: real-model verification that llama-cpp reports finish_reason='length' on truncation (llama-cpp-python <version>)"
```

---

## Self-Review

- **Spec coverage:** ticket A (corpus_sha256 + suite config in manifest → Task 1; heuristic replaced by positive check → Task 2, including the exact false-refusal case OPEN_QUESTIONS §4b feared). Ticket B (verify finish_reason='length' → Task 3 preflight wired before confirmatory generation + Task 4 real-model proof). §4b cross-reference updated (Task 2 Step 5).
- **Placeholders:** none; every step carries runnable code/commands.
- **Type consistency:** `_run_config` shape `{"model_id": str, "suites": dict}` used identically in Tasks 1 and 2; `TRUNCATION_PREFLIGHT_MAX_TOKENS`/`assert_truncation_finish_reason` names identical in Tasks 3 and 4.
