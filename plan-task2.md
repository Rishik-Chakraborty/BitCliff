# In-House Deranged Rungs (Task 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Quantize in-house sub-2-bit rungs (IQ1_S, IQ1_M, IQ2_XXS) for the 1.5B spectacle ladder with full provenance, extend the pilot ladder, rerun only the new rungs, and report whether the deranged zone is now actually deranged.

**Architecture:** One small schema/code change (spectacle-only labeling + metadata-enriched manifest), then operational work with the already-cloned llama.cpp (pinned commit `bf942164697d2d62c2237a17b677dc2c017ea8e7`): build the quantize/imatrix tools, generate an imatrix from a recorded public calibration text, quantize three files, and resume the existing `pilot-0a` run so only the three new rungs generate.

**Tech Stack:** existing bitcliff_pipeline (uv/pytest), llama.cpp (cmake + Metal), datasets (wikitext-2 calibration).

**Spec:** user directives (2026-08-26) + `claude/IDEA.md` §4 (in-house exception: 1.5B only, clearly labeled, published-files-only rule protects download recommendations — these files must never appear in any recommendation surface) + `bitcliff/pipeline/PILOT_RUNBOOK.md` fallback section.

## Global Constraints

- In-house files: `uploader: bitcliff-inhouse`, `imatrix: true`, `spectacle_only: true`; the manifest must record uploader, imatrix status, and SHA-256 per file. Spectacle-only files must be excludable from any future recommendation surface via the `spectacle_only` flag.
- Provenance recorded in `bitcliff/pipeline/INHOUSE_QUANTS.md`: llama.cpp commit hash `bf942164697d2d62c2237a17b677dc2c017ea8e7`, build command, calibration text source + extraction rule + SHA-256, imatrix command + SHA-256, per-quant SHA-256.
- The pilot rerun keeps `max_tokens: 640` (internal consistency of the throwaway run; the registered 1024 applies to confirmatory configs only). Same `--run-id pilot-0a`; only the three new rungs may generate — the 8 existing `outputs/*.jsonl` must be skipped.
- Zero cloud spend; all local. Pilot numbers remain exploratory.
- Tooling: uv + pytest; TDD for the code task; commit per task.

## File Structure

- Modify: `bitcliff/pipeline/src/bitcliff_pipeline/config.py` (QuantFile gains `spectacle_only: bool = False`)
- Modify: `bitcliff/pipeline/src/bitcliff_pipeline/__main__.py` (download stage enriches manifest entries with uploader/imatrix/spectacle_only)
- Modify: `bitcliff/pipeline/tests/test_config.py`, `bitcliff/pipeline/tests/test_cli.py` (new assertions)
- Modify: `bitcliff/pipeline/configs/qwen2.5-1.5b-pilot.yaml` (3 new rungs)
- Create: `bitcliff/pipeline/INHOUSE_QUANTS.md` (provenance), `bitcliff/pipeline/calibration/` (gitignored text), `bitcliff/pipeline/models/imatrix/` (gitignored)
- Modify: `bitcliff/pipeline/PILOT_NOTES.md` (deranged-zone verdict appended)

---

### Task 1: spectacle_only flag + metadata-enriched manifest

**Files:**
- Modify: `bitcliff/pipeline/src/bitcliff_pipeline/config.py`
- Modify: `bitcliff/pipeline/src/bitcliff_pipeline/__main__.py`
- Test: `bitcliff/pipeline/tests/test_config.py`, `bitcliff/pipeline/tests/test_cli.py`

**Interfaces:**
- Consumes: existing `QuantFile(label, filename, uploader, imatrix)`, `build_manifest`, download stage in `run_pipeline`.
- Produces: `QuantFile(label, filename, uploader, imatrix, spectacle_only: bool = False)` (default keeps every existing positional construction working). Manifest entries gain three keys: `uploader`, `imatrix`, `spectacle_only` — from config for quant labels; for `"F16"` use `uploader: "bitcliff-local-f16-conversion"`, `imatrix: false`, `spectacle_only: false`. `verify_manifest` is untouched (it only compares `sha256`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
def test_spectacle_only_defaults_false_and_parses(tmp_path):
    yaml_text = VALID_YAML.replace(
        "  - {label: Q4_K_M, filename: Qwen2.5-1.5B-Instruct-Q4_K_M.gguf, uploader: bartowski, imatrix: true}",
        "  - {label: IQ1_S, filename: x-IQ1_S.gguf, uploader: bitcliff-inhouse, imatrix: true, spectacle_only: true}",
    )
    cfg = load_config(write_yaml(tmp_path, yaml_text))
    assert cfg.quants[0].spectacle_only is False   # omitted -> default
    assert cfg.quants[1].spectacle_only is True
```

Add to `tests/test_cli.py` (inside `test_pipeline_end_to_end`, after the manifest existence assert):

```python
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["Q4_K_M"]["uploader"] == "bartowski"
    assert manifest["Q4_K_M"]["imatrix"] is True
    assert manifest["Q4_K_M"]["spectacle_only"] is False
    assert manifest["F16"]["uploader"] == "bitcliff-local-f16-conversion"
    assert manifest["F16"]["spectacle_only"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py tests/test_cli.py -v` — expect the new assertions to fail (TypeError on unexpected key / KeyError on manifest keys).

- [ ] **Step 3: Implement**

`config.py`: add `spectacle_only: bool = False` as the last field of `QuantFile`.

`__main__.py` download stage — after `manifest = build_manifest(paths)` (build it into a variable before writing), enrich:

```python
        meta = {
            q.label: {"uploader": q.uploader, "imatrix": q.imatrix, "spectacle_only": q.spectacle_only}
            for q in config.quants
        }
        meta["F16"] = {"uploader": "bitcliff-local-f16-conversion", "imatrix": False, "spectacle_only": False}
        for label, entry in manifest.items():
            entry.update(meta[label])
        write_manifest(manifest, manifest_path)
```

- [ ] **Step 4: Full suite green** — `uv run pytest -q` (all 43+ pass).

- [ ] **Step 5: Commit** — `feat(pipeline): spectacle_only flag and metadata-enriched manifest`

---

### Task 2: build tools, imatrix, quantize the three in-house files

Operational task; no TDD. Work from `bitcliff/pipeline`. Record every output listed below in `INHOUSE_QUANTS.md`.

- [ ] **Step 1: Verify the pin and build llama.cpp tools**

```bash
git -C ../llama.cpp rev-parse HEAD   # must print bf942164697d2d62c2237a17b677dc2c017ea8e7
cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_METAL=on -DCMAKE_BUILD_TYPE=Release
cmake --build ../llama.cpp/build -j --target llama-quantize llama-imatrix
```

- [ ] **Step 2: Produce the calibration text (recorded + hashed)**

Source: `Salesforce/wikitext`, config `wikitext-2-raw-v1`, split `train` (standard public calibration corpus). Extraction rule: concatenate the `text` field of all rows with `"\n"`, take the first 500,000 characters.

```bash
mkdir -p calibration
uv run python -c "
from datasets import load_dataset
ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
text = '\n'.join(r['text'] for r in ds)[:500000]
open('calibration/wikitext2-raw-v1-train-first500k.txt', 'w').write(text)
print(len(text), 'chars')"
shasum -a 256 calibration/wikitext2-raw-v1-train-first500k.txt
```

Add `calibration/` to `bitcliff/pipeline/.gitignore` (the hash + rule in INHOUSE_QUANTS.md is the record; the text is re-derivable).

- [ ] **Step 3: Generate the imatrix from F16**

```bash
mkdir -p models/imatrix
../llama.cpp/build/bin/llama-imatrix \
  -m models/f16/Qwen2.5-1.5B-Instruct-f16.gguf \
  -f calibration/wikitext2-raw-v1-train-first500k.txt \
  -o models/imatrix/qwen2.5-1.5b-wikitext2-first500k.imatrix \
  -ngl 99
shasum -a 256 models/imatrix/qwen2.5-1.5b-wikitext2-first500k.imatrix
```

(If the binary lands elsewhere under `../llama.cpp/build`, find it with `find ../llama.cpp/build -name 'llama-imatrix'` and use that path; record the actual command run.)

- [ ] **Step 4: Quantize IQ1_S, IQ1_M, IQ2_XXS**

```bash
for Q in IQ1_S IQ1_M IQ2_XXS; do
  ../llama.cpp/build/bin/llama-quantize \
    --imatrix models/imatrix/qwen2.5-1.5b-wikitext2-first500k.imatrix \
    models/f16/Qwen2.5-1.5B-Instruct-f16.gguf \
    models/Qwen2.5-1.5B-Instruct-$Q-bitcliff-inhouse.gguf $Q
done
shasum -a 256 models/Qwen2.5-1.5B-Instruct-IQ*-bitcliff-inhouse.gguf
ls -lh models/Qwen2.5-1.5B-Instruct-IQ*-bitcliff-inhouse.gguf
```

If a scheme fails to quantize for this architecture, record the exact error in INHOUSE_QUANTS.md and proceed with the schemes that succeed (IQ2_XXS is "if trivial"; IQ1_S/IQ1_M are the goal).

- [ ] **Step 5: Write `INHOUSE_QUANTS.md`** — provenance for spectacle-only in-house quants: purpose (spectacle only, never a download recommendation, per spec §4 exception), llama.cpp commit + build command, calibration source/rule/sha256, imatrix command + sha256, table of produced files with scheme, size, sha256.

- [ ] **Step 6: Extend the ladder config** — append to `configs/qwen2.5-1.5b-pilot.yaml` quants:

```yaml
  - {label: IQ2_XXS, filename: Qwen2.5-1.5B-Instruct-IQ2_XXS-bitcliff-inhouse.gguf, uploader: bitcliff-inhouse, imatrix: true, spectacle_only: true}
  - {label: IQ1_M,  filename: Qwen2.5-1.5B-Instruct-IQ1_M-bitcliff-inhouse.gguf,  uploader: bitcliff-inhouse, imatrix: true, spectacle_only: true}
  - {label: IQ1_S,  filename: Qwen2.5-1.5B-Instruct-IQ1_S-bitcliff-inhouse.gguf,  uploader: bitcliff-inhouse, imatrix: true, spectacle_only: true}
```

(Only include labels that actually quantized in Step 4.) Verify: `uv run python -c "from bitcliff_pipeline.config import load_config; print([q.label for q in load_config('configs/qwen2.5-1.5b-pilot.yaml').quants])"`

- [ ] **Step 7: Commit** — `feat(pipeline): in-house IQ1/IQ2 spectacle rungs with recorded provenance` (INHOUSE_QUANTS.md, config, .gitignore; model/calibration binaries stay untracked).

---

### Task 3: resume-run the new rungs, verdict the deranged zone

- [ ] **Step 1: Rerun with the same run-id** — `uv run python -m bitcliff_pipeline configs/qwen2.5-1.5b-pilot.yaml --run-id pilot-0a --stage all`. Verify from the output that all 8 pre-existing rungs print `skip <LABEL>: ... exists` and ONLY the new labels print `generating ...`. `max_tokens` stays 640. Grade + report rerun over all rungs (the enriched manifest is rewritten; the prompt tripwire guards staleness).

- [ ] **Step 2: Verify the manifest** — `runs/pilot-0a/manifest.json` now carries uploader/imatrix/spectacle_only for every entry; in-house entries say `bitcliff-inhouse` / `true` / `true`.

- [ ] **Step 3: Deranged-zone verdict** — read `results.csv` rows for the new rungs; eyeball at least 4 spectacle outputs and 2 arithmetic outputs per new rung (`runs/pilot-0a/outputs/IQ1_S.jsonl` etc.). The question: is IQ1_S/IQ1_M output visibly deranged (salad, loops, forgotten facts) in a way IQ2_M was not? Quote 2-3 of the most dramatic excerpts.

- [ ] **Step 4: Append the verdict to `PILOT_NOTES.md`** — new section "## Deranged-zone verdict (in-house rungs)": per-rung accuracy rows, the spectacle assessment with quoted excerpts, and whether the spectacle bar now passes. Scored-suite numbers for spectacle_only rungs are noted as spectacle-context only (these rungs never enter reference tables).

- [ ] **Step 5: Commit** — `docs(pipeline): deranged-zone verdict for in-house rungs`

## Out of scope

The freeze plan (Task 1 of the user's directive) is a separate plan-first document; nothing from it executes here.
