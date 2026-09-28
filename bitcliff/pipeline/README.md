# bitcliff-pipeline

The BitCliff measurement pipeline. `PREREG.md` (repo root) governs every
confirmatory run; this file only says how to drive the code.

## Entry points

```bash
uv run python -m bitcliff_pipeline <config.yaml> --run-id <id> \
    --stage {download,generate,grade,report,all} \
    [--models-dir models] [--runs-dir runs]
```

- **configs:** `configs/0b/` (0B runs, as run), `configs/0b2/` (factual_qa
  rerun on the registered item set), `configs/smoke/` (smoke,
  `exploratory: true`), plus the exploratory pilot configs at the top level.
- **download:** fetches every quant from Hugging Face and checks it against
  the config-pinned sha256, then writes `manifest.json`. F16 is a local
  conversion (`f16_path`) and is never downloaded.
- **generate:** re-verifies the manifest, builds items, runs the boot gates
  (below), and generates with llama-cpp-python (greedy). It writes
  `outputs/<label>.jsonl`, and a label whose output file already exists is
  skipped.
- **grade:** per-suite graders, writing `grades.jsonl`.
- **report:** per-(label, suite) accuracy and retention, written to
  `results.{csv,json}` and `retention.png`.
- **analysis:** `uv run python scripts/analyze_0b.py [--seed 8271]
  [--out-dir DIR]` runs the PREREG §8 machinery (cell states, CIs, McNemar,
  cliffs, Holm, §5 trigger) over `runs-cloud/pipeline/runs/` and writes
  `analysis/0b/cells.csv` and `FINDINGS_0B.md`. The 0B run ids are
  hardcoded.

## Registered constants

Every registered seed, n, weight vector, margin, and hash lives in
`src/bitcliff_pipeline/registered.py`. Never hand-copy them. Configs carry
literals only because YAML cannot import, and tests cross-check those
literals against `registered.py`.

## Boot gates (before any generation)

- **Item-set hash gate:** each suite at its registered n must hash to its
  pinned item-set sha256, or generation refuses. A suite at a
  non-registered n is currently left ungated (`__main__.py` line 65). A
  config with `exploratory: true` skips the gate with a loud warning; that
  is for smoke configs only.
- **Tokenizer match** (longctx): HF vs GGUF token ids on the first 20 items.
- **Truncation preflight:** a capped completion must report
  `finish_reason == "length"`.
- **File hashes:** every quant file is checked against its pinned sha256.

## Tests

```bash
uv run pytest -q
```

Caveat for cold machines: `tests/test_registered.py` rebuilds the
arithmetic item set from HF GSM8K (`openai/gsm8k`) through `datasets`, with
no skip guard. It passes from the local Hugging Face cache and needs
network (or a warm cache) on a fresh machine.
