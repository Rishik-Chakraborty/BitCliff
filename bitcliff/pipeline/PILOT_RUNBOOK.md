# Phase 0A Pilot Runbook — Qwen2.5-1.5B ladder, local hardware

**This run is EXPLORATORY (spec §10).** Its numbers are thrown away. They may inform
curation, the length budget, and parameter choices — and that fact gets declared in the
registration. Nothing here is confirmatory. Zero cloud spend.

## 1. One-time setup

```bash
cd bitcliff/pipeline
uv sync
```

Convert F16 locally (the reference; never a download recommendation):

```bash
# ~3 GB download + ~3 GB output; needs the llama.cpp conversion script
git clone --depth 1 https://github.com/ggml-org/llama.cpp ../llama.cpp
uv run --with-requirements ../llama.cpp/requirements/requirements-convert_hf_to_gguf.txt \
  hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/hf/Qwen2.5-1.5B-Instruct
uv run --with-requirements ../llama.cpp/requirements/requirements-convert_hf_to_gguf.txt \
  python ../llama.cpp/convert_hf_to_gguf.py models/hf/Qwen2.5-1.5B-Instruct \
  --outtype f16 --outfile models/f16/Qwen2.5-1.5B-Instruct-f16.gguf
```

(If the `--with-requirements` incantation fights back, make a scratch venv for the converter;
it only runs once. Any working conversion is fine — the F16 gets hashed either way.)

Sanity-check the ladder still matches the uploader's repo (filenames drift when
uploaders re-quantize; the manifest will catch silent content changes):

```bash
uv run python -c "
from huggingface_hub import list_repo_files
print('\n'.join(sorted(f for f in list_repo_files('bartowski/Qwen2.5-1.5B-Instruct-GGUF') if f.endswith('.gguf'))))"
```

## 2. The run

```bash
uv run python -m bitcliff_pipeline configs/qwen2.5-1.5b-pilot.yaml \
  --run-id pilot-0a --stage all
```

Notes:
- Ladder = 8 rungs × ~90 items (~40 retrieval, 40 arithmetic, 10 spectacle).
  Interrupt any time; `--stage generate` resumes, skipping finished rungs.
- Total quant downloads ≈ 8 GB. F16 is the slowest rung; start it before dinner.

## 3. Length-budget check (spec §6: the budget must be genuinely generous)

```bash
uv run python -c "
from pathlib import Path
from bitcliff_pipeline.generate import read_records
recs = read_records(Path('runs/pilot-0a/outputs/F16.jsonl'))
trunc = [r.item_id for r in recs if r.finish_reason == 'length']
print(f'{len(trunc)}/{len(recs)} F16 outputs truncated'); print(trunc)"
```

If ANY F16 output is truncated, raise `max_tokens` in the config, delete
`runs/pilot-0a/outputs/`, and rerun. The frozen budget must never clip full precision.

## 4. Read the curves (the whole point)

Open `runs/pilot-0a/retention.png` and `results.csv`. Read raw outputs for the
low rungs: `runs/pilot-0a/outputs/IQ2_M.jsonl` — this is where curation candidates
for the launch-50 come from.

## 5. The gate (spec §10 "Bars")

- **Spectacle bar:** does the 1.5B show visible drama on curated prompts by IQ2_M —
  forgotten facts, loops, salad? If yes, the spectacle half has content.
- **Signal bar:** are the retrieval/arithmetic curves informative? Flat-then-cliff
  COUNTS as informative. Only "every curve is noise" kills the project.
- Record the verdict, the chosen length budget, the run-id, and curation notes in
  `PILOT_NOTES.md` next to this runbook (committed; the run outputs themselves stay gitignored).

**If the gate passes → next plan: the freeze** (registration text, twin problems,
margins, license audit, file lists+hashes for the 8B ladders). Nothing confirmatory
runs and no GPU money is spent before that commit (spec §10; aws-ops).

**If the bottom rungs aren't deranged enough:** quantize in-house bottom rungs
(spec §4 exception, 1.5B only, clearly labeled) with llama.cpp's `llama-quantize`
and add them to the config with `uploader: bitcliff-inhouse, imatrix: false`.
