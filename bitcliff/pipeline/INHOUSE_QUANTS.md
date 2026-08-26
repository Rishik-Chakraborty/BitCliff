# In-house quants: provenance

## Purpose

These three GGUF files (`IQ1_S`, `IQ1_M`, `IQ2_XXS`) are **spectacle-only** rungs on the
`qwen2.5-1.5b-pilot` evaluation ladder. Per spec §4's exception for extreme sub-2-bit
quantization, they exist to demonstrate visible degradation on the spectacle suite and
are **never a download recommendation**. They are marked `spectacle_only: true` in the
ladder config and are excluded from any recommendation surface.

They were built in-house (not sourced from a third-party uploader) because no public
GGUF repo for `Qwen2.5-1.5B-Instruct` carried imatrix-calibrated `IQ1_S`/`IQ1_M`/`IQ2_XXS`
files at the time of this pipeline run.

## llama.cpp build

- Repo: `../llama.cpp` (relative to `bitcliff/pipeline`), shallow clone.
- Commit (pinned, verified via `git -C ../llama.cpp rev-parse HEAD`):
  `bf942164697d2d62c2237a17b677dc2c017ea8e7`
- Configure command:
  ```
  cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_METAL=on -DCMAKE_BUILD_TYPE=Release
  ```
- Build command (targets only, not the full suite):
  ```
  cmake --build ../llama.cpp/build -j --target llama-quantize llama-imatrix
  ```
- Both targets built cleanly (Metal + Accelerate + BLAS backends detected; no OpenMP,
  non-fatal). Resulting binaries:
  - `../llama.cpp/build/bin/llama-quantize`
  - `../llama.cpp/build/bin/llama-imatrix`
- No flag deviations were needed — `llama-imatrix --help` and `llama-quantize --help`
  on the built binaries confirmed `-o`/`--output`, `-ngl`, and `--imatrix` match the
  brief's commands exactly.

## Calibration text

- Source: `Salesforce/wikitext`, config `wikitext-2-raw-v1`, split `train` (standard
  public calibration corpus).
- Extraction rule: concatenate the `text` field of all rows with `"\n"`, take the first
  500,000 characters.
- Command:
  ```python
  from datasets import load_dataset
  ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
  text = '\n'.join(r['text'] for r in ds)[:500000]
  open('calibration/wikitext2-raw-v1-train-first500k.txt', 'w').write(text)
  ```
- Output: `calibration/wikitext2-raw-v1-train-first500k.txt`, 500,000 characters.
- SHA-256: `466e9b4a7408bfe5e96494c032ddea17c3044155c7c220ab68021988c6cb3025`
- The file itself is untracked (`calibration/` is in `.gitignore`) — it is
  deterministically re-derivable from the source/rule above; this hash is the record.

## Imatrix generation

- Command actually run (matches the brief exactly, no flag substitution needed):
  ```
  ../llama.cpp/build/bin/llama-imatrix \
    -m models/f16/Qwen2.5-1.5B-Instruct-f16.gguf \
    -f calibration/wikitext2-raw-v1-train-first500k.txt \
    -o models/imatrix/qwen2.5-1.5b-wikitext2-first500k.imatrix \
    -ngl 99
  ```
- Ran on Metal (`-ngl 99`, all layers offloaded). 221 chunks, `n_ctx=512`,
  `batch_size=2048`, `n_seq=4`. Wall time: ~43 seconds (`27.69s user 3.68s system`).
- Final calibration perplexity: `PPL = 11.0393 +/- 0.12691`.
- Output: `models/imatrix/qwen2.5-1.5b-wikitext2-first500k.imatrix` (GGUF-format
  imatrix; the tool warned it is "saving imatrix using GGUF format with a different
  suffix than .gguf" — cosmetic, expected, not an error), 2.0 MiB.
- SHA-256: `f99bee1e364baef748b88a8a0f44c88e785890e0e8f310ed4282faf38021797a`
- The imatrix file itself is untracked (`models/` is in `.gitignore`); this hash is
  the record.

## Quantization

Command run for each scheme (loop, matches the brief exactly):
```
for Q in IQ1_S IQ1_M IQ2_XXS; do
  ../llama.cpp/build/bin/llama-quantize \
    --imatrix models/imatrix/qwen2.5-1.5b-wikitext2-first500k.imatrix \
    models/f16/Qwen2.5-1.5B-Instruct-f16.gguf \
    models/Qwen2.5-1.5B-Instruct-$Q-bitcliff-inhouse.gguf $Q
done
```

All three schemes quantized successfully with exit code 0 — no errors. A full-log
re-run of `IQ1_S` (the most aggressive scheme) was captured and grepped for
`warn|error|fail`: no matches. No quality warnings were emitted by `llama-quantize`
for any of the three schemes on this architecture (Qwen2.5-1.5B, 28 layers). Note that
`IQ1_S`/`IQ1_M`/`IQ2_XXS` are, by design, extreme sub-2-bit quantizations expected to
show visible quality degradation on the spectacle suite in the next task — that
degradation is the point of these rungs, not a build-time defect.

### Produced files

| Scheme    | Filename                                              | Size (on disk) | Model-quantize reported size | BPW  | SHA-256 |
|-----------|--------------------------------------------------------|----------------|-------------------------------|------|---------|
| IQ1_S     | `Qwen2.5-1.5B-Instruct-IQ1_S-bitcliff-inhouse.gguf`     | 416 MiB (`ls -lh`) | 410.63 MiB | 2.23 | `5f25c4a0228456f01097f9da09d0d76f80c1bf9aea5088f26a978941ad9aae9b` |
| IQ1_M     | `Qwen2.5-1.5B-Instruct-IQ1_M-bitcliff-inhouse.gguf`     | 443 MiB (`ls -lh`) | 437.27 MiB | 2.38 | `864adcf3f2e1c3c97f42c7424b77d3256004b11314b331b44310b049491cbade` |
| IQ2_XXS   | `Qwen2.5-1.5B-Instruct-IQ2_XXS-bitcliff-inhouse.gguf`   | 487 MiB (`ls -lh`) | 481.67 MiB | 2.62 | `ddb112eec7dddf10ae1b8c22e23c6ce5b5819a01bb9b3ef1319370f0d0735ea5` |

Source F16 model for all quantizations: `models/f16/Qwen2.5-1.5B-Instruct-f16.gguf`
(2944.68 MiB / 16.00 BPW, unquantized baseline, per `llama_model_quantize_impl` log).

All three `.gguf` files are untracked (`models/` is in `.gitignore`); the sizes and
hashes above are the record.

## Ladder config

Appended to `configs/qwen2.5-1.5b-pilot.yaml`'s `quants:` list with `spectacle_only: true`
and `uploader: bitcliff-inhouse` — see that file for the exact entries. All three
labels (`IQ2_XXS`, `IQ1_M`, `IQ1_S`) were included since all three quantized
successfully in the step above.
