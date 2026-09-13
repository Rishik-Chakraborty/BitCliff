# Evidence: GGUF headers of the seven pinned bartowski Qwen2.5-1.5B-Instruct rung files

Purpose: document the imatrix status of the seven confirmatory quant rungs
pinned in `reference-manifests/qwen2.5-1.5b-bartowski.json` (Amendment 4).
Collected 2026-09-12. Read-only: only the first 16 MiB of each file was
fetched by HTTP range request; nothing was downloaded in full, nothing was
run from the repo, no model was loaded.

Pinned source: repo `bartowski/Qwen2.5-1.5B-Instruct-GGUF`, revision
`9eadc66189c7641e1ddd226b8267a9119b2ce2d4`. Every `resolve/` request
returned `x-repo-commit: 9eadc66189c7641e1ddd226b8267a9119b2ce2d4` and HTTP
206 partial content for `bytes 0-16777215`.

Method: `curl -L -r 0-16777215` against each URL below; the prefix was parsed
with the same stdlib-only GGUF v3 key/value reader used for
`qwen2.5-7b-official-gguf-headers.md` (magic, version, tensor count, KV count,
then every KV in order). In every file the entire KV block was consumed
within the fetched bytes (`header_bytes_consumed` ≈ 5.93 MB), so no metadata
key lies beyond what was read. Tokenizer arrays and the chat template were
parsed but are not listed. Manifest `sha256` values are the LFS hashes from
the Hugging Face tree API; a prefix cannot reproduce a whole-file hash.

URL pattern for every file:
`https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/9eadc66189c7641e1ddd226b8267a9119b2ce2d4/<file>`

## Keys common to all seven files

`general.*` (values identical across the seven unless noted):

| key | type | value |
|---|---|---|
| general.architecture | string | `qwen2` |
| general.type | string | `model` |
| general.name | string | `Qwen2.5 1.5B Instruct` |
| general.finetune | string | `Instruct` |
| general.basename | string | `Qwen2.5` |
| general.size_label | string | `1.5B` |
| general.license | string | `apache-2.0` |
| general.license.link | string | `https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE` (present in Q8_0, Q6_K, Q5_K_M, Q4_K_M only; absent in Q3_K_M, Q2_K, IQ2_M) |
| general.base_model.count | uint32 | 1 |
| general.base_model.0.name | string | `Qwen2.5 1.5B` |
| general.base_model.0.organization | string | `Qwen` |
| general.base_model.0.repo_url | string | `https://huggingface.co/Qwen/Qwen2.5-1.5B` |
| general.tags | string[2] | `chat`, `text-generation` |
| general.languages | string[1] | `en` |
| general.file_type | uint32 | per file, below |
| general.quantization_version | uint32 | 2 |

`quantize.*` (identical across all seven):

| key | type | value |
|---|---|---|
| quantize.imatrix.file | string | `/models_out/Qwen2.5-1.5B-Instruct-GGUF/Qwen2.5-1.5B-Instruct.imatrix` |
| quantize.imatrix.dataset | string | `/training_dir/calibration_datav3.txt` |
| quantize.imatrix.entries_count | int32 | 196 |
| quantize.imatrix.chunks_count | int32 | 128 |

Other keys present in every file (names only): qwen2.block_count,
qwen2.context_length, qwen2.embedding_length, qwen2.feed_forward_length,
qwen2.attention.head_count, qwen2.attention.head_count_kv,
qwen2.rope.freq_base, qwen2.attention.layer_norm_rms_epsilon,
tokenizer.ggml.model, tokenizer.ggml.pre, tokenizer.ggml.tokens,
tokenizer.ggml.token_type, tokenizer.ggml.merges, tokenizer.ggml.eos_token_id,
tokenizer.ggml.padding_token_id, tokenizer.ggml.bos_token_id,
tokenizer.ggml.add_bos_token, tokenizer.chat_template.

## Per file

| file | bytes fetched | total size (content-range) | manifest sha256 (LFS) | n_tensors | n_kv | general.file_type | quantize.imatrix.* present | quantize.imatrix.dataset |
|---|---|---|---|---|---|---|---|---|
| `Qwen2.5-1.5B-Instruct-Q8_0.gguf` | 16,777,216 | 1,646,573,312 | `7185d306cf45956c8c017cd0d3b05ecc6bc18b3ea8eb5c240dce40e87563db7f` | 338 | 38 | 7 (Q8_0) | yes, all four | `/training_dir/calibration_datav3.txt` |
| `Qwen2.5-1.5B-Instruct-Q6_K.gguf` | 16,777,216 | 1,272,740,096 | `1b01b4ea4ccdd5aa6a5972790002e120a5d500a5175be571114d642a8db4d14e` | 338 | 38 | 18 (Q6_K) | yes, all four | `/training_dir/calibration_datav3.txt` |
| `Qwen2.5-1.5B-Instruct-Q5_K_M.gguf` | 16,777,216 | 1,125,050,624 | `cf240adc57e126e86102335f6565fb23e523b28d287c75bdb0759f064e8bb572` | 338 | 38 | 17 (Q5_K_M) | yes, all four | `/training_dir/calibration_datav3.txt` |
| `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` | 16,777,216 | 986,048,768 | `1adf0b11065d8ad2e8123ea110d1ec956dab4ab038eab665614adba04b6c3370` | 338 | 38 | 15 (Q4_K_M) | yes, all four | `/training_dir/calibration_datav3.txt` |
| `Qwen2.5-1.5B-Instruct-Q3_K_M.gguf` | 16,777,216 | 824,178,784 | `7437ad04011a14fb890074cc783df4c1d537197942337353a824ff7e6115ef9b` | 338 | 37 | 12 (Q3_K_M) | yes, all four | `/training_dir/calibration_datav3.txt` |
| `Qwen2.5-1.5B-Instruct-Q2_K.gguf` | 16,777,216 | 676,304,992 | `a8880f0de2348db67d00519ef7f4b40326ef67012bf5f2e90bd1d47474e2355c` | 338 | 37 | 10 (Q2_K) | yes, all four | `/training_dir/calibration_datav3.txt` |
| `Qwen2.5-1.5B-Instruct-IQ2_M.gguf` | 16,777,216 | 601,054,816 | `cee720c998e71ff3f02fbb3d392c7598bc0f845ae08bbb585ef2ff5fcbd45b81` | 338 | 37 | 29 (IQ2_M) | yes, all four | `/training_dir/calibration_datav3.txt` |

The n_kv difference (38 vs 37) is exactly the presence or absence of
`general.license.link`; the quantize block is identical in all seven.

## Finding

All seven pinned rung files carry `quantize.imatrix.file`,
`quantize.imatrix.dataset`, `quantize.imatrix.entries_count`, and
`quantize.imatrix.chunks_count`. Each was quantized with the importance
matrix `Qwen2.5-1.5B-Instruct.imatrix` (the same file the repo ships, LFS
sha256 `2f85dbb69836a017b9a4dfd0b86ee5b4dae0a0ac9217c57688de13f2dbb16a99`)
computed over the uploader's `calibration_datav3.txt` dataset with 196
entries over 128 chunks. `imatrix: true` for every rung in the manifest is
therefore positive evidence from the files themselves, not an inference from
the repo listing.

Caveat: presence of these keys establishes that the writing `llama-quantize`
build recorded an imatrix; the calibration dataset itself is identified only
by the uploader's path string and is not independently verifiable from the
header.
