# Evidence: GGUF headers of the pinned official Qwen2.5-7B-Instruct GGUFs

Purpose: settle OPEN_QUESTIONS.md §10 (imatrix status of the two official
Qwen files run as PREREG §5 Arm 2). Collected 2026-09-08. Read-only: only the
first 16 MiB of each file was fetched by HTTP range request; nothing was
downloaded in full, nothing was run from the repo, no model was loaded.

Pinned source: `reference-manifests/qwen2.5-7b-official.json` — repo
`Qwen/Qwen2.5-7B-Instruct-GGUF`, revision
`bb5d59e06d9551d752d08b292a50eb208b07ab1f`. Both `resolve/` requests returned
`x-repo-commit: bb5d59e06d9551d752d08b292a50eb208b07ab1f` and redirected
(HTTP 302) to Hugging Face's own CDN host `us.aws.cdn.hf.co`, which served
HTTP 206 partial content.

Method: `curl -L -r 0-16777215` against each URL below; the prefix was parsed
with a stdlib-only GGUF v3 key/value reader (magic, version, tensor count, KV
count, then every KV in order). In both files the entire KV block was consumed
within the fetched bytes (`header_bytes_consumed` below), so no metadata key
lies beyond what was read. Tokenizer arrays (`tokenizer.ggml.tokens`,
`tokenizer.ggml.token_type`, `tokenizer.ggml.merges`) were parsed but are not
listed. Manifest `sha256` values are quoted from the manifest; they were not
recomputed here (a prefix cannot reproduce a whole-file hash).

## File 1: `qwen2.5-7b-instruct-q3_k_m.gguf`

- URL: `https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/bb5d59e06d9551d752d08b292a50eb208b07ab1f/qwen2.5-7b-instruct-q3_k_m.gguf`
- Bytes fetched: 16,777,216 (`content-range: bytes 0-16777215/3808391072`)
- Manifest sha256 (whole file): `a96b16179dc6cc9afdf0cf7a96a80c199cbd00b9be207c3465be21cb721cca5e`; size_bytes 3,808,391,072
- GGUF version 3; n_tensors 339; n_kv 26; header_bytes_consumed 5,934,129

`general.*` keys (all that exist):

| key | type | value |
|---|---|---|
| general.architecture | string | `qwen2` |
| general.type | string | `model` |
| general.name | string | `qwen2.5-7b-instruct` |
| general.version | string | `v0.1` |
| general.finetune | string | `qwen2.5-7b-instruct` |
| general.size_label | string | `7.6B` |
| general.file_type | uint32 | 12 (LLAMA_FTYPE_MOSTLY_Q3_K_M) |
| general.quantization_version | uint32 | 2 |

`quantize.*` keys: **none**. `quantize.imatrix.*` keys: **none present.**

Other keys present (names only): qwen2.block_count, qwen2.context_length,
qwen2.embedding_length, qwen2.feed_forward_length,
qwen2.attention.head_count, qwen2.attention.head_count_kv,
qwen2.rope.freq_base, qwen2.attention.layer_norm_rms_epsilon,
tokenizer.ggml.model, tokenizer.ggml.pre, tokenizer.ggml.tokens,
tokenizer.ggml.token_type, tokenizer.ggml.merges, tokenizer.ggml.eos_token_id,
tokenizer.ggml.padding_token_id, tokenizer.ggml.bos_token_id,
tokenizer.ggml.add_bos_token, tokenizer.chat_template.

## File 2: `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` (first shard of the Q4_K_M split)

- URL: `https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/bb5d59e06d9551d752d08b292a50eb208b07ab1f/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`
- Bytes fetched: 16,777,216 (`content-range: bytes 0-16777215/3993201344`)
- Manifest sha256 (whole file): `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db`; size_bytes 3,993,201,344
- GGUF version 3; n_tensors 280 (this shard); n_kv 29; header_bytes_consumed 5,934,211

`general.*` keys (all that exist):

| key | type | value |
|---|---|---|
| general.architecture | string | `qwen2` |
| general.type | string | `model` |
| general.name | string | `qwen2.5-7b-instruct` |
| general.version | string | `v0.1` |
| general.finetune | string | `qwen2.5-7b-instruct` |
| general.size_label | string | `7.6B` |
| general.file_type | uint32 | 15 (LLAMA_FTYPE_MOSTLY_Q4_K_M) |
| general.quantization_version | uint32 | 2 |

`quantize.*` keys: **none**. `quantize.imatrix.*` keys: **none present.**

Other keys present (names only): the same qwen2.* and tokenizer.* keys as
File 1, plus split.no = 0, split.count = 2, split.tensors.count = 339. The
second shard (`-00002-of-00002.gguf`) was not fetched; GGUF split metadata is
carried by the first shard, and the pinned config loads the first shard.

## Finding

Neither pinned official file carries any `quantize.imatrix.*` key
(`quantize.imatrix.file`, `quantize.imatrix.dataset`,
`quantize.imatrix.entries_count`, `quantize.imatrix.chunks_count`), nor any
other `quantize.*` key. No converter or quantizer provenance keys are present
either: no `general.source.*`, `general.base_model.*`, `general.url`,
`general.repo_url`, `general.license*`, or `general.quantized_by` key exists
in either header. The only quantization-related key is
`general.quantization_version = 2` in both files, with `general.file_type`
12 (Q3_K_M) and 15 (Q4_K_M).

Caveat: the absence of `quantize.imatrix.*` keys is evidence of no
importance-matrix calibration only for files written by a llama.cpp
`llama-quantize` build that records those keys (added upstream in
llama.cpp PR #6658, April 2024). These files carry no key identifying the
quantizer build or commit, so that condition cannot be confirmed from the
header alone; `general.quantization_version = 2` is the only version
indicator present and does not date the writing build. The ruling in
OPEN_QUESTIONS.md §10 rests on this evidence with that caveat attached.
