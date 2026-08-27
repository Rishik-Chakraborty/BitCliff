# CORPUS_MANIFEST.md — corpora acquisition and verification (Task 5)

Evidence record for the BitCliff freeze-execution plan, Task 5. Covers:

1. The Gutenberg 2b corpus (ruling 2: *The Count of Monte Cristo*, PG #1184).
2. Verification of the 2a corpus (`sgoel9/paul_graham_essays`) against the
   hash pinned in the vendored `multivalue2` generator / bundle provenance.
3. Verification of the multivalue2-bundle's item-construction digest, using
   the REAL Qwen2.5-1.5B-Instruct tokenizer (no model weights loaded).

All hashes below were computed directly by re-running the download/build in
this environment on 2026-08-26 — nothing is copied from another document
without independent recomputation.

---

## 1. Gutenberg 2b corpus — *The Count of Monte Cristo* (PG #1184)

Per binding ruling 2.

| field | value |
|---|---|
| PG file ID | 1184 |
| Source URL (actual, as fetched) | `https://www.gutenberg.org/cache/epub/1184/pg1184.txt` |
| HTTP response | `200 OK` |
| HTTP `content-type` | `text/plain; charset=utf-8` |
| HTTP `content-length` | 2,787,124 bytes (matches downloaded byte count) |
| Retrieval date | 2026-08-26 |
| Raw download size | 2,787,124 bytes |
| Raw download sha256 | `64f8d5cfa51fcecb904abf7312d395d512a71817e7359b91288beb50517c3836` |

### Strip rule (verbatim, as applied)

> Text strictly between the line beginning `*** START OF THE PROJECT
> GUTENBERG EBOOK ...` and the line beginning `*** END OF THE PROJECT
> GUTENBERG EBOOK ...`, **both marker lines excluded**.

Concretely: the raw file was decoded as UTF-8 and split on `\n` (Gutenberg's
plain-text release uses `\r\n` line endings; splitting on `\n` alone leaves a
trailing `\r` on each line, which is preserved — no other normalization was
applied). The two marker lines were located by `str.startswith`:

- Start marker line (0-based line index 27):
  `*** START OF THE PROJECT GUTENBERG EBOOK THE COUNT OF MONTE CRISTO ***`
- End marker line (0-based line index 61329):
  `*** END OF THE PROJECT GUTENBERG EBOOK THE COUNT OF MONTE CRISTO ***`

The kept text is `lines[28:61329]` joined with `\n`, i.e. every line strictly
between the two markers, both markers themselves dropped.

### Stripped output

| field | value |
|---|---|
| Output path | `bitcliff/pipeline/corpora/pg1184-monte-cristo.txt` (gitignored; not committed — see `.gitignore`) |
| Stripped char count | 2,688,565 |
| Stripped byte count (UTF-8) | 2,767,256 |
| Stripped text sha256 | `0a21a13834b5215876bd4019af8fbc436abbfbb61b2826db62223eb990071443` |

`corpora/` was added to `bitcliff/pipeline/.gitignore` — the corpus text
itself is not committed (Project Gutenberg license terms permit
redistribution with the standard PG header/footer retained; since this repo
strips that header/footer for use as filler text, the derived file is kept
local rather than committed). This manifest is the durable, committed record
of exactly what was downloaded and how it was transformed.

---

## 2. 2a corpus verification — `sgoel9/paul_graham_essays`

Per the task brief and `multivalue2-bundle/PROVENANCE.md` §"Data
dependencies" / §"Integrity anchors".

| field | value |
|---|---|
| Dataset | `sgoel9/paul_graham_essays` (Hugging Face `datasets`) |
| Split | `train` |
| Row count | 215 |
| Columns | `id`, `title`, `date`, `text` |
| Join rule | `"\n\n".join(ds["text"])` |
| Joined char count | 2,938,955 |
| Joined byte count (UTF-8) | 2,940,081 |
| **Computed sha256** | `b6135331a3132d08cb84262870ae8f9d9acb6bae4cd7f0278926a64c38f9329e` |
| **Expected sha256** (pinned in vendored generator `CORPUS_SHA256` / bundle docs) | `b6135331a3132d08cb84262870ae8f9d9acb6bae4cd7f0278926a64c38f9329e` |
| **Verdict** | **PASS** — hashes match exactly. |

---

## 3. Bundle digest verification — multivalue2 item construction

Per the task brief and `multivalue2-bundle/PROVENANCE.md` §"Integrity
anchors" / `PAPER_CONFIG.md` §"Drift between the paper version and current
code". This step reproduces the bundle's documented digest recipe
(`items_digest` in the vendored `generate_multivalue2.py`, byte-compatible
with the source project's `data.items_digest`) using:

- The **real** Qwen2.5-1.5B-Instruct tokenizer, loaded locally from
  `bitcliff/pipeline/models/hf/Qwen2.5-1.5B-Instruct` (the HF snapshot cached
  by the pilot conversion — tokenizer/chat-template files only; no model
  weights were loaded).
- The **verified** 2a corpus text from step 2 above (its sha256 matches the
  generator's pinned `CORPUS_SHA256`, so the vendored generator's own
  internal assertion also passes).
- The vendored generator's real `build_items` (not the test's stub
  tokenizer), called exactly as `bitcliff_pipeline.suites.longctx_retrieval`
  does: pre-populate `_STREAM_CACHE[tokenizer.name_or_path]` with the
  corpus's token ids, then call `mv2.build_items(tokenizer, "multivalue2",
  20, 4096, 2024)` (default depths `(0.1, 0.5, 0.9)`, matching
  `PAPER_CONFIG.md`'s "Common configuration").

| field | value |
|---|---|
| Tokenizer | `Qwen/Qwen2.5-1.5B-Instruct` (loaded from local path; `name_or_path` reports the local cache path, used only as the `_STREAM_CACHE` key) |
| Corpus token count | 638,164 |
| Variant | `multivalue2` |
| `n_items` | 20 |
| `target_tokens` | 4096 |
| `seed` | 2024 |
| Items built | 20 |
| **Computed `items_digest`** | `9220589bd8607bd0ff3be5bdcfecd23df07cac82d354d992468b15b60f398972` |
| **Expected digest** (bundle `PROVENANCE.md` §"Integrity anchors") | `9220589bd8607bd0ff3be5bdcfecd23df07cac82d354d992468b15b60f398972` |
| **Verdict** | **PASS** — digest matches exactly. |

This confirms the vendoring into `bitcliff_pipeline.vendor.generate_multivalue2`
did not alter item construction: with the real tokenizer and the verified
corpus, this environment reproduces the exact same first-20-item digest as
the source project and the bundle's own cross-check.

---

## Summary

| check | verdict |
|---|---|
| Gutenberg PG #1184 download + strip | done — raw & stripped sha256 recorded above |
| 2a corpus (`sgoel9/paul_graham_essays`) hash | **PASS** |
| multivalue2 bundle `items_digest` (real tokenizer, n=20, t=4096, seed=2024) | **PASS** |

No BLOCKED conditions were hit. Both required verifications passed on the
first run; no retries or corpus substitutions were needed for the 2a corpus
or the digest check.

**Dependency note:** `transformers` was added as a main dependency
(`uv add transformers`, resolved to `5.16.1`) — it was not previously
installed in this environment. No PyTorch/model weights were installed or
loaded; only `AutoTokenizer` was used.
