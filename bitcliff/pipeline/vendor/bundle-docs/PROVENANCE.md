<!-- Vendored verbatim from multivalue2-bundle/PROVENANCE.md of the source quantization
     repository (Apache-2.0; LICENSE added at commit 0d1885e, 2026-08-26 -- see
     bitcliff/pipeline/LICENSE_AUDIT.md section 7). The bundle documents the paper draft at
     commit 8071fc44d91b15842c57cbf26a92bdd968b0d522. Extracted 2026-08-26.
     Everything below this comment is byte-identical to the source file. -->
# PROVENANCE.md — authorship and license status

Extraction date: 2026-08-26. Source: `capability-targeted-quantization` repo,
commit `8071fc44d91b15842c57cbf26a92bdd968b0d522`, remote
`github.com:Rishik-Chakraborty/quantization`.

## The task itself

- **Author:** Rishik Chakraborty (the source repo's sole author). The task
  design — the variant ladder, the NATO-key/passcode needle format, conjunctive
  multivalue scoring, token-space assembly, and the calibration rule that
  selected `multivalue2` — is original to that project. `generate_multivalue2.py`
  in this bundle is that project's `src/data.py` retrieval path, vendored with
  logic unchanged.
- **License: ⚠ UNRESOLVED.** The source repo contains **no LICENSE file** and
  `pyproject.toml` declares none, so by default all rights are reserved by the
  author. Using this bundle inside the author's own BitCliff project is
  unproblematic; **redistributing it or publishing it as part of an open
  artifact requires the author to add a license first.** Flagged rather than
  assumed.
- **Intellectual lineage** (bundle-author's observation, not stated in the
  source repo): the general paradigm — needles hidden in Paul Graham essays,
  scored by exact match — follows the widely used "Needle In A Haystack"
  evaluation popularized by Greg Kamradt (2023), which also used PG essays as
  filler. The needle templates, key scheme, variant ladder, and scoring here
  are independent implementations; no NIAH code or data is used. A paper using
  this bundle should still cite the paradigm.

## Data dependencies

### Corpus (filler text): `sgoel9/paul_graham_essays` (Hugging Face)

- Split `train`, `text` column joined with `"\n\n"`; sha256 of the joined text
  pinned in the generator:
  `b6135331a3132d08cb84262870ae8f9d9acb6bae4cd7f0278926a64c38f9329e`.
- Dataset card metadata (verified directly against the HF API on 2026-08-26,
  not from a summary): license tag **MIT**, DOI `10.57967/hf/2212`, last
  modified 2024-04-20, uploader `sgoel9`.
- **⚠ License caveat:** the MIT tag is the *uploader's* declaration and at most
  covers the dataset packaging. The underlying essays are Paul Graham's
  copyrighted writing from paulgraham.com; neither the dataset card nor
  anything in the source repo documents permission from the author to
  redistribute them. This is the norm across NIAH-style evals (the same essays
  are used by many public benchmarks), but it is an unresolved status, not a
  cleared one. Consequences for the bundle: the generator **downloads** the
  corpus at build time (or takes a local path) rather than shipping essay text;
  the 20 sample items DO embed ~4k tokens of essay text each in their
  `prompt_text`/token ids, so treat `samples/` as carrying the same caveat if
  the bundle is ever distributed publicly.

### Tokenizer / chat template: `Qwen/Qwen2.5-1.5B-Instruct`

- **Apache-2.0** (verified against the HF API on 2026-08-26). Only the
  tokenizer and chat template are needed to build items; no model weights ship
  with or are required by this bundle.
- The Qwen chat template injects its default system prompt ("You are Qwen,
  created by Alibaba Cloud...") — visible in the samples; that text is part of
  Qwen's Apache-2.0-licensed distribution.

### Not dependencies

- GSM8K is used by the source project's *arithmetic* arm only; nothing in this
  bundle touches it.
- The passcodes and keys are synthetic (RNG digits; NATO alphabet words —
  public domain).

## Integrity anchors

- Corpus sha256: pinned above; the generator refuses a mismatched corpus.
- Bundle-vs-source item digest for (multivalue2, t=4096, seed=2024, first 20):
  `9220589bd8607bd0ff3be5bdcfecd23df07cac82d354d992468b15b60f398972`
  (sha256 over item ids + token ids + masks + golds, byte-compatible with the
  source project's `data.items_digest`). Re-run the generator and compare this
  printed digest to confirm an environment reproduces the paper's items.
