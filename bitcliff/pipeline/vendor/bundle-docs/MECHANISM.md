<!-- Vendored verbatim from multivalue2-bundle/MECHANISM.md of the source quantization
     repository (Apache-2.0; LICENSE added at commit 0d1885e, 2026-08-26 -- see
     bitcliff/pipeline/LICENSE_AUDIT.md section 7). The bundle documents the paper draft at
     commit 8071fc44d91b15842c57cbf26a92bdd968b0d522. Extracted 2026-08-26.
     Everything below this comment is byte-identical to the source file. -->
# MECHANISM.md — what capability this task actually measures

## Verdict

**`multivalue2` is pure in-context lookup — copying from the prompt. It contains
no parametric recall component at all.** This is decidable from the task
construction, not from probing:

1. **The gold values cannot be in the weights.** Each passcode is drawn fresh
   from `rng.randrange(1000, 10000)` per item, seeded by
   `(seed, variant, target_tokens, item_index)`. The value `4404` for key
   `bravo` in item 0 exists nowhere except inside that item's prompt. No
   training corpus contains the association; a model with the prompt truncated
   before the needle has a 1-in-9000 chance per code.
2. **The keys are chosen to defeat semantic shortcuts.** Keys are NATO-alphabet
   words (`bravo`, `sierra`, …) with no topical relationship to the essay
   filler, precisely so the model cannot locate the needle by subject-matter
   association — the only route to the answer is matching the literal key
   string and copying the adjacent value.
3. **The filler is constant across the ladder.** Difficulty is manipulated by
   needle structure and conjunctive scoring, never by making the *content*
   harder to know. Nothing in the task rewards knowing anything.

Mechanistically, success requires: attending from the question back to the two
needle sentences across ~4k tokens (find both instances of the key), binding
each key occurrence to its adjacent value, and emitting the two values —
induction/copying circuitry plus long-range positional machinery, executed at
inference time. The unquantized 1.5B model's answer-span NLL of ~0.2 (n=96
baseline) reflects this: the answer is nearly determined by the context, and
what quantization degrades is the *transport*, not stored knowledge.

## What this implies for comparing against memorized-fact recall

If BitCliff wants to compare quantization damage on this task against damage on
parametric recall (facts stored in weights), the comparison is between **two
different mechanisms**, not two difficulty levels of one thing:

- **Different substrate.** In-context copying is generally attributed to
  attention (induction-head-like) circuits and, at 4k, to positional/long-range
  fidelity; parametric fact recall is generally attributed to MLP/feed-forward
  associative storage. Quantizing the same bit-width can plausibly hit these
  very differently — indeed the source project's own component map at 3 bits
  puts the largest retrieval effects in attention blocks (`attn_block__L0`,
  `attn_block__L12`), which is consistent with, though not proof of, the
  attention-transport reading.
- **Different failure semantics.** Here, a failure means the model *had* the
  answer in its context window and could not move it to the output. A
  parametric-recall failure means the stored association degraded. A headline
  like "quantization hurts retrieval more than facts" is really "quantization
  hurts in-context copying more than weight-stored associations" — say it that
  way.
- **Different baseline floors.** Teacher-forced NLL on a copy task is low
  (~0.2 here) because the context nearly determines the answer; NLL on a
  parametric fact reflects genuine uncertainty. Equal ΔNLL therefore means
  different relative degradation; the source project standardizes with dz
  (mean Δ / sd Δ) partly for this reason. Compare on dz or on accuracy drops,
  not raw ΔNLL, and say which.
- **The task cannot detect knowledge loss even in principle.** A quantized
  model that forgot every fact it ever knew but kept its copying circuitry
  would score perfectly here. Do not use `multivalue2` as evidence about
  knowledge retention.

One honest caveat to the "pure" claim: executing the task still uses generic
learned competence (instruction following, digit emission, the chat format).
Those are parametric in the trivial sense that all behavior is. The claim that
matters — the mapping from question to gold answer exists only in the prompt —
holds absolutely by construction.
