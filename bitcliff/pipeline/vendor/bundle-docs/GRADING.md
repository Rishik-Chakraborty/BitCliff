<!-- Vendored verbatim from multivalue2-bundle/GRADING.md of the source quantization
     repository (Apache-2.0; LICENSE added at commit 0d1885e, 2026-08-26 -- see
     bitcliff/pipeline/LICENSE_AUDIT.md section 7). The bundle documents the paper draft at
     commit 8071fc44d91b15842c57cbf26a92bdd968b0d522. Extracted 2026-08-26.
     Everything below this comment is byte-identical to the source file. -->
# GRADING.md — exact grading rules as used in the paper

Two metrics are recorded. The **boolean exact-match** metric is the
confirmation/accuracy metric; the **teacher-forced NLL** is the search metric
(and the one behind most published dz values). Both come verbatim from
`src/evaluate.py` of the source project.

## 1. Boolean exact match (accuracy, McNemar)

Generation:

- Greedy decoding: `do_sample=False`, `temperature=None`, `top_p=None`, `top_k=None`.
- `max_new_tokens = 32` for retrieval.
- Batched with **left** padding (generation continues from the right edge).
- The completion (everything after the prompt) is decoded with
  `skip_special_tokens=True`.

Scoring — this is the entire rule:

```python
hit = all(s in text for s in item.match_strings)
```

i.e. an item is correct iff **every** gold passcode appears as a plain Python
substring of the decoded completion. For `multivalue2`, both 4-digit strings
must appear. "No fuzzy matching, no partial credit, no grader model: a match is
a match."

### Edge cases (all consequences of the rule above; none are special-cased)

- **Order is not checked.** The prompt says "List both, separated by commas",
  but `4404, 5614` and `5614, 4404` both score correct.
- **Separators are not checked.** Commas, newlines, prose ("The passcodes are
  4404 and 5614") all score correct as long as both substrings occur.
- **Substring match is not boundary-anchored.** A gold `4404` inside a longer
  digit run (e.g. `44045614` or `14404`) counts as a match. Passcodes are drawn
  from 1000–9999, so a gold can occur by coincidence inside an unrelated number
  the model emits. No published analysis quantified this; treat it as a known
  (small) laxity of the grader.
- **Distinctness is guaranteed within an item** (the two values differ), so a
  model that retrieves one value twice cannot score correct by duplication.
- **Truncation:** with 32 new tokens the answer fits many times over (~10 answer
  tokens); a model that pads its answer with long preamble could in principle be
  truncated before emitting both codes, and that scores incorrect.
- **Case/whitespace:** irrelevant — golds are digit strings.
- **When generation is not run** (`with_generation: false`, as in the rung-2
  sweep configs), `exact_match` is `None`, not `False` — absent, never defaulted.

### Aggregation

- Accuracy = mean of hits.
- Paired comparison between baseline and intervened model: **McNemar's exact
  test** on discordant pairs (two-sided binomial at q=0.5 over the b+c items
  that flipped; p=1.0 when nothing flips). Identical item sets on both sides
  are asserted, never assumed.

## 2. Teacher-forced NLL (the search metric behind the dz numbers)

- Sequence: `nll_input_ids = gen_prompt_ids + answer_ids` where
  `answer_ids = tokenizer(", ".join(match_strings), add_special_tokens=False)`.
- One forward pass, **right**-padded (padding sits after everything scored).
- Item score = **mean cross-entropy over the answer-token positions only**
  (see ANSWER_TOKENS.md for the exact positions). Token at position p is
  predicted from the hidden state at p−1; logits are computed in float32 at the
  scored positions only.
- NaN NLL is an assertion failure, never skipped.

### Aggregation

- Per-item paired delta: `Δᵢ = NLLᵢ(intervened) − NLLᵢ(baseline)` on identical
  items (asserted).
- p-value: **paired sign-flip permutation test** (10,000 sign patterns) on
  |mean Δ|.
- CI: bootstrap on the mean of Δ.
- Effect size: **Cohen's dz = mean(Δ) / sd(Δ, ddof=1)** — this is the "dz" in
  every published number. Zero-variance deltas leave dz undefined (None) and
  the gate raises rather than defaulting.
- The project's adopted significance gate ("B1", adopted 2026-08-04):
  permutation p < 0.05 **and** |dz| ≥ 0.2, reported alongside (never instead
  of) the older noise-band criterion `|mean Δ| > band`, where the band is the
  95% CI half-width of the baseline mean NLL from the same run's noise floor.

Note the teacher-forced answer string fixes an order (`v1, v2` in generation
order, which is needle-placement order — reversed relative to document order in
the depth-0.9 third of items). The NLL metric therefore *is* order-sensitive
even though the boolean metric is not.
