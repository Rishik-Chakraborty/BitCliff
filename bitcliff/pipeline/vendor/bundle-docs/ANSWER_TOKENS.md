<!-- Vendored verbatim from multivalue2-bundle/ANSWER_TOKENS.md of the source quantization
     repository (Apache-2.0; LICENSE added at commit 0d1885e, 2026-08-26 -- see
     bitcliff/pipeline/LICENSE_AUDIT.md section 7). The bundle documents the paper draft at
     commit 8071fc44d91b15842c57cbf26a92bdd968b0d522. Extracted 2026-08-26.
     Everything below this comment is byte-identical to the source file. -->
# ANSWER_TOKENS.md — exactly which tokens are the answer

This is precise enough to compute per-token divergence on answer tokens only,
using the fields shipped in the sample JSONL (or produced by the generator).

## Construction

For every retrieval item:

```
answer_text    = ", ".join(match_strings)              # e.g. "4404, 5614"
answer_ids     = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
nll_input_ids  = gen_prompt_ids + answer_ids
nll_target_mask = [False]*len(gen_prompt_ids) + [True]*len(answer_ids)
```

The scored span is a **contiguous suffix**: positions
`len(gen_prompt_ids) … len(gen_prompt_ids)+len(answer_ids)−1` of
`nll_input_ids`. (The mask representation exists in the source project because
its arithmetic items score scattered positions; for retrieval it is always this
suffix.) The JSONL stores `gen_prompt_ids` and `answer_ids` separately;
reconstruct the sequence and mask as above.

## Alignment convention

Token at position `p` is scored against the model's prediction from the hidden
state at position `p−1`:

- logits for the **first answer token** come from the **last prompt token**
  (the final token of the chat template's generation prompt, i.e. the position
  where free generation would start);
- logits for answer token `k` come from the position of answer token `k−1`.

So per-token divergence on answer tokens between two models means: run both on
the identical `nll_input_ids`, take logits at positions
`[len(prompt)−1 … len(prompt)+n_ans−2]`, and compare them against each other
and/or against targets `answer_ids`. Batches in the source project are
**right-padded** for this pass, so padding can never sit before a scored
position; single-sequence evaluation needs no padding at all.

## What the span contains (Qwen2.5 tokenizer)

Qwen2.5 tokenizes digits individually. For `multivalue2` the answer span is
**exactly 10 tokens**, e.g. for `"4404, 5614"`:

| pos (rel) | token id | text |
|---|---|---|
| 0 | 19 | `4` |
| 1 | 19 | `4` |
| 2 | 15 | `0` |
| 3 | 19 | `4` |
| 4 | 11 | `,` |
| 5 | 220 | ` ` (space) |
| 6 | 20 | `5` |
| 7 | 21 | `6` |
| 8 | 16 | `1` |
| 9 | 19 | `4` |

Two caveats that matter for per-token analysis:

1. **The separator tokens `,` and ` ` are scored.** They carry near-zero NLL for
   any intact model and dilute the per-token mean by ~20%. The published item
   NLL is the mean over all 10 tokens, separators included. If you want
   passcode-only divergence, restrict to the 8 digit positions — but state that
   you did, because it departs from the paper's metric.
2. **The first digit of each passcode is where the information is concentrated.**
   Positions 1–3 and 7–9 are heavily conditioned by teacher forcing (given
   `4`,`4`,`0` the fourth digit is still uncertain, but given the needle was
   retrieved at all, later digits tend to follow). Position 0 (first digit of
   value 1, predicted from the prompt alone) and position 6 (first digit of
   value 2, predicted after ", ") are the two retrieval events. This is an
   observation to guide analysis, not part of the published metric.

`n_answer_tokens` is recorded per item; with another tokenizer the count will
differ (a tokenizer that chunks digits could yield as few as ~4 answer tokens),
and the items themselves will differ too — see the tokenizer warning in
PAPER_CONFIG.md.

## The gold "answer" order

`match_strings` order = value-generation order = needle-placement order (depths
`d` and `(d+0.5) mod 1`). In the depth-0.9 third of items this is the reverse
of document order. The teacher-forced target commits to this order; the boolean
grader does not.
