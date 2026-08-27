<!-- Vendored verbatim from multivalue2-bundle/TASK.md of the source quantization
     repository (Apache-2.0; LICENSE added at commit 0d1885e, 2026-08-26 -- see
     bitcliff/pipeline/LICENSE_AUDIT.md section 7). The bundle documents the paper draft at
     commit 8071fc44d91b15842c57cbf26a92bdd968b0d522. Extracted 2026-08-26.
     Everything below this comment is byte-identical to the source file. -->
# TASK.md — the `multivalue2` retrieval task

Extracted from the capability-targeted-quantization project (commit
`8071fc44d91b15842c57cbf26a92bdd968b0d522`, 2026-08-26). Generator:
`generate_multivalue2.py` (logic vendored verbatim from the project's
`src/data.py`; bit-identical output verified — see PAPER_CONFIG.md §Verification).

## What the task is

Needle-in-a-haystack retrieval with **conjunctive scoring**. Each item hides
**two distinct 4-digit passcodes under one shared key** inside a document of
exactly **4096 tokens** of filler drawn from a fixed public corpus (Paul Graham
essays). The question asks for both passcodes; an item is correct only if **both**
appear in the model's output.

## Prompt format (real item, `retrieval__multivalue2__t4096__s2024__0000`)

The prompt is a single user turn under the model's chat template
(Qwen2.5-Instruct shown; the template is whatever tokenizer you build with):

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
 accessibility. The application described here is one that we wrote in order to test a new Lisp dialect called [Arc](arc.html) ...

    [ ~360 tokens of essay filler ... ]

The secret passcode for bravo is 4404.

    [ ~2000 tokens of essay filler ... ]

The secret passcode for bravo is 5614.

    [ ~1650 tokens of essay filler ... ]

There are two secret passcodes for bravo. List both, separated by commas.<|im_end|>
<|im_start|>assistant
```

Gold answer: `4404, 5614` (as text) / `match_strings = ["4404", "5614"]`.

Notes on the format:

- The filler is a **random contiguous window** of the tokenized corpus, so the
  document typically starts and ends mid-sentence. That is by design — the
  window is part of per-item variance.
- Each needle is spliced in as the token ids of `"\n{needle}\n"`.
- Needle sentence template: `The secret passcode for {key} is {value}.`
- Question template (N=2): `There are two secret passcodes for {key}. List both,
  separated by commas.` (For N≥3 the phrasing is `List all three/four, ...` —
  "List all two" was deliberately avoided as it reads as an error and could tax
  comprehension rather than retrieval.)

## How items are constructed

Everything is assembled in **token space** — nothing is decoded and re-encoded,
so "4096 tokens" is exact and a needle at depth 0.5 sits at token `0.5 × filler_len`.

Per item `i` (deterministic given `(tokenizer, corpus, variant, target_tokens, seed)`):

1. A per-item RNG is derived as `random.Random(f"{seed}|{variant}|{target_tokens}|{i}")`.
   Item `i` is therefore **identical whether you build 20 items or 200** — the
   20 samples shipped here are literally the first 20 of every published run at
   seed 2024.
2. `key` = one of 12 NATO-alphabet words (`alpha bravo delta sierra tango victor
   yankee zulu kilo romeo papa foxtrot`), chosen so the key has no topical
   association with the essay filler (a topical key would let the model find the
   needle by subject matter rather than retrieval).
3. Two **distinct** passcodes are drawn uniformly from 1000–9999 (as strings),
   then shuffled. Distinctness is load-bearing: with a shared value, retrieving
   the wrong needle would still score correct.
4. The item's base depth cycles through `(0.1, 0.5, 0.9)` by item index. The two
   needles are placed at depths `d` and `(d + 0.5) mod 1.0` — so the pair sits at
   (0.1, 0.6), (0.5, 1.0→end), or (0.9, 0.4) across the cycle. Note the third
   case places the **second** listed value *earlier* in the document than the first.
5. A filler window of `target_tokens − needle_tokens` tokens is taken from a
   random start position in the tokenized corpus; needles are inserted
   deepest-first so insertions don't shift each other's indices. The finished
   document is exactly `target_tokens` tokens.
6. The chat template is rendered once around a placeholder and split on it, so
   the template's special tokens are preserved exactly; the document ids are
   spliced between head and tail. Final prompt = `head_ids + doc_ids + tail_ids`
   (4143 tokens total for Qwen2.5 at target 4096).

## Dataset size / parameterization

There is **no fixed dataset** — the generator is the dataset. It is seedable and
unbounded (any `n_items`). Published runs used seed **2024** with n = 24, 48, and
96 (nested by construction: the n=24 items are the first 24 of the n=96 items),
and seed **2025** for a deliberately disjoint re-measurement set. Shipped here:
`samples/multivalue2_seed2024_first20.jsonl` — the first 20 items of the paper's
seed-2024 stream, with full token ids and golds.

## Difficulty knobs

The source project treats difficulty as a two-part dial plus context length:

1. **Form** — what makes the task hard:
   - `single`: one needle, one query (easiest).
   - `multikeyN`: N needles under N different keys, only one is queried —
     difficulty from **distractor pressure**, not conjunctive scoring.
   - `multiqueryN`: N needles under N keys, all N queried **in order** —
     conjunctive, plus order is requested (though grading ignores order; see
     GRADING.md).
   - `multivalueN`: N values under **one** key, all N required — conjunctive
     scoring with maximal key collision (every needle looks identical except
     the value).
2. **N** — number of needles/values (2–12; keys cap it at 12).
   Calibrated ladder, easiest → hardest:
   `single, multikey4, multikey8, multikey12, multivalue2, multiquery2,
   multiquery3, multivalue3, multiquery4, multivalue4`.
3. **`target_tokens`** — context length. This is a real difficulty dial, not
   incidental: the project observed NLL rising 0.1137 → 0.4274 on a multikey
   variant going 4k → 16k while accuracy held. Capability identity in the source
   project includes it (`retrieval__multivalue2__t4096`).
4. **Depth** — needle position; cycled (0.1, 0.5, 0.9) rather than swept, so it
   contributes item variance, not a difficulty setting.

`multivalue2` was selected by a pre-registered calibration rule: the hardest
variant whose unquantized baseline accuracy lies in [0.85, 0.98] (measured
0.917), with a McNemar-significant drop under uniform 4-bit quantization and an
NLL rise clearing the measured noise floor. Variants easier on the ladder sat at
ceiling (1.00 — no headroom for any drop to register); `multiquery3` at 0.833
was below the floor and failed all three criteria.
