#!/usr/bin/env python3
"""Standalone, seedable generator for the `multivalue2` retrieval task.

Extracted verbatim (logic unchanged) from `src/data.py` of the
capability-targeted-quantization repo at commit 8071fc44d91b15842c57cbf26a92bdd968b0d522.
This file has no dependency on that repo. It reproduces the paper's items
EXACTLY: for the same (tokenizer, corpus, variant, target_tokens, seed, depths),
item i here is bit-identical to item i in the source project, for any n_items
(per-item RNG derivation means item i does not depend on how many items you build).

Dependencies:
    transformers   (tokenizer only; no model is loaded)
    datasets       (only if the corpus is not already cached locally)

Usage:
    python generate_multivalue2.py \
        --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
        --variant multivalue2 --n-items 20 --target-tokens 4096 --seed 2024 \
        --corpus /path/to/corpus.txt \
        --out samples.jsonl

The corpus is the Paul Graham essay collection (HF: sgoel9/paul_graham_essays,
"train" split, `text` column joined with "\\n\\n"). Its sha256 is pinned below;
the script refuses to run on a corpus that does not match, because a different
corpus silently changes every document.

Output: one JSON object per line. See ANSWER_TOKENS.md for how to reconstruct
the teacher-forcing sequence and the answer-token mask from the stored fields.

--- BitCliff vendoring provenance --------------------------------------------

Vendored verbatim (no logic changed; this note is the only addition) into
`bitcliff_pipeline.vendor.generate_multivalue2` from
`multivalue2-bundle/generate_multivalue2.py`
(commit d69a363edc3b0ac4368c9d11941220a1e1f2b937 of the `quantization` repo,
sha256 0f35a29f78d22f62c9aafc5865aa21904d0cd34a3fc1c67ef90cbff8d2ef92f4 of the
original file) on 2026-08-26, for the BitCliff freeze-execution plan, Task 1
(the longctx_retrieval suite). See
`.superpowers/sdd/plan-freeze/task-1-brief.md` and `task-1-report.md` in the
BitCliff repo for the adapter (`bitcliff_pipeline.suites.longctx_retrieval`)
that wraps this generator with a stub-tokenizer-friendly, cache-safe API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (verbatim from src/data.py)
# ---------------------------------------------------------------------------

CORPUS_DATASET = "sgoel9/paul_graham_essays"

# sha256 of the joined corpus text as used in every published run.
CORPUS_SHA256 = "b6135331a3132d08cb84262870ae8f9d9acb6bae4cd7f0278926a64c38f9329e"

# Retrieval difficulty is a two-part dial: which FORM the task takes, and how
# many needles are involved.
#   single        one needle, one query
#   multikeyN     N needles, 1 is the target, N-1 are same-format distractors
#   multiqueryN   N keys, all N must be answered  (conjunctive scoring)
#   multivalueN   N values under ONE key, all N required (conjunctive scoring)
RETRIEVAL_FORMS = ("single", "multikey", "multiquery", "multivalue")
DEFAULT_N = 4

# Roughly ordered easiest to hardest (the source project's calibration ladder).
DIFFICULTY_ORDER = (
    "single",
    "multikey4",
    "multikey8",
    "multikey12",
    "multivalue2",
    "multiquery2",
    "multiquery3",
    "multivalue3",
    "multiquery4",
    "multivalue4",
)

DEFAULT_DEPTHS = (0.1, 0.5, 0.9)

# Distinct, unambiguous keys. Deliberately mundane NATO-alphabet words: a key
# that appears naturally in the corpus would let the model find the needle by
# topic rather than by retrieval.
KEYS = (
    "alpha",
    "bravo",
    "delta",
    "sierra",
    "tango",
    "victor",
    "yankee",
    "zulu",
    "kilo",
    "romeo",
    "papa",
    "foxtrot",
)

NEEDLE_TEMPLATE = "The secret passcode for {key} is {value}."
SINGLE_NEEDLE_TEMPLATE = "The secret passcode is {value}."

DOC_PLACEHOLDER = "<<<DOCUMENT_GOES_HERE>>>"

_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 8: "eight", 12: "twelve"}


# ---------------------------------------------------------------------------
# Variant parsing (verbatim)
# ---------------------------------------------------------------------------


def parse_variant(variant: str) -> tuple[str, int]:
    """`multikey8` -> ('multikey', 8). Bare forms default to N=4."""
    for form in RETRIEVAL_FORMS:
        if variant == form:
            return form, 1 if form == "single" else DEFAULT_N
        if variant.startswith(form) and variant[len(form) :].isdigit():
            n = int(variant[len(form) :])
            assert form != "single", "the `single` form takes no count"
            assert 2 <= n <= len(KEYS), (
                f"variant {variant!r} needs {n} distinct keys but only "
                f"{len(KEYS)} are defined"
            )
            return form, n
    raise AssertionError(
        f"unknown retrieval variant {variant!r}; expected one of {DIFFICULTY_ORDER}"
    )


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def load_corpus(corpus_path: Path | None) -> str:
    """Return the corpus text, downloading it if no local path is given.

    Whatever the source, the text must hash to CORPUS_SHA256 — a different
    corpus silently changes every document.
    """
    if corpus_path is not None:
        text = Path(corpus_path).read_text(encoding="utf-8")
    else:
        from datasets import load_dataset

        ds = load_dataset(CORPUS_DATASET, split="train")
        assert "text" in ds.column_names, (
            f"{CORPUS_DATASET} has columns {ds.column_names}, expected 'text'"
        )
        text = "\n\n".join(ds["text"])

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert digest == CORPUS_SHA256, (
        f"corpus sha256 is {digest}, expected {CORPUS_SHA256}. A different "
        f"corpus produces different documents; items would not be comparable "
        f"to the published runs. If the HF dataset has changed upstream, "
        f"obtain the original corpus.txt from the source project."
    )
    return text


# ---------------------------------------------------------------------------
# Token-space prompt assembly (verbatim)
# ---------------------------------------------------------------------------


def _split_chat_template(tokenizer, question: str) -> tuple[list[int], list[int]]:
    """Tokenize the chat wrapper either side of where the document goes."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{DOC_PLACEHOLDER}\n\n{question}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert rendered.count(DOC_PLACEHOLDER) == 1, (
        f"chat template did not preserve the document placeholder exactly once "
        f"(found {rendered.count(DOC_PLACEHOLDER)})"
    )
    head, tail = rendered.split(DOC_PLACEHOLDER)
    head_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
    tail_ids = tokenizer(tail, add_special_tokens=False)["input_ids"]
    assert head_ids and tail_ids
    return head_ids, tail_ids


def _build_document_ids(
    tokenizer,
    stream: list[int],
    target_tokens: int,
    needles: list[tuple[float, str]],
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    """Document of exactly `target_tokens` tokens with needles at the requested
    depths. Returns (document_ids, actual_insertion_positions)."""
    needle_id_lists = [
        tokenizer(f"\n{text}\n", add_special_tokens=False)["input_ids"]
        for _, text in needles
    ]
    needle_total = sum(len(n) for n in needle_id_lists)
    filler_len = target_tokens - needle_total
    assert filler_len > 0, (
        f"needles alone are {needle_total} tokens, which does not fit in a "
        f"{target_tokens}-token document"
    )
    assert filler_len + 1 < len(stream), (
        f"need {filler_len} filler tokens but the corpus has only {len(stream)}"
    )

    # A random window per item, so different items see different filler.
    start = rng.randrange(0, len(stream) - filler_len)
    filler = list(stream[start : start + filler_len])
    assert len(filler) == filler_len

    # Insert from the deepest position backwards so earlier insertions do not
    # shift the indices of later ones.
    order = sorted(range(len(needles)), key=lambda i: needles[i][0], reverse=True)
    positions = [0] * len(needles)
    for i in order:
        depth = needles[i][0]
        assert 0.0 <= depth <= 1.0, f"depth {depth} outside [0, 1]"
        pos = int(round(depth * filler_len))
        pos = max(0, min(pos, len(filler)))
        filler[pos:pos] = needle_id_lists[i]
        positions[i] = pos

    assert len(filler) == target_tokens
    return filler, positions


# ---------------------------------------------------------------------------
# Item content (verbatim)
# ---------------------------------------------------------------------------


def _distinct_values(rng: random.Random, n: int) -> list[str]:
    """n distinct 4-digit passcodes, in an order that depends only on `rng`.

    The accumulator is a LIST and the set is used only for membership. This is
    load-bearing: an earlier version accumulated into a set and sorted by
    rng.random(), which draws sort keys in set-iteration (string-hash) order —
    randomised per process by CPython. Same seed, different passcodes per
    process. Do not "simplify" this function.
    """
    values: list[str] = []
    seen: set[str] = set()
    while len(values) < n:
        candidate = f"{rng.randrange(1000, 10000)}"
        if candidate not in seen:
            seen.add(candidate)
            values.append(candidate)
    rng.shuffle(values)
    return values


def _retrieval_content(variant: str, rng: random.Random) -> tuple[list, str, list[str]]:
    """(needle_texts, question, match_strings) for a variant."""
    form, n = parse_variant(variant)

    if form == "single":
        value = _distinct_values(rng, 1)[0]
        return (
            [SINGLE_NEEDLE_TEMPLATE.format(value=value)],
            "What is the secret passcode? Reply with the number only.",
            [value],
        )

    if form == "multivalue":
        # One key, n values. Difficulty comes from conjunctive scoring: all n
        # must be recovered.
        key = rng.choice(KEYS)
        values = _distinct_values(rng, n)
        word = _NUMBER_WORDS.get(n, str(n))
        listing = "List both" if n == 2 else f"List all {word}"
        return (
            [NEEDLE_TEMPLATE.format(key=key, value=v) for v in values],
            f"There are {word} secret passcodes for {key}. {listing}, "
            f"separated by commas.",
            values,
        )

    keys = rng.sample(KEYS, n)
    values = _distinct_values(rng, n)
    needles = [NEEDLE_TEMPLATE.format(key=k, value=v) for k, v in zip(keys, values)]

    if form == "multikey":
        target = rng.randrange(n)
        return (
            needles,
            f"What is the secret passcode for {keys[target]}? "
            f"Reply with the number only.",
            [values[target]],
        )

    assert form == "multiquery", f"unhandled retrieval form {form!r}"
    word = _NUMBER_WORDS.get(n, str(n))
    return (
        needles,
        "What are the secret passcodes for "
        + ", ".join(keys)
        + f"? Reply with the {word} numbers in that order, separated by commas.",
        values,
    )


# ---------------------------------------------------------------------------
# Item construction
# ---------------------------------------------------------------------------


def build_items(
    tokenizer,
    variant: str,
    n_items: int,
    target_tokens: int,
    seed: int,
    depths: tuple[float, ...] = DEFAULT_DEPTHS,
) -> list[dict]:
    """Deterministic from (tokenizer, corpus, variant, n_items, target_tokens,
    seed, depths). Returns a list of dicts (see JSONL schema in the docstring
    of `main`)."""
    parse_variant(variant)
    assert n_items > 0 and target_tokens > 0
    assert depths, "at least one depth is required"

    stream = _token_stream(tokenizer)
    items: list[dict] = []

    for i in range(n_items):
        # Per-item RNG derived from the run seed, so item i is identical
        # whether you build 20 items or 200. NOTE: seeded with a STRING —
        # random.Random(f"...") — exactly as in the source project.
        rng = random.Random(f"{seed}|{variant}|{target_tokens}|{i}")
        depth = depths[i % len(depths)]

        needle_texts, question, match_strings = _retrieval_content(variant, rng)

        if len(needle_texts) == 1:
            placed = [(depth, needle_texts[0])]
        else:
            # Spread multiple needles evenly, offset by this item's depth so
            # the whole pattern moves through the document across items.
            placed = [
                (((depth + j / len(needle_texts)) % 1.0), t)
                for j, t in enumerate(needle_texts)
            ]

        doc_ids, positions = _build_document_ids(
            tokenizer, stream, target_tokens, placed, rng
        )
        head_ids, tail_ids = _split_chat_template(tokenizer, question)
        prompt_ids = head_ids + doc_ids + tail_ids

        answer_text = ", ".join(match_strings)
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        assert answer_ids, "answer tokenized to nothing"

        items.append(
            {
                "item_id": f"retrieval__{variant}__t{target_tokens}__s{seed}__{i:04d}",
                "capability": "retrieval",
                "variant": variant,
                "seed": seed,
                "index": i,
                "depth": depth,
                "target_tokens": target_tokens,
                "question": question,
                "needle_texts": [t for _, t in placed],
                "needle_depths": [d for d, _ in placed],
                "needle_token_positions": positions,
                "match_strings": match_strings,
                "answer_text": answer_text,
                "prompt_text": tokenizer.decode(prompt_ids),
                "gen_prompt_ids": prompt_ids,
                "answer_ids": answer_ids,
                "prompt_tokens": len(prompt_ids),
                "document_tokens": len(doc_ids),
                "n_answer_tokens": len(answer_ids),
                "tokenizer": tokenizer.name_or_path,
                "corpus_sha256": CORPUS_SHA256,
            }
        )

    # Distinctness assertions, as in the source project.
    ids = [it["item_id"] for it in items]
    assert len(set(ids)) == len(ids), "duplicate item_id"
    prompts = {tuple(it["gen_prompt_ids"]) for it in items}
    assert len(prompts) == len(items), "two items share an identical prompt"
    return items


_STREAM_CACHE: dict[str, list[int]] = {}
_CORPUS_PATH: Path | None = None


def _token_stream(tokenizer) -> list[int]:
    key = tokenizer.name_or_path
    if key not in _STREAM_CACHE:
        text = load_corpus(_CORPUS_PATH)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        assert len(ids) > 200_000, "corpus tokenized to too few tokens"
        _STREAM_CACHE[key] = ids
    return _STREAM_CACHE[key]


def items_digest(items: list[dict]) -> str:
    """Byte-compatible with the source project's `data.items_digest`: a hash of
    what the items ARE (ids + token content), so a bundle build can be checked
    against a source-project build rather than assumed equal."""
    h = hashlib.sha256()
    for it in items:
        nll_input_ids = it["gen_prompt_ids"] + it["answer_ids"]
        nll_target_mask = [False] * len(it["gen_prompt_ids"]) + [True] * len(
            it["answer_ids"]
        )
        h.update(it["item_id"].encode())
        h.update(b"\x00")
        h.update(json.dumps(nll_input_ids).encode())
        h.update(json.dumps(nll_target_mask).encode())
        h.update(json.dumps(it["gen_prompt_ids"]).encode())
        h.update(json.dumps(sorted(it["match_strings"])).encode())
        h.update(b"\x01")
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--variant", default="multivalue2", help=f"one of {DIFFICULTY_ORDER}")
    ap.add_argument("--n-items", type=int, default=20)
    ap.add_argument("--target-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument(
        "--depths",
        default="0.1,0.5,0.9",
        help="comma-separated depth cycle (default matches the paper)",
    )
    ap.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="path to a local corpus.txt; omitted -> download from HF and verify hash",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--no-ids",
        action="store_true",
        help="omit token-id fields (smaller files; text only)",
    )
    args = ap.parse_args()

    global _CORPUS_PATH
    _CORPUS_PATH = args.corpus

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    depths = tuple(float(d) for d in args.depths.split(","))

    items = build_items(
        tokenizer, args.variant, args.n_items, args.target_tokens, args.seed, depths
    )
    digest = items_digest(items)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            if args.no_ids:
                it = {
                    k: v
                    for k, v in it.items()
                    if k not in ("gen_prompt_ids", "answer_ids")
                }
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"wrote {len(items)} items to {args.out}")
    print(f"items_digest: {digest}")


if __name__ == "__main__":
    main()
