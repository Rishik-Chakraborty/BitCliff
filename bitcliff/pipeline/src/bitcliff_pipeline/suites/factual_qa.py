"""Closed-book factual QA suite (PopQA).

Dataset: `akariasai/PopQA`, `test` split (user ruling 1). The HF mirror's
dataset card carries no license metadata of its own, but the canonical
source release, github.com/AlexTMallen/adaptive-retrieval (Mallen et al.
2023), is MIT-licensed and ships the same data as `data/popQA.tsv`; the HF
mirror is maintained by paper co-author Akari Asai as an access path to
that same MIT-licensed data. See ../../../LICENSE_AUDIT.md section 1 for
the full verification trail.

Schema fields used (verified via `datasets-server` first-rows against the
real PopQA `test` split — see LICENSE_AUDIT.md section 1):
  - question: the `question` field (str).
  - aliases: the `possible_answers` field — a JSON-encoded list string of
    gold aliases, e.g. '["politician", "political leader"]'. Parsed with
    `json.loads`. A record whose `possible_answers` is already a Python
    list (as used by this module's own network-free unit tests) is
    accepted as-is.
  - popularity: the `s_pop` field (int) — the subject entity's Wikipedia
    monthly pageview count. (`o_pop`, the object entity's count, is also
    present in the schema but is not used here.)

Stratified sampling (`items_from_records`):
  1. Sort records by `s_pop` ascending (Python's sort is stable, so ties
     keep their original relative order).
  2. Split into 10 contiguous deciles. Sizes are as equal as possible:
     with N records, `base = N // 10` and `extra = N % 10`; the first
     `extra` deciles get `base + 1` records each, the remaining deciles
     get `base`.
  3. Draw from each decile with `random.Random(seed)`, processing deciles
     in order 0..9. With the default uniform mix (`weights=None`), the
     per-decile draw count uses the same "remainder to the earliest
     buckets" rule as the decile sizing above: `base_n = n_items // 10`,
     `remainder = n_items % 10`; the first `remainder` deciles draw
     `base_n + 1` items, the rest draw `base_n`. With an explicit
     `weights` vector (a registered non-uniform mix, PREREG §7), per-decile
     counts are the **largest-remainder apportionment** of
     `weights[i] * n_items`: floor each raw count, then hand the shortfall
     to the deciles with the largest fractional remainders, ties broken
     toward the lower decile index. `weights[i]` applies to `deciles[i]` as
     built here (ascending `s_pop`). Either way the seeded per-decile draw
     itself is unchanged and draws exactly `n_items` in total (given each
     decile has enough records to satisfy its draw).
  4. Items are id'd `factual_qa-{seed}-{i:04d}` in the order drawn (all of
     decile 0's picks first, then decile 1's, and so on).
"""

import json
import random
import re

from ..items import EvalItem

PROMPT_PREFIX = "Answer with just the answer: "

_WHITESPACE = re.compile(r"\s+")


def _popularity(record) -> int:
    return int(record["s_pop"])


def _parse_aliases(record) -> tuple[str, ...]:
    raw = record["possible_answers"]
    if isinstance(raw, str):
        raw = json.loads(raw)
    return tuple(raw)


def _split_into_deciles(sorted_records: list) -> list[list]:
    n = len(sorted_records)
    base, extra = divmod(n, 10)
    deciles = []
    idx = 0
    for i in range(10):
        size = base + (1 if i < extra else 0)
        deciles.append(sorted_records[idx : idx + size])
        idx += size
    return deciles


def _apportion_counts(weights: tuple[float, ...], n_items: int) -> list[int]:
    """Largest-remainder apportionment of weights[i] * n_items (PREREG §7):
    floor each raw count, then give the shortfall to the largest fractional
    remainders, ties broken toward the lower decile index."""
    raw = [w * n_items for w in weights]
    counts = [int(r) for r in raw]
    shortfall = n_items - sum(counts)
    by_remainder = sorted(range(len(weights)), key=lambda i: (counts[i] - raw[i], i))
    for i in by_remainder[:shortfall]:
        counts[i] += 1
    return counts


def items_from_records(
    records, n_items: int, seed: int, weights: tuple[float, ...] | None = None
) -> list[EvalItem]:
    sorted_records = sorted(records, key=_popularity)
    deciles = _split_into_deciles(sorted_records)

    if weights is None:
        base_n, remainder = divmod(n_items, 10)
        takes = [base_n + (1 if i < remainder else 0) for i in range(10)]
    else:
        if len(weights) != len(deciles):
            raise ValueError(f"weights must have {len(deciles)} entries, got {len(weights)}")
        takes = _apportion_counts(tuple(weights), n_items)

    rng = random.Random(seed)
    picked = []
    for decile, take in zip(deciles, takes):
        picked.extend(rng.sample(decile, take))

    items = []
    for i, r in enumerate(picked):
        items.append(
            EvalItem(
                id=f"factual_qa-{seed}-{i:04d}",
                suite="factual_qa",
                prompt=PROMPT_PREFIX + r["question"],
                expected=_parse_aliases(r),
            )
        )
    return items


def load_popqa_items(n_items: int, seed: int) -> list[EvalItem]:
    from datasets import load_dataset

    ds = load_dataset("akariasai/PopQA", split="test")
    return items_from_records(ds, n_items, seed)


def _normalize(s: str) -> str:
    return _WHITESPACE.sub(" ", s.strip().lower())


def _word_boundary_match(alias_norm: str, haystack_norm: str) -> bool:
    if not alias_norm:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(alias_norm)}(?!\w)", haystack_norm))


def _question_text(prompt: str) -> str:
    if prompt.startswith(PROMPT_PREFIX):
        return prompt[len(PROMPT_PREFIX) :]
    return prompt


def grade(item: EvalItem, text: str) -> str:
    """Grade `text` against item.expected (the alias list).

    Subject-echo guard (characterization round 2): an alias that already
    word-boundary-matches inside the *question* itself is dropped from the
    set of aliases eligible to score a hit. This defends against items
    where the question's subject-entity name coincides with a valid alias
    for the expected answer (e.g. "What is Asti the capital of?" with
    "Asti" itself in the alias list for "Province of Asti") — without the
    guard, an output that merely echoes the subject scores "correct" even
    when its actual claimed answer is wrong. If every alias is disqualified
    this way, the item grades "wrong" regardless of the output text.
    """
    text_norm = _normalize(text)
    question_norm = _normalize(_question_text(item.prompt))
    effective_aliases = [
        alias
        for alias in item.expected
        if not _word_boundary_match(_normalize(alias), question_norm)
    ]
    for alias in effective_aliases:
        if _word_boundary_match(_normalize(alias), text_norm):
            return "correct"
    return "wrong"
