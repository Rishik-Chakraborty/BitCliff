#!/usr/bin/env python3
"""Tiered comparator for the NLL reproduction gate (RUN_0B_PRIME.md §4,
match rule, user ruling 2026-10-03).

Usage: nll_gate_compare.py PRODUCED COMMITTED   (two `nll/<label>.jsonl`)

Records are compared in recorded (file) order. Excluded as metadata: the
`machine` string (the run id is not a record field). Compared: `item_id`,
`suite`, `quant_label`, `model_sha256`, `n_answer_tokens`, `per_token_nll`,
`span_mean_nll`, `digits_only_mean_nll`.

- tier 1: same count and order, every compared field equal (per-token
  values exactly).
- tier 2: same item ids in the same order, same `n_answer_tokens` and
  `per_token_nll` lengths, same `suite`/`quant_label`/`model_sha256`, and
  max |per-token diff| <= 1e-3. Reports that max and the max diffs of
  `span_mean_nll` and `digits_only_mean_nll`.
- anything else: MISMATCH, naming the first offending item/field.

Exit 0 for tier 1 or 2, 1 for mismatch.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from bitcliff_pipeline import nll_scorer

TOLERANCE = 1e-3
EXACT_FIELDS = ("suite", "quant_label", "model_sha256", "n_answer_tokens")


@dataclass(frozen=True)
class GateResult:
    outcome: str  # "tier1" | "tier2" | "mismatch"
    n_records: int
    max_per_token_diff: float | None = None
    max_span_mean_diff: float | None = None
    max_digits_only_diff: float | None = None
    offending_item: str | None = None
    offending_field: str | None = None
    detail: str | None = None

    @property
    def passed(self) -> bool:
        return self.outcome in ("tier1", "tier2")


def _mismatch(n, item, field, detail) -> GateResult:
    return GateResult("mismatch", n, offending_item=item, offending_field=field, detail=detail)


def compare_records(produced, committed) -> GateResult:
    n = len(committed)
    if len(produced) != len(committed):
        return _mismatch(
            n, None, "record_count", f"produced {len(produced)} records, committed {len(committed)}"
        )
    exact = True
    max_pt = max_span = max_dig = 0.0
    for k, (p, c) in enumerate(zip(produced, committed)):
        if p.item_id != c.item_id:
            return _mismatch(
                n, c.item_id, "item_id",
                f"position {k}: produced {p.item_id!r}, committed {c.item_id!r}",
            )
        for f in EXACT_FIELDS:
            if getattr(p, f) != getattr(c, f):
                return _mismatch(
                    n, c.item_id, f, f"produced {getattr(p, f)!r}, committed {getattr(c, f)!r}"
                )
        if len(p.per_token_nll) != len(c.per_token_nll):
            return _mismatch(
                n, c.item_id, "per_token_nll",
                f"length produced {len(p.per_token_nll)}, committed {len(c.per_token_nll)}",
            )
        for j, (a, b) in enumerate(zip(p.per_token_nll, c.per_token_nll)):
            d = abs(a - b)
            if not (math.isfinite(a) and math.isfinite(b) and d <= TOLERANCE):
                return _mismatch(
                    n, c.item_id, "per_token_nll",
                    f"position {j}: produced {a!r}, committed {b!r}, |diff| {d:.3g} "
                    f"(non-finite or > {TOLERANCE})",
                )
            max_pt = max(max_pt, d)
        if p.per_token_nll != c.per_token_nll:
            exact = False
        if not (math.isfinite(p.span_mean_nll) and math.isfinite(c.span_mean_nll)):
            return _mismatch(
                n, c.item_id, "span_mean_nll",
                f"non-finite: produced {p.span_mean_nll!r}, committed {c.span_mean_nll!r}",
            )
        d = abs(p.span_mean_nll - c.span_mean_nll)
        max_span = max(max_span, d)
        if p.span_mean_nll != c.span_mean_nll:
            exact = False
        pd, cd = p.digits_only_mean_nll, c.digits_only_mean_nll
        if (pd is None) != (cd is None):
            return _mismatch(
                n, c.item_id, "digits_only_mean_nll", f"produced {pd!r}, committed {cd!r}"
            )
        if pd is not None:
            if not (math.isfinite(pd) and math.isfinite(cd)):
                return _mismatch(
                    n, c.item_id, "digits_only_mean_nll",
                    f"non-finite: produced {pd!r}, committed {cd!r}",
                )
            max_dig = max(max_dig, abs(pd - cd))
            if pd != cd:
                exact = False
    return GateResult(
        "tier1" if exact else "tier2", n,
        max_per_token_diff=max_pt, max_span_mean_diff=max_span, max_digits_only_diff=max_dig,
    )


def compare_files(produced: Path, committed: Path) -> GateResult:
    return compare_records(nll_scorer.read_records(produced), nll_scorer.read_records(committed))


def format_result(r: GateResult) -> str:
    order = "compared in recorded (file) order"
    if r.outcome == "tier1":
        return f"TIER 1: reproduced exactly ({r.n_records} records, {order})"
    if r.outcome == "tier2":
        return (
            f"TIER 2: reproduced within floating-point tolerance ({r.n_records} records, "
            f"{order}); max |per-token NLL diff| {r.max_per_token_diff:.6g} "
            f"(bound {TOLERANCE}); max |span_mean_nll diff| {r.max_span_mean_diff:.6g}; "
            f"max |digits_only_mean_nll diff| {r.max_digits_only_diff:.6g}"
        )
    where = f"first offending item {r.offending_item!r}, field {r.offending_field}"
    return f"MISMATCH ({order}): {where}: {r.detail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tiered NLL reproduction-gate comparator.")
    parser.add_argument("produced", type=Path)
    parser.add_argument("committed", type=Path)
    args = parser.parse_args(argv)
    result = compare_files(args.produced, args.committed)
    print(format_result(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
