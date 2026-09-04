"""PREREG.md §8 statistics core.

Pure functions over paired (F16, quant) correctness records — no I/O here
(see ``scripts/analyze_0b.py`` for the driver that reads run data and writes
``analysis/0b/cells.csv`` / ``FINDINGS_0B.md``).

Registered definitions implemented verbatim from PREREG §8:

- Margin M = 0.03 absolute accuracy per suite.
- Per-cell inference: McNemar exact two-sided test on the paired accuracy
  difference, plus a two-sided 95% CI on delta-accuracy via paired
  bootstrap (10,000 resamples).
- Four cell states: damaged, small_real_loss, equivalent, indeterminate.
- Cliff: per (model, suite), the highest-precision rung whose cell is
  Damaged; non-monotonic rungs are flagged, never smoothed.
- Multiplicity: per-cell p-values are descriptive (no correction); only a
  headline claim that aggregates across cells is Holm-Bonferroni corrected
  over the family of cells it aggregates.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Hashable, TypeVar

K = TypeVar("K", bound=Hashable)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test p-value on discordant pairs (b, c).

    b = items F16 got right that the quant lost (F16-right, quant-wrong).
    c = items the quant gained (F16-wrong, quant-right).

    Under the null, the discordant count b is Binomial(n=b+c, p=0.5)
    distributed. The exact two-sided p-value is twice the smaller tail:

        n = b + c
        k = min(b, c)
        p = min(1, 2 * P(X <= k))   for X ~ Binomial(n, 0.5)

    p = 1.0 when b + c == 0 (no discordant pairs at all).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = Fraction(sum(math.comb(n, i) for i in range(k + 1)), 2**n)
    return min(1.0, float(2 * tail))


def paired_bootstrap_ci(
    pairs: list[tuple[bool, bool]], n_resamples: int, rng: random.Random
) -> tuple[float, float]:
    """Two-sided 95% percentile-bootstrap CI on delta-accuracy = quant - f16.

    Each pair is (f16_correct, quant_correct) for one item. Each resample
    draws n items with replacement (n = len(pairs)) and computes
    Δacc = mean(quant) - mean(f16) over that resample. The CI is the
    2.5th/97.5th empirical percentile of the ``n_resamples`` resample
    deltas.

    Percentile convention (documented per the 0B analysis plan): sort the
    resample deltas ascending, then

        lo_index = floor(0.025 * (n_resamples - 1))
        hi_index = ceil(0.975 * (n_resamples - 1))

    and return (sorted_deltas[lo_index], sorted_deltas[hi_index]).

    Degenerate case: an empty ``pairs`` list returns (0.0, 0.0).
    """
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0)

    deltas = []
    for _ in range(n_resamples):
        resampled = [pairs[rng.randrange(n)] for _ in range(n)]
        f16_acc = sum(1 for f16_ok, _ in resampled if f16_ok) / n
        quant_acc = sum(1 for _, quant_ok in resampled if quant_ok) / n
        deltas.append(quant_acc - f16_acc)

    deltas.sort()
    lo_idx = math.floor(0.025 * (n_resamples - 1))
    hi_idx = math.ceil(0.975 * (n_resamples - 1))
    return (deltas[lo_idx], deltas[hi_idx])


def cell_state(delta: float, ci: tuple[float, float], margin: float) -> str:
    """Classify a cell into one of the four PREREG §8 states.

    - "damaged": CI excludes 0 and delta < -margin (exceeds M loss).
    - "small_real_loss": CI excludes 0 and NOT damaged — this includes a
      loss within margin AND a significant gain within margin (both flip
      directions are reported; §8's "point estimate is within M" governs
      either sign once the CI has already excluded 0).
    - "equivalent": the ENTIRE CI lies strictly within (-margin, +margin).
      A CI that merely touches the boundary (lo == -margin or hi == margin)
      does NOT count as equivalent — §8 requires strict inclusion.
    - "indeterminate": none of the above.
    """
    lo, hi = ci
    excludes_zero = lo > 0 or hi < 0
    if excludes_zero:
        return "damaged" if delta < -margin else "small_real_loss"
    if lo > -margin and hi < margin:
        return "equivalent"
    return "indeterminate"


@dataclass(frozen=True)
class Cell:
    delta: float
    ci_lo: float
    ci_hi: float
    p: float
    b: int
    c: int
    state: str
    n: int


def analyze_cell(
    pairs: list[tuple[bool, bool]],
    *,
    margin: float = 0.03,
    n_resamples: int = 10_000,
    rng: random.Random,
) -> Cell:
    """Run the full §8 per-cell pipeline on one (quant, suite) pair set."""
    n = len(pairs)
    f16_acc = sum(1 for f16_ok, _ in pairs if f16_ok) / n
    quant_acc = sum(1 for _, quant_ok in pairs if quant_ok) / n
    delta = quant_acc - f16_acc

    b = sum(1 for f16_ok, quant_ok in pairs if f16_ok and not quant_ok)
    c = sum(1 for f16_ok, quant_ok in pairs if not f16_ok and quant_ok)
    p = mcnemar_exact_p(b, c)

    ci_lo, ci_hi = paired_bootstrap_ci(pairs, n_resamples, rng)
    state = cell_state(delta, (ci_lo, ci_hi), margin)

    return Cell(
        delta=delta,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        p=p,
        b=b,
        c=c,
        state=state,
        n=n,
    )


def holm(pvalues: dict[K, float], alpha: float = 0.05) -> dict[K, bool]:
    """Holm-Bonferroni step-down correction. Returns {key: rejected}."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    result: dict[K, bool] = {}
    stopped = False
    for i, (key, p) in enumerate(items):
        threshold = alpha / (m - i)
        if not stopped and p <= threshold:
            result[key] = True
        else:
            stopped = True
            result[key] = False
    return result


def cliff(
    states_by_rung: list[tuple[str, str]], ladder_order: list[str]
) -> tuple[str | None, bool]:
    """The highest-precision rung whose cell is Damaged, plus a
    non-monotonic flag.

    ``states_by_rung`` is a list of (rung_label, state) pairs; only rungs
    present are considered. Rungs are walked in ``ladder_order`` (highest
    precision first). The cliff is the FIRST rung in that order whose state
    is "damaged" (None if no rung is damaged) — this is taken literally
    per PREREG §8, even when the damaged states don't form a clean suffix.

    ``non_monotonic`` is True iff a "damaged" rung appears earlier (higher
    precision) in ladder order than some rung whose state is NOT "damaged"
    — i.e. the damaged states don't form a clean suffix down the ladder.
    """
    state_map = dict(states_by_rung)
    ordered_states = [state_map[r] for r in ladder_order if r in state_map]

    cliff_rung = None
    for label in ladder_order:
        if label in state_map and state_map[label] == "damaged":
            cliff_rung = label
            break

    non_monotonic = False
    damaged_seen = False
    for state in ordered_states:
        if state == "damaged":
            damaged_seen = True
        elif damaged_seen:
            non_monotonic = True
            break

    return cliff_rung, non_monotonic
