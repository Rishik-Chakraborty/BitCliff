import math
import random

import pytest

from bitcliff_pipeline.analysis import (
    Cell,
    analyze_cell,
    cell_state,
    cliff,
    holm,
    mcnemar_exact_p,
    paired_bootstrap_ci,
)


# ---------------------------------------------------------------------------
# mcnemar_exact_p
# ---------------------------------------------------------------------------


def test_mcnemar_b_c_zero_is_one():
    assert mcnemar_exact_p(0, 0) == 1.0


def test_mcnemar_b10_c0():
    # n = 10, k = min(b, c) = 0 -> p = 2 * P(X<=0) = 2 * 0.5**10
    p = mcnemar_exact_p(10, 0)
    assert math.isclose(p, 2 * 0.5**10, rel_tol=1e-9)


def test_mcnemar_c10_b0_symmetric():
    # swapping b and c must give the identical p-value
    assert mcnemar_exact_p(0, 10) == mcnemar_exact_p(10, 0)


def test_mcnemar_symmetric_equal_b_c():
    # b == c: the exact discordant-pair count is central -> p should be 1.0
    assert mcnemar_exact_p(5, 5) == 1.0


def test_mcnemar_p_never_exceeds_one():
    for b, c in [(1, 1), (2, 3), (0, 1), (50, 49)]:
        p = mcnemar_exact_p(b, c)
        assert 0.0 <= p <= 1.0


def test_mcnemar_large_n_still_computes():
    # n up to ~500 per the spec; must not blow up or raise
    p = mcnemar_exact_p(260, 240)
    assert 0.0 <= p <= 1.0


def test_mcnemar_one_sided_extreme_small_p():
    # b=0, c=1 -> n=1, k=0 -> p = 2 * P(X<=0) = 2*0.5 = 1.0 (capped)
    assert mcnemar_exact_p(0, 1) == 1.0


def test_mcnemar_mid_case_2_and_8():
    # n=10, k=min(2,8)=2 -> tail = C(10,0)+C(10,1)+C(10,2) = 1+10+45 = 56
    # p = 2*56/1024 = 112/1024
    assert mcnemar_exact_p(2, 8) == pytest.approx(112 / 1024)


# ---------------------------------------------------------------------------
# paired_bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_all_pairs_identical_gives_zero_width_ci():
    # every item: f16 correct, quant correct -> delta is always 0 no matter
    # how we resample -> CI collapses to (0.0, 0.0)
    pairs = [(True, True)] * 20
    lo, hi = paired_bootstrap_ci(pairs, 1000, random.Random(1))
    assert lo == 0.0
    assert hi == 0.0


def test_bootstrap_ci_all_pairs_identical_wrong_gives_zero_width_ci():
    pairs = [(False, False)] * 20
    lo, hi = paired_bootstrap_ci(pairs, 1000, random.Random(1))
    assert lo == 0.0
    assert hi == 0.0


def test_bootstrap_ci_deterministic_same_seed():
    pairs = [(True, False), (False, True), (True, True), (False, False)] * 10
    ci_a = paired_bootstrap_ci(pairs, 2000, random.Random(8271))
    ci_b = paired_bootstrap_ci(pairs, 2000, random.Random(8271))
    assert ci_a == ci_b


def test_bootstrap_ci_different_seed_may_differ():
    pairs = [(True, False), (False, True), (True, True), (False, False)] * 10
    ci_a = paired_bootstrap_ci(pairs, 2000, random.Random(1))
    ci_b = paired_bootstrap_ci(pairs, 2000, random.Random(2))
    assert ci_a != ci_b


def test_bootstrap_ci_sign_convention_quant_minus_f16():
    # every item: f16 correct, quant wrong -> every resample has
    # f16_acc = 1.0, quant_acc = 0.0 -> delta = quant - f16 = -1.0, always.
    # Pins the sign of the reported delta (quant minus F16, not the reverse).
    pairs = [(True, False)] * 20
    lo, hi = paired_bootstrap_ci(pairs, 500, random.Random(3))
    assert (lo, hi) == (-1.0, -1.0)


def test_bootstrap_ci_empty_pairs_raises():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], 100, random.Random(1))


def test_bootstrap_ci_lo_le_hi():
    pairs = [(True, False), (False, True), (True, True), (False, False)] * 25
    lo, hi = paired_bootstrap_ci(pairs, 5000, random.Random(42))
    assert lo <= hi


def test_bootstrap_ci_percentile_index_convention():
    # Documented convention: sort ascending, lo index = floor(0.025*(N-1)),
    # hi index = ceil(0.975*(N-1)). Verify against a hand-rolled resample
    # using the exact same RNG consumption pattern as the implementation.
    pairs = [(True, False), (False, True), (True, True), (False, False)] * 5
    n = len(pairs)
    n_resamples = 200
    rng_ours = random.Random(99)
    lo, hi = paired_bootstrap_ci(pairs, n_resamples, rng_ours)

    rng_ref = random.Random(99)
    deltas = []
    for _ in range(n_resamples):
        resampled = [pairs[rng_ref.randrange(n)] for _ in range(n)]
        f16_acc = sum(1 for f, q in resampled if f) / n
        quant_acc = sum(1 for f, q in resampled if q) / n
        deltas.append(quant_acc - f16_acc)
    deltas.sort()
    lo_idx = math.floor(0.025 * (n_resamples - 1))
    hi_idx = math.ceil(0.975 * (n_resamples - 1))
    assert lo == deltas[lo_idx]
    assert hi == deltas[hi_idx]


# ---------------------------------------------------------------------------
# cell_state
# ---------------------------------------------------------------------------


def test_cell_state_damaged():
    # CI excludes 0, delta beyond -margin
    assert cell_state(-0.10, (-0.15, -0.05), 0.03) == "damaged"


def test_cell_state_small_real_loss():
    # CI excludes 0, delta within margin (loss direction)
    assert cell_state(-0.01, (-0.02, -0.005), 0.03) == "small_real_loss"


def test_cell_state_delta_exactly_at_margin_is_not_damaged():
    # damaged requires delta STRICTLY < -margin; delta == -margin exactly
    # is small_real_loss, not damaged.
    assert cell_state(-0.03, (-0.05, -0.01), 0.03) == "small_real_loss"


def test_cell_state_small_real_loss_on_significant_gain():
    # CI excludes 0 (all positive), delta is a GAIN but |delta| <= margin
    # -> still small_real_loss per §8 precision rule (both flip directions).
    assert cell_state(0.015, (0.005, 0.025), 0.03) == "small_real_loss"


def test_cell_state_equivalent_strictly_within_margin():
    assert cell_state(0.0, (-0.01, 0.01), 0.03) == "equivalent"


def test_cell_state_boundary_touching_margin_is_not_equivalent():
    # CI touches -margin exactly (not strictly within) and includes 0
    # -> not equivalent (must fall to indeterminate)
    assert cell_state(-0.01, (-0.03, 0.005), 0.03) == "indeterminate"


def test_cell_state_boundary_touching_positive_margin_is_not_equivalent():
    assert cell_state(0.01, (-0.005, 0.03), 0.03) == "indeterminate"


def test_cell_state_indeterminate():
    # CI includes 0 and is not entirely within margin
    assert cell_state(-0.02, (-0.10, 0.02), 0.03) == "indeterminate"


def test_cell_state_ci_excludes_zero_positive_delta_beyond_margin_is_indeterminate():
    # CI excludes 0, delta is a significant GAIN beyond +margin: none of
    # §8's states 1-3 apply (not a loss -> not damaged; |delta| > margin ->
    # not "within M" -> not small_real_loss; CI excludes 0 -> not
    # equivalent). Falls through to indeterminate.
    assert cell_state(0.10, (0.05, 0.15), 0.03) == "indeterminate"


# ---------------------------------------------------------------------------
# analyze_cell
# ---------------------------------------------------------------------------


def test_analyze_cell_basic_fields():
    pairs = [(True, True)] * 8 + [(True, False)] * 2  # b=2, c=0
    cell = analyze_cell(pairs, margin=0.03, n_resamples=500, rng=random.Random(8271))
    assert isinstance(cell, Cell)
    assert cell.n == 10
    assert cell.b == 2
    assert cell.c == 0
    assert math.isclose(cell.delta, 0.8 - 1.0)
    assert cell.p == mcnemar_exact_p(2, 0)
    assert cell.ci_lo <= cell.delta <= cell.ci_hi
    assert cell.state in {"damaged", "small_real_loss", "equivalent", "indeterminate"}


def test_analyze_cell_all_equivalent():
    pairs = [(True, True)] * 50
    cell = analyze_cell(pairs, margin=0.03, n_resamples=200, rng=random.Random(1))
    assert cell.delta == 0.0
    assert cell.ci_lo == 0.0 and cell.ci_hi == 0.0
    assert cell.state == "equivalent"
    assert cell.b == 0
    assert cell.c == 0
    assert cell.p == 1.0


def test_analyze_cell_flip_directions_counted_independently():
    # b = F16-right-quant-wrong, c = F16-wrong-quant-right
    pairs = [(True, False)] * 3 + [(False, True)] * 5 + [(True, True)] * 2
    cell = analyze_cell(pairs, margin=0.03, n_resamples=100, rng=random.Random(2))
    assert cell.b == 3
    assert cell.c == 5


def test_analyze_cell_empty_pairs_raises():
    with pytest.raises(ValueError):
        analyze_cell([], margin=0.03, n_resamples=100, rng=random.Random(1))


# ---------------------------------------------------------------------------
# holm
# ---------------------------------------------------------------------------


def test_holm_worked_three_p_example():
    # Classic worked example: p = 0.01, 0.02, 0.03; alpha = 0.05
    # sorted: 0.01 (thresh 0.05/3=0.01667, reject), 0.02 (thresh 0.05/2=0.025, reject),
    # 0.03 (thresh 0.05/1=0.05, reject)
    pvalues = {"a": 0.01, "b": 0.02, "c": 0.03}
    result = holm(pvalues, alpha=0.05)
    assert result == {"a": True, "b": True, "c": True}


def test_holm_stops_at_first_failure():
    # p = 0.01, 0.04, 0.20; alpha = 0.05
    # sorted: 0.01 (thresh 0.01667, reject), 0.04 (thresh 0.025, FAIL -> stop),
    # 0.20 (thresh 0.05, not reached because already stopped)
    pvalues = {"a": 0.01, "b": 0.04, "c": 0.20}
    result = holm(pvalues, alpha=0.05)
    assert result == {"a": True, "b": False, "c": False}


def test_holm_step_down_stops_even_if_later_p_passes_its_own_threshold():
    # p = 0.03, 0.04; alpha = 0.05. Sorted: a=0.03 (thresh 0.05/2=0.025,
    # FAILS -> procedure stops), b=0.04 (thresh 0.05/1=0.05, would pass on
    # its own, but the step-down has already stopped at a, so b is also
    # False). An implementation that tests each p independently against
    # its own threshold (no stopping) would wrongly report b: True.
    pvalues = {"a": 0.03, "b": 0.04}
    result = holm(pvalues, alpha=0.05)
    assert result == {"a": False, "b": False}


def test_holm_all_fail():
    pvalues = {"a": 0.9, "b": 0.8}
    result = holm(pvalues, alpha=0.05)
    assert result == {"a": False, "b": False}


def test_holm_single_pvalue():
    assert holm({"a": 0.04}, alpha=0.05) == {"a": True}
    assert holm({"a": 0.06}, alpha=0.05) == {"a": False}


def test_holm_returns_all_keys():
    pvalues = {"a": 0.001, "b": 0.5, "c": 0.9, "d": 0.002}
    result = holm(pvalues, alpha=0.05)
    assert set(result.keys()) == set(pvalues.keys())


# ---------------------------------------------------------------------------
# cliff
# ---------------------------------------------------------------------------

LADDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K", "IQ2_M"]


def test_cliff_monotone_case():
    states = [
        ("Q8_0", "equivalent"),
        ("Q6_K", "equivalent"),
        ("Q5_K_M", "equivalent"),
        ("Q4_K_M", "damaged"),
        ("Q3_K_M", "damaged"),
        ("Q2_K", "damaged"),
        ("IQ2_M", "damaged"),
    ]
    rung, non_monotonic = cliff(states, LADDER)
    assert rung == "Q4_K_M"
    assert non_monotonic is False


def test_cliff_no_damage_returns_none():
    states = [(r, "equivalent") for r in LADDER]
    rung, non_monotonic = cliff(states, LADDER)
    assert rung is None
    assert non_monotonic is False


def test_cliff_non_monotonic_case():
    states = [
        ("Q8_0", "equivalent"),
        ("Q6_K", "damaged"),
        ("Q5_K_M", "equivalent"),
        ("Q4_K_M", "damaged"),
        ("Q3_K_M", "damaged"),
        ("Q2_K", "damaged"),
        ("IQ2_M", "damaged"),
    ]
    rung, non_monotonic = cliff(states, LADDER)
    assert rung == "Q6_K"  # highest-precision damaged rung, literally
    assert non_monotonic is True


def test_cliff_all_damaged():
    states = [(r, "damaged") for r in LADDER]
    rung, non_monotonic = cliff(states, LADDER)
    assert rung == "Q8_0"
    assert non_monotonic is False


def test_cliff_single_damaged_at_bottom():
    states = [(r, "equivalent") for r in LADDER[:-1]] + [(LADDER[-1], "damaged")]
    rung, non_monotonic = cliff(states, LADDER)
    assert rung == "IQ2_M"
    assert non_monotonic is False


def test_cliff_unknown_rung_raises():
    # A typo'd rung label (not in ladder_order) must not be silently
    # dropped -- especially dangerous if it was "damaged".
    states = [("Q8_0", "equivalent"), ("Q4_K_MM", "damaged")]  # typo'd label
    with pytest.raises(ValueError):
        cliff(states, LADDER)
