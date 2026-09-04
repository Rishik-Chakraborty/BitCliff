#!/usr/bin/env python3
"""0B confirmatory analysis driver, PREREG §8 (+ §5 shootout trigger, §4 ladder).

Reads the four 0B run trees under ``runs-cloud/pipeline/runs/`` and writes
``analysis/0b/cells.csv`` + ``analysis/0b/FINDINGS_0B.md``:

- **Ladder cells**: for each of the two reference models
  (0b-llama-8b-ladder / 0b-qwen-7b-ladder), each of the 7 registered ladder
  rungs (Amendment 2 §C) vs that run's own F16, across all 4 scored suites.
- **Arm cells**: for each of the two shootout arms
  (0b-shootout-arm1 / 0b-arm2-official), each non-F16 file vs that run's
  own F16, across all 4 scored suites.
- **Cliffs + non-monotonic flags** per (ladder run, suite), via
  ``bitcliff_pipeline.analysis.cliff``.
- **Holm headline verdicts**: per (ladder run, suite) with a cliff, the
  claim "damaged from the cliff rung down" survives iff every cell at the
  cliff rung and below is Holm-rejected over that suite's 7-cell ladder
  family (PREREG §8's dual rule; family membership per plan
  ``docs/superpowers/plans/2026-09-03-0b-analysis.md``).
- **Shootout trigger** (PREREG §5): same-label quant-vs-quant pairs across
  Arm 1's four uploaders (bartowski from the ladder run + unsloth /
  mradermacher_static / mradermacher_i1 from the arm run) at Q4_K_M and
  Q3_K_M, and Arm 2's official-vs-bartowski at Q4_K_M and Q3_K_M. A
  cross-run pair is only computed when the two runs' item id sets agree
  for that suite (checked against ``items.jsonl``); otherwise it is
  reported "not pairable (item sets differ)" and excluded from the trigger
  evaluation. The trigger fires if any *computed* pair's bootstrap CI
  excludes 0, or the pair's two files land in different §8 cell states
  against their own run's F16.

No I/O happens in ``bitcliff_pipeline.analysis`` (the pure §8 stats core);
all file reading/writing lives here.

**Seed rule (PROVISIONAL pending OPEN_QUESTIONS.md §7 ratification):**
every per-cell bootstrap uses ``random.Random(f"8271:{run_id}:{quant_label}:{suite}")``.
Every shootout pairwise comparison uses
``random.Random(f"8271:pair:{name_a}:{name_b}:{suite}")`` where
``name_a``/``name_b`` are the two files' shootout display names
(e.g. ``bartowski_Q4_K_M``, ``unsloth_Q4_K_M``), sorted alphabetically so
the seed does not depend on enumeration order.

Deterministic: two runs over the same data produce byte-identical
``cells.csv`` / ``FINDINGS_0B.md`` (no timestamps, no commit hashes in the
content).
"""

from __future__ import annotations

import csv
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bitcliff_pipeline.analysis import Cell, analyze_cell, cliff, holm

# ---------------------------------------------------------------------------
# Registered constants
# ---------------------------------------------------------------------------

MARGIN = 0.03
ALPHA = 0.05
N_RESAMPLES = 10_000

LADDER_ORDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K", "IQ2_M"]
SUITES = ["longctx_retrieval", "arithmetic", "arithmetic_twins", "factual_qa"]

SEED_RULE = 'random.Random(f"8271:{run_id}:{quant_label}:{suite}")'
PAIR_SEED_RULE = 'random.Random(f"8271:pair:{name_a}:{name_b}:{suite}")  # name_a, name_b sorted alphabetically'

RUN_IDS = [
    "0b-llama-8b-ladder",
    "0b-qwen-7b-ladder",
    "0b-shootout-arm1",
    "0b-arm2-official",
]
LADDER_RUN_IDS = ["0b-llama-8b-ladder", "0b-qwen-7b-ladder"]
RUN_KIND = {
    "0b-llama-8b-ladder": "ladder",
    "0b-qwen-7b-ladder": "ladder",
    "0b-shootout-arm1": "arm",
    "0b-arm2-official": "arm",
}
RUN_MODEL = {
    "0b-llama-8b-ladder": "llama-3.1-8b-instruct",
    "0b-qwen-7b-ladder": "qwen2.5-7b-instruct",
    "0b-shootout-arm1": "llama-3.1-8b-instruct",
    "0b-arm2-official": "qwen2.5-7b-instruct",
}

# Shootout registered scope (PREREG §5). display name -> (run_id, quant_label).
ARM1_LEVELS = ["Q4_K_M", "Q3_K_M"]
ARM2_LEVELS = ["Q4_K_M", "Q3_K_M"]


def _arm1_files(level: str) -> list[tuple[str, str, str]]:
    return [
        (f"bartowski_{level}", "0b-llama-8b-ladder", level),
        (f"unsloth_{level}", "0b-shootout-arm1", f"unsloth_{level}"),
        (f"mradermacher_static_{level}", "0b-shootout-arm1", f"mradermacher_static_{level}"),
        (f"mradermacher_i1_{level}", "0b-shootout-arm1", f"mradermacher_i1_{level}"),
    ]


def _arm2_files(level: str) -> list[tuple[str, str, str]]:
    return [
        (f"official_{level}", "0b-arm2-official", level),
        (f"bartowski_{level}", "0b-qwen-7b-ladder", level),
    ]


CELLS_CSV_FIELDS = [
    "run_id",
    "model",
    "suite",
    "quant_label",
    "n",
    "acc_f16",
    "acc_quant",
    "delta",
    "ci_lo",
    "ci_hi",
    "p_mcnemar",
    "b_lost",
    "c_gained",
    "truncated_f16",
    "truncated_quant",
    "state",
    "seed_rule",
]


# ---------------------------------------------------------------------------
# Data loading (I/O boundary)
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


@dataclass(frozen=True)
class RunData:
    run_id: str
    graded: dict[tuple[str, str], dict[str, tuple[bool, bool]]]
    item_ids: dict[str, set[str]]


def load_run(run_id: str, run_dir: Path) -> RunData:
    """Load one run's grades + item id sets.

    ``graded[(suite, quant_label)][item_id] = (is_correct, truncated)``.
    ``item_ids[suite]`` is the set of item ids in that run's ``items.jsonl``
    for that suite (used for cross-run pairability checks in the shootout).
    """
    graded: dict[tuple[str, str], dict[str, tuple[bool, bool]]] = {}
    for rec in read_jsonl(run_dir / "grades.jsonl"):
        key = (rec["suite"], rec["quant_label"])
        graded.setdefault(key, {})[rec["item_id"]] = (
            rec["state"] == "correct",
            bool(rec["truncated"]),
        )
    item_ids: dict[str, set[str]] = {}
    for rec in read_jsonl(run_dir / "items.jsonl"):
        item_ids.setdefault(rec["suite"], set()).add(rec["id"])
    return RunData(run_id=run_id, graded=graded, item_ids=item_ids)


def load_all_runs(runs_root: Path, run_ids: list[str] = RUN_IDS) -> dict[str, RunData]:
    return {run_id: load_run(run_id, runs_root / run_id) for run_id in run_ids}


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def build_pairs(
    map_a: dict[str, tuple[bool, bool]],
    map_b: dict[str, tuple[bool, bool]],
    *,
    context: str,
) -> tuple[list[tuple[bool, bool]], int, int]:
    """Pair two (item_id -> (correct, truncated)) maps on identical item ids.

    Returns (pairs, truncated_a, truncated_b). ``pairs[i] = (a_correct, b_correct)``
    in item_id-sorted order (order-independent of jsonl row order).

    Hard errors (raises ``ValueError``) on any id-set mismatch — a silent
    partial pairing would corrupt every downstream statistic.
    """
    ids_a = set(map_a)
    ids_b = set(map_b)
    if ids_a != ids_b:
        only_a = sorted(ids_a - ids_b)
        only_b = sorted(ids_b - ids_a)
        raise ValueError(
            f"item id mismatch ({context}): "
            f"{len(only_a)} id(s) only on the first side, "
            f"{len(only_b)} id(s) only on the second side "
            f"(first mismatches: {only_a[:5] + only_b[:5]})"
        )
    item_ids = sorted(ids_a)
    pairs = [(map_a[i][0], map_b[i][0]) for i in item_ids]
    truncated_a = sum(1 for i in item_ids if map_a[i][1])
    truncated_b = sum(1 for i in item_ids if map_b[i][1])
    return pairs, truncated_a, truncated_b


# ---------------------------------------------------------------------------
# Cells (quant vs own-run F16)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellRow:
    run_id: str
    model: str
    suite: str
    quant_label: str
    n: int
    acc_f16: float
    acc_quant: float
    delta: float
    ci_lo: float
    ci_hi: float
    p_mcnemar: float
    b_lost: int
    c_gained: int
    truncated_f16: int
    truncated_quant: int
    state: str
    seed_rule: str


def compute_cell_row(
    run_data: RunData,
    model: str,
    suite: str,
    quant_label: str,
    *,
    margin: float,
    n_resamples: int,
) -> CellRow:
    f16_map = run_data.graded[(suite, "F16")]
    quant_map = run_data.graded[(suite, quant_label)]
    pairs, trunc_f16, trunc_quant = build_pairs(
        f16_map, quant_map,
        context=f"run={run_data.run_id} suite={suite} F16 vs {quant_label}",
    )
    n = len(pairs)
    acc_f16 = sum(1 for f16_ok, _ in pairs if f16_ok) / n
    acc_quant = sum(1 for _, quant_ok in pairs if quant_ok) / n
    rng = random.Random(f"8271:{run_data.run_id}:{quant_label}:{suite}")
    cell: Cell = analyze_cell(pairs, margin=margin, n_resamples=n_resamples, rng=rng)
    return CellRow(
        run_id=run_data.run_id,
        model=model,
        suite=suite,
        quant_label=quant_label,
        n=n,
        acc_f16=acc_f16,
        acc_quant=acc_quant,
        delta=cell.delta,
        ci_lo=cell.ci_lo,
        ci_hi=cell.ci_hi,
        p_mcnemar=cell.p,
        b_lost=cell.b,
        c_gained=cell.c,
        truncated_f16=trunc_f16,
        truncated_quant=trunc_quant,
        state=cell.state,
        seed_rule=SEED_RULE,
    )


def _quant_labels_for(run_data: RunData, suite: str) -> set[str]:
    return {label for (s, label) in run_data.graded if s == suite} - {"F16"}


def compute_all_cells(
    runs_data: dict[str, RunData],
    *,
    margin: float = MARGIN,
    n_resamples: int = N_RESAMPLES,
    suites: list[str] = SUITES,
    ladder_order: list[str] = LADDER_ORDER,
    run_ids: list[str] = RUN_IDS,
) -> list[CellRow]:
    """Every ladder + arm cell, in deterministic (run, suite, quant_label) order.

    Ladder runs: hard error if a suite's non-F16 labels don't exactly match
    ``ladder_order`` (a missing or extra rung must not be silently dropped
    or silently included).
    """
    rows: list[CellRow] = []
    for run_id in run_ids:
        run_data = runs_data[run_id]
        model = RUN_MODEL[run_id]
        kind = RUN_KIND[run_id]
        for suite in suites:
            available = _quant_labels_for(run_data, suite)
            if kind == "ladder":
                missing = set(ladder_order) - available
                extra = available - set(ladder_order)
                if missing or extra:
                    raise ValueError(
                        f"run={run_id} suite={suite}: ladder label mismatch "
                        f"(missing={sorted(missing)}, extra={sorted(extra)})"
                    )
                labels = list(ladder_order)
            else:
                labels = sorted(available)
            for quant_label in labels:
                rows.append(
                    compute_cell_row(
                        run_data, model, suite, quant_label,
                        margin=margin, n_resamples=n_resamples,
                    )
                )
    return rows


# ---------------------------------------------------------------------------
# Cliffs + Holm headline verdicts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CliffResult:
    run_id: str
    suite: str
    cliff_rung: Optional[str]
    non_monotonic: bool


def compute_cliffs(
    cells: list[CellRow],
    *,
    ladder_run_ids: list[str] = LADDER_RUN_IDS,
    suites: list[str] = SUITES,
    ladder_order: list[str] = LADDER_ORDER,
) -> dict[tuple[str, str], CliffResult]:
    by_key = {(c.run_id, c.suite, c.quant_label): c for c in cells}
    results: dict[tuple[str, str], CliffResult] = {}
    for run_id in ladder_run_ids:
        for suite in suites:
            states_by_rung = [
                (rung, by_key[(run_id, suite, rung)].state) for rung in ladder_order
            ]
            rung, non_monotonic = cliff(states_by_rung, ladder_order)
            results[(run_id, suite)] = CliffResult(
                run_id=run_id, suite=suite, cliff_rung=rung, non_monotonic=non_monotonic
            )
    return results


@dataclass(frozen=True)
class HolmVerdict:
    run_id: str
    suite: str
    cliff_rung: str
    rungs_at_or_below: list[str]
    holm_result: dict[str, bool]
    failing_rungs: list[str]
    survives: bool


def compute_holm_verdicts(
    cells: list[CellRow],
    cliffs: dict[tuple[str, str], CliffResult],
    *,
    ladder_order: list[str] = LADDER_ORDER,
    alpha: float = ALPHA,
) -> dict[tuple[str, str], Optional[HolmVerdict]]:
    """Per (ladder run, suite) with a cliff: Holm over that suite's 7-cell
    ladder family; the "damaged from the cliff rung down" claim survives
    iff every rung at-or-below the cliff (in ladder order) is rejected.

    No cliff -> no candidate (``None``).
    """
    by_key = {(c.run_id, c.suite, c.quant_label): c for c in cells}
    verdicts: dict[tuple[str, str], Optional[HolmVerdict]] = {}
    for (run_id, suite), cliff_result in cliffs.items():
        if cliff_result.cliff_rung is None:
            verdicts[(run_id, suite)] = None
            continue
        family_p = {rung: by_key[(run_id, suite, rung)].p_mcnemar for rung in ladder_order}
        holm_result = holm(family_p, alpha=alpha)
        idx = ladder_order.index(cliff_result.cliff_rung)
        rungs_at_or_below = ladder_order[idx:]
        failing = [rung for rung in rungs_at_or_below if not holm_result[rung]]
        verdicts[(run_id, suite)] = HolmVerdict(
            run_id=run_id,
            suite=suite,
            cliff_rung=cliff_result.cliff_rung,
            rungs_at_or_below=rungs_at_or_below,
            holm_result=holm_result,
            failing_rungs=failing,
            survives=len(failing) == 0,
        )
    return verdicts


# ---------------------------------------------------------------------------
# Shootout trigger (PREREG §5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairResult:
    arm: str
    level: str
    suite: str
    name_a: str
    name_b: str
    pairable: bool
    reason: Optional[str]
    delta: Optional[float]
    ci_lo: Optional[float]
    ci_hi: Optional[float]
    excludes_zero: Optional[bool]
    state_a: Optional[str]
    state_b: Optional[str]
    states_differ: Optional[bool]


def _is_pairable(runs_data: dict[str, RunData], run_a: str, run_b: str, suite: str) -> bool:
    if run_a == run_b:
        return True
    return runs_data[run_a].item_ids.get(suite, set()) == runs_data[run_b].item_ids.get(
        suite, set()
    )


def compute_pair(
    runs_data: dict[str, RunData],
    cells_by_key: dict[tuple[str, str, str], CellRow],
    arm: str,
    level: str,
    suite: str,
    file_a: tuple[str, str, str],
    file_b: tuple[str, str, str],
    *,
    margin: float,
    n_resamples: int,
) -> PairResult:
    name_a, run_a, label_a = file_a
    name_b, run_b, label_b = file_b
    if not _is_pairable(runs_data, run_a, run_b, suite):
        return PairResult(
            arm=arm, level=level, suite=suite, name_a=name_a, name_b=name_b,
            pairable=False, reason="not pairable (item sets differ)",
            delta=None, ci_lo=None, ci_hi=None, excludes_zero=None,
            state_a=None, state_b=None, states_differ=None,
        )
    map_a = runs_data[run_a].graded[(suite, label_a)]
    map_b = runs_data[run_b].graded[(suite, label_b)]
    pairs, _, _ = build_pairs(
        map_a, map_b, context=f"shootout {arm} {level} {suite}: {name_a} vs {name_b}"
    )
    seed_a, seed_b = sorted([name_a, name_b])
    rng = random.Random(f"8271:pair:{seed_a}:{seed_b}:{suite}")
    cell = analyze_cell(pairs, margin=margin, n_resamples=n_resamples, rng=rng)
    excludes_zero = cell.ci_lo > 0 or cell.ci_hi < 0
    state_a = cells_by_key[(run_a, suite, label_a)].state
    state_b = cells_by_key[(run_b, suite, label_b)].state
    return PairResult(
        arm=arm, level=level, suite=suite, name_a=name_a, name_b=name_b,
        pairable=True, reason=None,
        delta=cell.delta, ci_lo=cell.ci_lo, ci_hi=cell.ci_hi,
        excludes_zero=excludes_zero,
        state_a=state_a, state_b=state_b, states_differ=state_a != state_b,
    )


@dataclass(frozen=True)
class ShootoutReport:
    pairs: list[PairResult]
    triggered: bool


def shootout_triggered(pairs: list[PairResult]) -> bool:
    """PREREG §5 trigger: any *computed* (pairable) same-label pair whose
    bootstrap CI excludes 0, OR whose two files land in different §8 cell
    states against their own run's F16."""
    return any(pr.pairable and (pr.excludes_zero or pr.states_differ) for pr in pairs)


def compute_shootout(
    runs_data: dict[str, RunData],
    cells: list[CellRow],
    *,
    margin: float = MARGIN,
    n_resamples: int = N_RESAMPLES,
    arm1_levels: list[str] = ARM1_LEVELS,
    arm2_levels: list[str] = ARM2_LEVELS,
    suites: list[str] = SUITES,
) -> ShootoutReport:
    cells_by_key = {(c.run_id, c.suite, c.quant_label): c for c in cells}
    results: list[PairResult] = []
    for level in arm1_levels:
        files = _arm1_files(level)
        for suite in suites:
            for file_a, file_b in itertools.combinations(files, 2):
                results.append(
                    compute_pair(
                        runs_data, cells_by_key, "arm1", level, suite, file_a, file_b,
                        margin=margin, n_resamples=n_resamples,
                    )
                )
    for level in arm2_levels:
        files = _arm2_files(level)
        for suite in suites:
            for file_a, file_b in itertools.combinations(files, 2):
                results.append(
                    compute_pair(
                        runs_data, cells_by_key, "arm2", level, suite, file_a, file_b,
                        margin=margin, n_resamples=n_resamples,
                    )
                )
    return ShootoutReport(pairs=results, triggered=shootout_triggered(results))


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisResult:
    cells: list[CellRow]
    cliffs: dict[tuple[str, str], CliffResult]
    holm_verdicts: dict[tuple[str, str], Optional[HolmVerdict]]
    shootout: ShootoutReport


def run_analysis(
    runs_root: Path,
    *,
    margin: float = MARGIN,
    n_resamples: int = N_RESAMPLES,
    run_ids: list[str] = RUN_IDS,
    ladder_run_ids: list[str] = LADDER_RUN_IDS,
    suites: list[str] = SUITES,
    ladder_order: list[str] = LADDER_ORDER,
    arm1_levels: list[str] = ARM1_LEVELS,
    arm2_levels: list[str] = ARM2_LEVELS,
) -> AnalysisResult:
    runs_data = load_all_runs(runs_root, run_ids)
    cells = compute_all_cells(
        runs_data, margin=margin, n_resamples=n_resamples,
        suites=suites, ladder_order=ladder_order, run_ids=run_ids,
    )
    cliffs = compute_cliffs(
        cells, ladder_run_ids=ladder_run_ids, suites=suites, ladder_order=ladder_order,
    )
    holm_verdicts = compute_holm_verdicts(cells, cliffs, ladder_order=ladder_order, alpha=ALPHA)
    shootout = compute_shootout(
        runs_data, cells, margin=margin, n_resamples=n_resamples,
        arm1_levels=arm1_levels, arm2_levels=arm2_levels, suites=suites,
    )
    return AnalysisResult(cells=cells, cliffs=cliffs, holm_verdicts=holm_verdicts, shootout=shootout)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _fmt(x: float, digits: int = 6) -> str:
    return f"{x:.{digits}f}"


def write_cells_csv(cells: list[CellRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(CELLS_CSV_FIELDS)
        for c in cells:
            writer.writerow([
                c.run_id,
                c.model,
                c.suite,
                c.quant_label,
                c.n,
                _fmt(c.acc_f16),
                _fmt(c.acc_quant),
                _fmt(c.delta),
                _fmt(c.ci_lo),
                _fmt(c.ci_hi),
                _fmt(c.p_mcnemar),
                c.b_lost,
                c.c_gained,
                c.truncated_f16,
                c.truncated_quant,
                c.state,
                c.seed_rule,
            ])


def _cell_table_row(c: CellRow) -> str:
    return (
        f"| {c.quant_label} | {c.n} | {_fmt(c.acc_f16, 4)} | {_fmt(c.acc_quant, 4)} "
        f"| {_fmt(c.delta, 4)} | [{_fmt(c.ci_lo, 4)}, {_fmt(c.ci_hi, 4)}] "
        f"| {_fmt(c.p_mcnemar, 4)} | {c.b_lost} | {c.c_gained} "
        f"| {c.truncated_f16} | {c.truncated_quant} | {c.state} |"
    )


CELL_TABLE_HEADER = (
    "| quant_label | n | acc_f16 | acc_quant | delta | 95% CI | p (McNemar) "
    "| b (lost) | c (gained) | trunc F16 | trunc quant | state |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def write_findings_md(result: AnalysisResult, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# 0B Confirmatory Findings (PREREG §8)")
    lines.append("")
    lines.append(
        "Generated by `scripts/analyze_0b.py` from `runs-cloud/pipeline/runs/` "
        "grade records. Pure statistics core: `src/bitcliff_pipeline/analysis.py` "
        "(PREREG §8 machinery, TDD, no I/O)."
    )
    lines.append("")

    # --- Fingerprint reference ---
    lines.append("## Machine / software fingerprint")
    lines.append("")
    lines.append(
        "Every generation run's determinism scope (PREREG §6) is bounded by the "
        "serving machine's hardware/software fingerprint, recorded at "
        "`runs-cloud/fingerprint.txt`. See that file for instance type, GPU, "
        "CUDA, llama.cpp commit, and llama-cpp-python build. Not restated here "
        "to avoid a second copy drifting out of sync."
    )
    lines.append("")

    # --- Seed disclosure ---
    lines.append("## Seed disclosure (PROVISIONAL — OPEN_QUESTIONS.md §7)")
    lines.append("")
    lines.append(
        "PREREG §8 registers the per-cell inference machinery (McNemar exact; "
        "two-sided 95% CI on Δaccuracy via paired bootstrap, 10,000 resamples) "
        "but no RNG seed for the bootstrap — unlike every sampling seed in §3. "
        "This run uses **seed 8271** (fresh; distinct from every registered "
        "seed: 42, 1301, 2024, 2718, 3141, 7411, 20260828), applied per cell as:"
    )
    lines.append("")
    lines.append(f"    {SEED_RULE}")
    lines.append("")
    lines.append(
        "so that cells are order-independent and reproducible. Every "
        "shootout pairwise comparison (§5, below) uses its own per-pair seed:"
    )
    lines.append("")
    lines.append(f"    {PAIR_SEED_RULE}")
    lines.append("")
    lines.append(
        "**This is provisional pending user ratification (OPEN_QUESTIONS.md "
        "§7).** If ratified, these numbers stand; any other seed choice "
        "requires a cheap, local, deterministic re-run of this script. "
        f"Registered values used: margin M = {MARGIN}, α = {ALPHA}, "
        f"n_resamples = {N_RESAMPLES}, two-sided 95% CI (percentile method)."
    )
    lines.append("")

    # --- Scope note ---
    lines.append("## Scope note (PREREG §4)")
    lines.append("")
    lines.append(
        "0B covers only the two precompute-only reference models "
        "(Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct) at the registered "
        "7-rung bartowski ladder (Amendment 2 §C: Q8_0, Q6_K, Q5_K_M, "
        "Q4_K_M, Q3_K_M, Q2_K, IQ2_M) plus the two uploader-shootout arms "
        "(§5). The 1.5B spectacle model and its in-house sub-2-bit spectacle "
        "rungs (IQ1_S / IQ1_M / IQ2_XXS; PREREG §4, `INHOUSE_QUANTS.md`) are "
        "out of scope here by construction — spectacle rungs never appear in "
        "reference tables, cliff badges, or Holm families (enforced in code, "
        "not by convention), and this analysis touches none of them."
    )
    lines.append("")

    # --- Per-run cell tables ---
    cells_by_run: dict[str, list[CellRow]] = {}
    for c in result.cells:
        cells_by_run.setdefault(c.run_id, []).append(c)

    lines.append("## Per-cell results")
    lines.append("")
    for run_id in RUN_IDS:
        run_cells = cells_by_run.get(run_id, [])
        if not run_cells:
            continue
        model = RUN_MODEL[run_id]
        kind = RUN_KIND[run_id]
        lines.append(f"### `{run_id}` ({model}, {kind})")
        lines.append("")
        cells_by_suite: dict[str, list[CellRow]] = {}
        for c in run_cells:
            cells_by_suite.setdefault(c.suite, []).append(c)
        for suite in SUITES:
            suite_cells = cells_by_suite.get(suite)
            if not suite_cells:
                continue
            lines.append(f"**{suite}**")
            lines.append("")
            lines.append(CELL_TABLE_HEADER)
            for c in suite_cells:
                lines.append(_cell_table_row(c))
            lines.append("")

    # --- Cliff table ---
    lines.append("## Cliffs (ladder runs only)")
    lines.append("")
    lines.append("| run_id | suite | cliff rung | non-monotonic |")
    lines.append("|---|---|---|---|")
    for run_id in LADDER_RUN_IDS:
        for suite in SUITES:
            cr = result.cliffs.get((run_id, suite))
            if cr is None:
                continue
            rung_str = cr.cliff_rung if cr.cliff_rung is not None else "(none)"
            lines.append(f"| {run_id} | {suite} | {rung_str} | {cr.non_monotonic} |")
    lines.append("")

    # --- Holm verdicts ---
    lines.append("## Holm-corrected headline verdicts")
    lines.append("")
    lines.append(
        'Candidate claim per (ladder run, suite) with a cliff: "damaged from '
        "the cliff rung down.\" Family = that suite's 7 ladder cells' McNemar "
        "p-values. Survives iff every cell at the cliff rung and below (in "
        "ladder order) is Holm-rejected over that family."
    )
    lines.append("")
    lines.append("| run_id | suite | cliff rung | rungs at/below cliff | verdict | failing rungs |")
    lines.append("|---|---|---|---|---|---|")
    for run_id in LADDER_RUN_IDS:
        for suite in SUITES:
            verdict = result.holm_verdicts.get((run_id, suite))
            if verdict is None:
                lines.append(f"| {run_id} | {suite} | (none) | — | no candidate (no cliff) | — |")
                continue
            rungs_str = ", ".join(verdict.rungs_at_or_below)
            verdict_str = "SURVIVES" if verdict.survives else "FAILS"
            failing_str = ", ".join(verdict.failing_rungs) if verdict.failing_rungs else "—"
            lines.append(
                f"| {run_id} | {suite} | {verdict.cliff_rung} | {rungs_str} "
                f"| {verdict_str} | {failing_str} |"
            )
    lines.append("")

    # --- Shootout trigger ---
    lines.append("## Uploader shootout trigger (PREREG §5)")
    lines.append("")
    lines.append(
        "Same-label quant-vs-quant pairs, paired on identical items: Arm 1 "
        "(bartowski from the ladder run, unsloth, mradermacher_static, "
        "mradermacher_i1 — 6 unordered pairs per label) at Q4_K_M and "
        "Q3_K_M; Arm 2 (official vs bartowski) at Q4_K_M and Q3_K_M. A "
        "cross-run pair is computed only where the two runs' item id sets "
        "agree for that suite (checked against `items.jsonl`); otherwise it "
        "is reported not pairable and excluded below."
    )
    lines.append("")
    lines.append(
        "| arm | level | suite | pair | pairable | delta | 95% CI | CI excludes 0 "
        "| state A | state B | states differ |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for pr in result.shootout.pairs:
        pair_str = f"{pr.name_a} vs {pr.name_b}"
        if not pr.pairable:
            lines.append(
                f"| {pr.arm} | {pr.level} | {pr.suite} | {pair_str} | NO ({pr.reason}) "
                f"| — | — | — | — | — | — |"
            )
            continue
        ci_str = f"[{_fmt(pr.ci_lo, 4)}, {_fmt(pr.ci_hi, 4)}]"
        lines.append(
            f"| {pr.arm} | {pr.level} | {pr.suite} | {pair_str} | yes "
            f"| {_fmt(pr.delta, 4)} | {ci_str} | {pr.excludes_zero} "
            f"| {pr.state_a} | {pr.state_b} | {pr.states_differ} |"
        )
    lines.append("")

    if result.shootout.triggered:
        lines.append(
            "**Trigger: FIRED.** At least one computed same-label pair's CI "
            "excludes 0, or lands its two files in different §8 cell states "
            "against their own run's F16. Per PREREG §5, extending the "
            "uploader shootout to Qwen2.5-7B-Instruct becomes a **registered "
            "follow-up measurement**, run after the launch analyses."
        )
    else:
        lines.append(
            "**Trigger: not fired.** No computed same-label pair's CI "
            "excludes 0, and no pair's two files land in different §8 cell "
            "states against their own run's F16. Absent the trigger, no 7B "
            "uploader shootout runs."
        )
    lines.append("")

    # --- Registered-machinery disclosure ---
    lines.append("## Disclosure: zero-discordance cells and the percentile bootstrap")
    lines.append("")
    lines.append(
        "Zero-discordance cells (b + c = 0, i.e. every resample reproduces "
        "the same paired accuracy) produce a degenerate (0, 0) bootstrap CI "
        "and classify Equivalent regardless of n under the registered "
        "percentile method. This is disclosed here because near-ceiling "
        "`longctx_retrieval` cells hit this condition: an exact-binomial "
        "upper bound on the discordance rate (e.g. ~3.7% at 0/96) is what the "
        "percentile bootstrap cannot see. A cell classified Equivalent under "
        "this rule is equivalent under the registered machinery, not "
        "necessarily under every alternative inferential lens."
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    pipeline_dir = Path(__file__).resolve().parents[1]
    runs_root = pipeline_dir / "runs-cloud" / "pipeline" / "runs"
    out_dir = pipeline_dir / "analysis" / "0b"

    result = run_analysis(runs_root)
    write_cells_csv(result.cells, out_dir / "cells.csv")
    write_findings_md(result, out_dir / "FINDINGS_0B.md")
    print(f"wrote {len(result.cells)} cells to {out_dir / 'cells.csv'}")
    print(f"wrote findings to {out_dir / 'FINDINGS_0B.md'}")
    n_cliffs = sum(1 for cr in result.cliffs.values() if cr.cliff_rung is not None)
    print(f"cliffs found: {n_cliffs} / {len(result.cliffs)} (ladder run, suite) pairs")
    print(f"shootout trigger: {'FIRED' if result.shootout.triggered else 'not fired'}")


if __name__ == "__main__":
    main()
