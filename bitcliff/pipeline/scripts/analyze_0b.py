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

**Seed rule (RATIFIED 2026-09-04, OPEN_QUESTIONS.md §7; seed in
``registered.BOOTSTRAP_SEED``):** every per-cell bootstrap uses ``random.Random(f"8271:{run_id}:{quant_label}:{suite}")``.
Every shootout pairwise comparison uses
``random.Random(f"8271:pair:{name_a}:{name_b}:{suite}")`` where
``name_a``/``name_b`` are the two files' shootout display names
(e.g. ``bartowski_Q4_K_M``, ``unsloth_Q4_K_M``), sorted alphabetically so
the seed does not depend on enumeration order.

Deterministic: two runs over the same data produce byte-identical
``cells.csv`` / ``FINDINGS_0B.md`` (no timestamps, no commit hashes in the
content).

**Annotations (Task 8, code turn 2026-09-27):** every note that used to be
hand-appended to the committed FINDINGS_0B.md after generation is emitted
here, each gated on the inputs that make it applicable, so a regeneration
keeps them and a non-0B invocation (e.g. the 1.5B/0B-prime ladder) never
prints 0B-only prose: the ratified-seed wording (seed ==
``registered.BOOTSTRAP_SEED``; the 2026-09-03 provisional history only for
the 0B run set), the Arm 2 imatrix footnote (``RUN_FOOTNOTES``, keyed by
run id), the pair-delta sign convention (verified row-by-row against the
per-file accuracies; the dated 2026-09-27 sentence only for the 0B run
set), the seed-robustness sweep section (0B run set + ratified seed +
``analysis/0b/sweep/`` data) and Appendix A (0B run set + ratified seed +
the first-pass factual_qa measurement). Where the committed text is fixed
prose, every data claim it makes is re-derived and checked here, and a
mismatch is a hard error rather than a silently stale sentence.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bitcliff_pipeline import registered
from bitcliff_pipeline.analysis import Cell, analyze_cell, cliff, holm

# ---------------------------------------------------------------------------
# Registered constants -- imported from bitcliff_pipeline.registered (the
# single home of every PREREG-registered constant, pre-rerun hardening,
# OPEN_QUESTIONS §8) instead of restated here.
# ---------------------------------------------------------------------------

MARGIN = registered.MARGIN
ALPHA = registered.ALPHA
N_RESAMPLES = registered.N_RESAMPLES

LADDER_ORDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K", "IQ2_M"]
SUITES = ["longctx_retrieval", "arithmetic", "arithmetic_twins", "factual_qa"]

SEED = registered.BOOTSTRAP_SEED  # ratified (OPEN_QUESTIONS §7); overridable via --seed for robustness sweeps


def seed_rule() -> str:
    return f'random.Random(f"{SEED}:{{run_id}}:{{quant_label}}:{{suite}}")'


def pair_seed_rule() -> str:
    return (f'random.Random(f"{SEED}:pair:{{name_a}}:{{name_b}}:{{suite}}")'
            "  # name_a, name_b sorted alphabetically")


SEED_RULE = seed_rule()
PAIR_SEED_RULE = pair_seed_rule()

RUN_IDS = [
    "0b-llama-8b-ladder",
    "0b-qwen-7b-ladder",
    "0b-shootout-arm1",
    "0b-arm2-official",
]
LADDER_RUN_IDS = ["0b-llama-8b-ladder", "0b-qwen-7b-ladder"]
# A run_id's kind ("ladder" vs "arm") is derived as membership in
# ladder_run_ids (Task 7) rather than kept as a separate hardcoded table --
# see compute_all_cells / write_findings_md.
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


def shootout_files(arm: str, level: str) -> list[tuple[str, str, str]]:
    """(display name, run_id, quant_label) for every file in ``arm`` at ``level``."""
    return {"arm1": _arm1_files, "arm2": _arm2_files}[arm](level)


def is_0b_run_set(run_ids: list[str], ladder_run_ids: list[str]) -> bool:
    """True iff this invocation analyzes exactly the 0B run set -- the gate
    for every 0B-only FINDINGS annotation (dated 0B prose, the seed sweep,
    Appendix A)."""
    return list(run_ids) == RUN_IDS and list(ladder_run_ids) == LADDER_RUN_IDS


CELLS_CSV_FIELDS = [
    "run_id",
    "source_run_id",
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
    "acc_inversion",
    "inversion_above",
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
    suite_source: dict[str, str] = dataclasses.field(default_factory=dict)
    """suite -> run_id the grades/item-ids were actually loaded from, when
    it differs from ``run_id`` (OPEN_QUESTIONS §8 factual_qa substitution);
    absent keys mean "this run's own data"."""


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


# OPEN_QUESTIONS §8 resolution (user ruling 2026-09-04, option (a)): the
# first 0B pass sampled a non-registered factual_qa item set (wrong-order
# M3 weights); the 0b2 rerun regenerated factual_qa for every rung on the
# registered set (item_set sha256 2e53ca0e…, gate-enforced). The analysis
# therefore sources factual_qa grades/item-ids from the 0b2 runs; every
# other suite still comes from the original 0b runs. cells.csv carries a
# source_run_id column making the substitution auditable per cell.
FACTUAL_QA_SOURCE = {
    "0b-llama-8b-ladder": "0b2-llama-8b-ladder",
    "0b-shootout-arm1": "0b2-shootout-arm1",
    "0b-qwen-7b-ladder": "0b2-qwen-7b-ladder",
    "0b-arm2-official": "0b2-arm2-official",
}


def load_all_runs(
    runs_root: Path,
    run_ids: list[str] = RUN_IDS,
    factual_qa_source: dict[str, str] = FACTUAL_QA_SOURCE,
) -> dict[str, RunData]:
    runs = {run_id: load_run(run_id, runs_root / run_id) for run_id in run_ids}
    for run_id, source_id in factual_qa_source.items():
        if run_id not in runs:
            continue
        # Substitution applies only when the run actually carries
        # factual_qa data (synthetic test trees may not).
        if not any(k[0] == "factual_qa" for k in runs[run_id].graded):
            continue
        src_dir = runs_root / source_id
        if not src_dir.exists():
            raise FileNotFoundError(
                f"factual_qa source run {source_id!r} (OPEN_QUESTIONS §8 "
                f"substitution for {run_id!r}) not found under {runs_root}"
            )
        src = load_run(source_id, src_dir)
        run = runs[run_id]
        graded = {k: v for k, v in run.graded.items() if k[0] != "factual_qa"}
        graded.update({k: v for k, v in src.graded.items() if k[0] == "factual_qa"})
        item_ids = dict(run.item_ids)
        item_ids["factual_qa"] = src.item_ids["factual_qa"]
        runs[run_id] = dataclasses.replace(
            run, graded=graded, item_ids=item_ids,
            suite_source={**run.suite_source, "factual_qa": source_id},
        )
    return runs


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
    source_run_id: str
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


def accuracy_inversions(
    cells: list["CellRow"], *, ladder_order: list[str] = LADDER_ORDER
) -> dict[tuple[str, str, str], str]:
    """PREREG §8's accuracy-level non-monotonicity flag (user ruling
    2026-09-06, OPEN_QUESTIONS §9): map (run_id, suite, quant_label) ->
    the higher-bits neighbor label this cell scored strictly ABOVE.

    Neighbor definition: for ladder runs, the previous rung in the
    registered LADDER_ORDER (F16 above Q8_0); for arm files, the
    same-uploader Q4_K_M is Q3_K_M's neighbor and F16 is every arm
    Q4_K_M's neighbor. Flags attach to the LOWER-bits cell. Strict
    inequality: ties are not inversions.
    """
    by_family: dict[tuple[str, str], dict[str, "CellRow"]] = {}
    for c in cells:
        by_family.setdefault((c.run_id, c.suite), {})[c.quant_label] = c
    out: dict[tuple[str, str, str], str] = {}
    for (run_id, suite), fam in by_family.items():
        labels = set(fam)
        if labels <= set(ladder_order):
            chain = ["F16"] + [q for q in ladder_order if q in fam]
        else:
            uploaders = sorted({l.rsplit("_Q", 1)[0] for l in labels if "_Q" in l and not l.startswith("Q")})
            chain = None
            pairs = []
            for l in sorted(labels):
                if l.endswith("Q4_K_M"):
                    pairs.append(("F16", l))
                elif l.endswith("Q3_K_M"):
                    q4 = l.replace("Q3_K_M", "Q4_K_M")
                    pairs.append((q4 if q4 in fam else "F16", l))
            for hi, lo in pairs:
                hi_acc = fam[lo].acc_f16 if hi == "F16" else fam[hi].acc_quant
                if fam[lo].acc_quant > hi_acc:
                    out[(run_id, suite, lo)] = hi
            continue
        for hi, lo in zip(chain, chain[1:]):
            hi_acc = fam[lo].acc_f16 if hi == "F16" else fam[hi].acc_quant
            if fam[lo].acc_quant > hi_acc:
                out[(run_id, suite, lo)] = hi
    return out


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
    rng = random.Random(f"{SEED}:{run_data.run_id}:{quant_label}:{suite}")
    cell: Cell = analyze_cell(pairs, margin=margin, n_resamples=n_resamples, rng=rng)
    return CellRow(
        run_id=run_data.run_id,
        source_run_id=run_data.suite_source.get(suite, run_data.run_id),
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
    ladder_run_ids: list[str] = LADDER_RUN_IDS,
    run_model: dict[str, str] = RUN_MODEL,
) -> list[CellRow]:
    """Every ladder + arm cell, in deterministic (run, suite, quant_label) order.

    Ladder runs: hard error if a suite's non-F16 labels don't exactly match
    ``ladder_order`` (a missing or extra rung must not be silently dropped
    or silently included). A run_id's kind is "ladder" iff it is in
    ``ladder_run_ids`` -- not a separate hardcoded table -- so a caller
    supplying new run ids (Task 7: the 1.5B/0B-prime single-ladder-run
    shape) never needs a matching RUN_KIND entry.
    """
    rows: list[CellRow] = []
    for run_id in run_ids:
        run_data = runs_data[run_id]
        model = run_model[run_id]
        kind = "ladder" if run_id in ladder_run_ids else "arm"
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
    rng = random.Random(f"{SEED}:pair:{seed_a}:{seed_b}:{suite}")
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
    inversions: dict[tuple[str, str, str], str] = dataclasses.field(default_factory=dict)
    # Task 7: the run/suite shape actually used to build this result, carried
    # alongside it so write_findings_md (and any other consumer) renders off
    # this run's shape instead of the module-level 0B defaults.
    run_ids: list[str] = dataclasses.field(default_factory=lambda: list(RUN_IDS))
    ladder_run_ids: list[str] = dataclasses.field(default_factory=lambda: list(LADDER_RUN_IDS))
    suites: list[str] = dataclasses.field(default_factory=lambda: list(SUITES))
    run_model: dict[str, str] = dataclasses.field(default_factory=lambda: dict(RUN_MODEL))


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
    run_model: dict[str, str] = RUN_MODEL,
    factual_qa_source: dict[str, str] = FACTUAL_QA_SOURCE,
) -> AnalysisResult:
    runs_data = load_all_runs(runs_root, run_ids, factual_qa_source=factual_qa_source)
    cells = compute_all_cells(
        runs_data, margin=margin, n_resamples=n_resamples,
        suites=suites, ladder_order=ladder_order, run_ids=run_ids,
        ladder_run_ids=ladder_run_ids, run_model=run_model,
    )
    cliffs = compute_cliffs(
        cells, ladder_run_ids=ladder_run_ids, suites=suites, ladder_order=ladder_order,
    )
    holm_verdicts = compute_holm_verdicts(cells, cliffs, ladder_order=ladder_order, alpha=ALPHA)
    shootout = compute_shootout(
        runs_data, cells, margin=margin, n_resamples=n_resamples,
        arm1_levels=arm1_levels, arm2_levels=arm2_levels, suites=suites,
    )
    return AnalysisResult(
        cells=cells, cliffs=cliffs, holm_verdicts=holm_verdicts, shootout=shootout,
        inversions=accuracy_inversions(cells, ladder_order=ladder_order),
        run_ids=list(run_ids), ladder_run_ids=list(ladder_run_ids), suites=list(suites),
        run_model=dict(run_model),
    )


def compute_first_pass_factual_qa(
    runs_root: Path,
    *,
    ladder_run_ids: list[str] = LADDER_RUN_IDS,
    ladder_order: list[str] = LADDER_ORDER,
    run_model: dict[str, str] = RUN_MODEL,
) -> AnalysisResult:
    """The first 0B pass's factual_qa ladder cells -- the NON-registered
    item set each 0b run itself sampled (OPEN_QUESTIONS §8), i.e. the same
    analysis with the 0b2 substitution switched off. Feeds Appendix A only;
    never enters a headline table."""
    return run_analysis(
        runs_root, run_ids=ladder_run_ids, ladder_run_ids=ladder_run_ids,
        suites=["factual_qa"], ladder_order=ladder_order,
        arm1_levels=[], arm2_levels=[], run_model=run_model, factual_qa_source={},
    )


def read_cells_csv(path: Path) -> list[CellRow]:
    """Parse a cells.csv (current or pre-``acc_inversion`` column set, e.g.
    the committed ``analysis/0b/sweep/cells-seed*.csv``) back into rows."""
    rows: list[CellRow] = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append(CellRow(
                run_id=r["run_id"], source_run_id=r["source_run_id"], model=r["model"],
                suite=r["suite"], quant_label=r["quant_label"], n=int(r["n"]),
                acc_f16=float(r["acc_f16"]), acc_quant=float(r["acc_quant"]),
                delta=float(r["delta"]), ci_lo=float(r["ci_lo"]), ci_hi=float(r["ci_hi"]),
                p_mcnemar=float(r["p_mcnemar"]), b_lost=int(r["b_lost"]),
                c_gained=int(r["c_gained"]), truncated_f16=int(r["truncated_f16"]),
                truncated_quant=int(r["truncated_quant"]), state=r["state"],
                seed_rule=r["seed_rule"],
            ))
    return rows


def load_sweep(sweep_dir: Path) -> dict[str, list[CellRow]]:
    """``{seed prefix: cells}`` from ``<sweep_dir>/cells-seed<N>.csv`` (the
    robustness sweep's outputs, ``--seed N``), ordered by seed."""
    files = sorted(sweep_dir.glob("cells-seed*.csv"),
                   key=lambda p: int(p.stem.removeprefix("cells-seed")))
    return {p.stem.removeprefix("cells-seed"): read_cells_csv(p) for p in files}


# ---------------------------------------------------------------------------
# Post-generation annotations (Task 8). Each was once hand-added to the
# committed FINDINGS_0B.md; the text below is byte-identical to it. Fixed
# prose tied to a dated event is emitted verbatim; the numbers inside it are
# filled from data, and every qualitative claim it makes is re-checked
# against the data (ValueError on mismatch -- never a stale sentence).
# ---------------------------------------------------------------------------

# Per-run footnotes, emitted directly under that run's per-cell heading
# whenever the run is analyzed.
RUN_FOOTNOTES = {
    "0b-arm2-official": (
        "Footnote (2026-09-08, OPEN_QUESTIONS §10): the two official Qwen files ran "
        "and are reported as `imatrix: false`; confirmed from their GGUF headers at "
        "the pinned revision (no `quantize.imatrix.*` keys) — evidence: "
        "`reference-manifests/evidence/qwen2.5-7b-official-gguf-headers.md`."
    ),
}

# OPEN_QUESTIONS §15(c): the sentence added by hand on 2026-09-27 (0B run
# set only), kept byte-identical -- including its now-fulfilled forward
# reference to this script.
SIGN_CONVENTION_0B = (
    'Sign convention (added 2026-09-27, OPEN_QUESTIONS §15(c)): for a row "A vs B", '
    "delta = acc(B) minus acc(A). All {n} rows below were checked against the "
    "per-file accuracies in `cells.csv`, and none contradicts it. This sentence was "
    "added by hand; it moves into `scripts/analyze_0b.py` in the next code turn so a "
    "regeneration keeps it."
)
SIGN_CONVENTION_GENERIC = (
    'Sign convention (OPEN_QUESTIONS §15(c)): for a row "A vs B", delta = acc(B) '
    "minus acc(A). {n_checked} of the {n_rows} rows below are computed; each was "
    "checked against the per-file accuracies in `cells.csv`, and none contradicts it."
)

_STATE_DISPLAY = {
    "equivalent": "Equivalent",
    "small_real_loss": "Small real loss",
    "indeterminate": "Indeterminate",
    "damaged": "Damaged",
}
_COUNT_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "five", 6: "six"}


def _u3(x: float) -> str:
    """3-dp with a Unicode minus (the sweep section's hand-typeset style)."""
    return f"{x:.3f}".replace("-", "\u2212")


# Seed-robustness sweep (2026-09-05): the cells whose §8 state is not
# invariant across the ratified seed + sweep seeds, with the per-seed states
# the prose asserts, a check for the prose's CI claim, and the prose itself.
# The set of seed-marginal cells in the data must equal this table's keys.
SEED_SWEEP_CELL_NOTES = {
    ("0b-qwen-7b-ladder", "factual_qa", "Q4_K_M"): {
        "states": {"ratified": "small_real_loss", "1": "small_real_loss",
                   "2": "indeterminate", "3": "indeterminate", "4": "indeterminate"},
        # "under seeds 2, 3 and 4 the CI upper bound touches/crosses 0"
        "check": lambda rows: all(rows[s].ci_hi >= 0 for s in ("2", "3", "4")),
        "text": (
            "- `{key}`: Δ = {delta}, CI\n"
            "  [{lo}, {hi}] under seed {seed} (→ {state}); under seeds 2, 3\n"
            "  and 4 the CI upper bound touches/crosses 0 (→ Indeterminate). Either\n"
            "  way the cell is neither Damaged nor Equivalent: a ~{pp}pp loss that the\n"
            "  n={n} draw cannot cleanly resolve against the ±{margin_pp}pp margin."
        ),
    },
    ("0b-shootout-arm1", "arithmetic", "mradermacher_static_Q3_K_M"): {
        "states": {"ratified": "damaged", "1": "damaged", "2": "damaged",
                   "3": "indeterminate", "4": "damaged"},
        # "Indeterminate under seed 3 (CI upper bound exactly 0)"
        "check": lambda rows: rows["3"].ci_hi == 0,
        "text": (
            "- `{key}`: Δ =\n"
            "  {delta}, Damaged under seeds {seed}/1/2/4, Indeterminate under seed 3 (CI\n"
            "  upper bound exactly 0). Arm-scoped; enters no ladder, cliff, or Holm\n"
            "  family; the trigger fires regardless."
        ),
    },
}


def _states_differ_trigger(result: AnalysisResult, cells: list[CellRow]) -> bool:
    """The §5 trigger's states-differ leg alone, re-evaluated on ``cells``
    over the computed (pairable) pairs of ``result`` -- a sufficient
    condition for FIRED that needs no pair bootstrap."""
    by_key = {(c.run_id, c.suite, c.quant_label): c for c in cells}
    for pr in result.shootout.pairs:
        if not pr.pairable:
            continue
        files = {name: (run, label) for name, run, label in shootout_files(pr.arm, pr.level)}
        (ra, la), (rb, lb) = files[pr.name_a], files[pr.name_b]
        if by_key[(ra, pr.suite, la)].state != by_key[(rb, pr.suite, lb)].state:
            return True
    return False


def _holm_summary(verdicts: dict) -> dict:
    return {k: (None if v is None else (v.cliff_rung, v.survives)) for k, v in verdicts.items()}


def _seed_sweep_lines(result: AnalysisResult, sweep: dict[str, list[CellRow]]) -> list[str]:
    """The "Seed robustness sweep" section (0B run set, ratified seed).
    Every claim is re-derived from the ratified cells + the sweep's cells."""
    seeds = sorted(sweep, key=int)
    ints = [int(s) for s in seeds]
    if not seeds or ints != list(range(ints[0], ints[0] + len(ints))):
        raise ValueError(f"seed sweep: expected contiguous seed prefixes, got {seeds}")
    ratified = {(c.run_id, c.suite, c.quant_label): c for c in result.cells}
    by_seed: dict[str, dict] = {}
    for s in seeds:
        by_seed[s] = {(c.run_id, c.suite, c.quant_label): c for c in sweep[s]}
        if set(by_seed[s]) != set(ratified):
            raise ValueError(f"seed sweep: seed {s} cell set differs from the ratified cells")

    # Invariance of cliffs, Holm verdicts and the trigger across all seeds.
    ratified_cliffs = {k: v.cliff_rung for k, v in result.cliffs.items()}
    ratified_holm = _holm_summary(result.holm_verdicts)
    if not result.shootout.triggered:
        raise ValueError("seed sweep: trigger invariance is only verifiable when it FIRED")
    for s in seeds:
        cells_s = sweep[s]
        cliffs_s = compute_cliffs(cells_s, ladder_run_ids=result.ladder_run_ids, suites=result.suites)
        if {k: v.cliff_rung for k, v in cliffs_s.items()} != ratified_cliffs:
            raise ValueError(f"seed sweep: cliff rungs differ under seed {s}")
        if _holm_summary(compute_holm_verdicts(cells_s, cliffs_s)) != ratified_holm:
            raise ValueError(f"seed sweep: Holm verdicts differ under seed {s}")
        if not _states_differ_trigger(result, cells_s):
            raise ValueError(f"seed sweep: trigger not re-established under seed {s}")

    # Seed-marginal cells: data-derived set must equal the annotated set.
    marginal = [
        k for k in ratified
        if len({ratified[k].state} | {by_seed[s][k].state for s in seeds}) > 1
    ]
    if set(marginal) != set(SEED_SWEEP_CELL_NOTES):
        raise ValueError(
            f"seed sweep: seed-marginal cells {sorted(marginal)} differ from the "
            f"annotated set {sorted(SEED_SWEEP_CELL_NOTES)}"
        )
    bullets: list[str] = []
    for key in marginal:
        note = SEED_SWEEP_CELL_NOTES[key]
        rows = {"ratified": ratified[key], **{s: by_seed[s][key] for s in seeds}}
        if {s: r.state for s, r in rows.items()} != note["states"] or not note["check"](rows):
            raise ValueError(f"seed sweep: seed-marginal cell {key} no longer matches its note")
        c = ratified[key]
        if key[0] in result.ladder_run_ids and "Arm-scoped" in note["text"]:
            raise ValueError(f"seed sweep: {key} is a ladder cell, note says arm-scoped")
        bullets.append(note["text"].format(
            key=" / ".join(key), delta=_u3(c.delta), lo=_u3(c.ci_lo), hi=_u3(c.ci_hi),
            seed=SEED, state=_STATE_DISPLAY[c.state], pp=f"{abs(c.delta) * 100:.1f}",
            n=c.n, margin_pp=f"{MARGIN * 100:g}",
        ))

    return [
        "## Seed robustness sweep (final data, 2026-09-05)",
        "",
        "<!-- Post-generation annotation, maintained by the sweep runner; re-append",
        "this section and Appendix A after any FINDINGS regeneration. Sweep",
        "outputs: analysis/0b/sweep/. -->",
        "",
        f"The ratified bootstrap seed is {SEED} (OPEN_QUESTIONS §7). The full",
        "analysis over the FINAL data (registered factual_qa from 0b2 + all other",
        f"suites from 0b) was re-run under seed prefixes {seeds[0]}–{seeds[-1]}. "
        f"**All {len(result.cliffs)} cliff rungs,",
        "all Holm headline verdicts, and the §5 shootout trigger (FIRED) are",
        f"identical across all {_COUNT_WORDS[len(seeds) + 1]} seeds.** "
        f"{_COUNT_WORDS[len(marginal)]} cells are seed-marginal, disclosed",
        "(the ratified seed's state governs, per the standing user ruling):",
        "",
        *"\n".join(bullets).split("\n"),
        "",
    ]


# Appendix A's cliff-comparison prose asserts, per ladder run, how far the
# registered-set cliff sits from the first-pass one in LADDER_ORDER
# ("Llama item-set-robust"; "Qwen's cliff moves one rung up").
APPENDIX_A_CLIFF_SHIFT = {"0b-llama-8b-ladder": 0, "0b-qwen-7b-ladder": -1}


def _cliff_holm_str(result: AnalysisResult, run_id: str) -> str:
    v = result.holm_verdicts.get((run_id, "factual_qa"))
    if v is None:
        return "no cliff"
    return f"cliff {v.cliff_rung} — Holm {'SURVIVES' if v.survives else 'FAILS'}"


def _appendix_a_lines(result: AnalysisResult, first_pass: AnalysisResult) -> list[str]:
    """Appendix A (0B run set, ratified seed): first-pass vs registered
    factual_qa ladder cells. Tables are built from the two analyses; the
    prose's claims (cliff shifts, F16 below the §7 band) are re-checked."""
    ladder = result.ladder_run_ids
    reg = {(c.run_id, c.quant_label): c for c in result.cells
           if c.suite == "factual_qa" and c.run_id in ladder}
    first = {(c.run_id, c.quant_label): c for c in first_pass.cells
             if c.suite == "factual_qa" and c.run_id in ladder}
    if set(reg) != set(first) or not reg:
        raise ValueError("Appendix A: first-pass and registered factual_qa ladder cells differ")
    for run_id in ladder:
        a = first_pass.cliffs.get((run_id, "factual_qa"))
        b = result.cliffs.get((run_id, "factual_qa"))
        if a is None or b is None or a.cliff_rung is None or b.cliff_rung is None or (
            LADDER_ORDER.index(b.cliff_rung) - LADDER_ORDER.index(a.cliff_rung)
            != APPENDIX_A_CLIFF_SHIFT.get(run_id)
        ):
            raise ValueError(f"Appendix A: cliff-comparison prose no longer holds for {run_id}")
    f16 = {run_id: reg[(run_id, LADDER_ORDER[0])].acc_f16 for run_id in ladder}
    if not all(acc < registered.BAND_LOW for acc in f16.values()):
        raise ValueError("Appendix A: an F16 baseline is no longer below the §7 band")
    substituted = {(c.run_id, c.quant_label) for c in result.cells
                   if c.suite == "factual_qa" and c.source_run_id != c.run_id}
    n_rungs = len(substituted) + len({run for run, _ in substituted})  # + each run's F16

    def short(run_id: str) -> str:
        return result.run_model[run_id].removesuffix("-instruct")

    lines = [
        "## Appendix A: first-pass factual_qa (non-registered item set) — disclosed sensitivity run",
        "",
        "<!-- Post-generation annotation, maintained with the sweep section; re-append",
        "after any FINDINGS regeneration. -->",
        "",
        "The first 0B pass measured factual_qa on a NON-registered item set",
        "(sha256 `ac5cb282…`) — the M3 weight vector applied in the wrong decile",
        "order (OPEN_QUESTIONS §8; mechanism and evidence:",
        "`analysis/0b/F16_CROSS_MACHINE.md`). Per the user ruling (option (a)),",
        f"factual_qa was regenerated for all {n_rungs} rungs on the REGISTERED set",
        "(`2e53ca0e…`, boot-time gate-enforced) in run 0b2 (2026-09-05), and the",
        "tables above use the registered measurements (see the `source_run_id`",
        "column). The first-pass numbers are retained here as a disclosed",
        "sensitivity run: a same-machinery measurement on a harder,",
        "popularity-shifted draw of the same PopQA source.",
        "",
        "**Cliff / Holm comparison across the two item sets:**",
        "",
        "| model | first pass (non-registered) | registered (0b2) |",
        "|---|---|---|",
    ]
    for run_id in ladder:
        lines.append(
            f"| {short(run_id)} | {_cliff_holm_str(first_pass, run_id)} "
            f"| {_cliff_holm_str(result, run_id)} |"
        )
    lines += [
        "",
        "The Llama verdict is item-set-robust. Qwen's cliff moves one rung up on",
        "the registered set (Q2_K's damage resolves clearly at the registered",
        "mix); direction and character of the curve are unchanged.",
        "",
        "**Per-cell states, ladder cells (first → registered):**",
        "",
        "| run | rung | first Δ | first state | registered Δ | registered state |",
        "|---|---|---|---|---|---|",
    ]
    for run_id in ladder:
        for rung in LADDER_ORDER:
            a, b = first[(run_id, rung)], reg[(run_id, rung)]
            lines.append(
                f"| {run_id} | {rung} | {a.delta:+.3f} | {a.state} "
                f"| {b.delta:+.3f} | {b.state} |"
            )
    baselines = ", ".join(f"{short(r)} F16 = {f16[r]:.3f}" for r in ladder)
    lines += [
        "",
        "**F16 baselines vs the PREREG §7 headroom band "
        f"[{registered.BAND_LOW}, {registered.BAND_HIGH}] on the",
        f"registered set:** {baselines} —",
        "both exactly reproduce Amendment 1 §C's calibration values (item-set",
        "identity + cross-machine grade agreement), and both remain below the",
        "band on the same side as at calibration. The registered M3",
        "out-of-band-low fallback disclosure (Amendment 1, per §7) is unchanged",
        "in kind and now unchanged in value.",
        "",
    ]
    return lines


def _sign_convention_sentence(result: AnalysisResult) -> Optional[str]:
    """OPEN_QUESTIONS §15(c): state the pair-delta sign convention and
    verify every computed row against the per-file accuracies (a row that
    contradicts it is a hard error). None when there are no pairs."""
    pairs = result.shootout.pairs
    if not pairs:
        return None
    by_key = {(c.run_id, c.suite, c.quant_label): c for c in result.cells}
    n_checked = 0
    for pr in pairs:
        if not pr.pairable:
            continue
        files = {name: (run, label) for name, run, label in shootout_files(pr.arm, pr.level)}
        (ra, la), (rb, lb) = files[pr.name_a], files[pr.name_b]
        expected = by_key[(rb, pr.suite, lb)].acc_quant - by_key[(ra, pr.suite, la)].acc_quant
        if abs(pr.delta - expected) > 1e-9:
            raise ValueError(
                f"shootout sign convention violated: {pr.arm} {pr.level} {pr.suite} "
                f"{pr.name_a} vs {pr.name_b}: delta {pr.delta} != acc(B)-acc(A) {expected}"
            )
        n_checked += 1
    if is_0b_run_set(result.run_ids, result.ladder_run_ids) and n_checked == len(pairs):
        return SIGN_CONVENTION_0B.format(n=n_checked)
    return SIGN_CONVENTION_GENERIC.format(n_checked=n_checked, n_rows=len(pairs))


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _fmt(x: float, digits: int = 6) -> str:
    return f"{x:.{digits}f}"


def write_cells_csv(
    cells: list[CellRow],
    out_path: Path,
    inversions: dict[tuple[str, str, str], str] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(CELLS_CSV_FIELDS)
        for c in cells:
            writer.writerow([
                c.run_id,
                c.source_run_id,
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
                str((c.run_id, c.suite, c.quant_label) in (inversions or {})),
                (inversions or {}).get((c.run_id, c.suite, c.quant_label), ""),
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


def write_findings_md(
    result: AnalysisResult,
    out_path: Path,
    *,
    sweep: Optional[dict[str, list[CellRow]]] = None,
    first_pass: Optional[AnalysisResult] = None,
) -> None:
    """Render FINDINGS markdown off ``result``. ``sweep`` (seed prefix ->
    cells, ``load_sweep``) and ``first_pass`` (``compute_first_pass_factual_qa``)
    feed the 0B-only seed-sweep section and Appendix A; both are emitted
    only for the 0B run set at the ratified seed (ignored otherwise)."""
    is_0b = is_0b_run_set(result.run_ids, result.ladder_run_ids)
    ratified = str(SEED) == registered.BOOTSTRAP_SEED
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
    if ratified:
        lines.append(f"## Seed disclosure (seed {SEED} — RATIFIED 2026-09-04, OPEN_QUESTIONS.md §7)")
    else:
        lines.append(
            f"## Seed disclosure (seed {SEED} — NOT the ratified seed "
            f"{registered.BOOTSTRAP_SEED}; robustness run, OPEN_QUESTIONS.md §7)"
        )
    lines.append("")
    lines.append(
        "PREREG §8 registers the per-cell inference machinery (McNemar exact; "
        "two-sided 95% CI on Δaccuracy via paired bootstrap, 10,000 resamples) "
        "but no RNG seed for the bootstrap — unlike every sampling seed in §3. "
        f"This run uses **seed {SEED}** (fresh; distinct from every registered "
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
    if ratified and is_0b:
        # 0B history: the seed was provisional when 0B first ran (2026-09-03).
        seed_status = (
            f"**Seed {SEED} was provisional when this analysis first ran "
            "(2026-09-03) and was RATIFIED by user ruling on 2026-09-04 "
            "(OPEN_QUESTIONS.md §7), together with the per-cell and per-pair "
            "rules above; these numbers stand. Any other seed would require a "
            "cheap, local, deterministic re-run of this script.**"
        )
    elif ratified:
        seed_status = (
            f"**Seed {SEED} is the bootstrap seed RATIFIED by user ruling on "
            "2026-09-04 (OPEN_QUESTIONS.md §7), together with the per-cell and "
            "per-pair rules above; these numbers stand.**"
        )
    else:
        seed_status = (
            f"**Seed {SEED} is NOT the ratified seed ({registered.BOOTSTRAP_SEED}, "
            "OPEN_QUESTIONS.md §7): this output is a seed-robustness run, and "
            "wherever a cell's state differs, the ratified seed's state governs.**"
        )
    lines.append(
        f"{seed_status} "
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
    for run_id in result.run_ids:
        run_cells = cells_by_run.get(run_id, [])
        if not run_cells:
            continue
        model = result.run_model[run_id]
        kind = "ladder" if run_id in result.ladder_run_ids else "arm"
        lines.append(f"### `{run_id}` ({model}, {kind})")
        lines.append("")
        if run_id in RUN_FOOTNOTES:
            lines.append(RUN_FOOTNOTES[run_id])
            lines.append("")
        cells_by_suite: dict[str, list[CellRow]] = {}
        for c in run_cells:
            cells_by_suite.setdefault(c.suite, []).append(c)
        for suite in result.suites:
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
    lines.append(
        "The column below is the STATE-sequence flag: True iff a Damaged "
        "rung sits above (higher precision than) a non-Damaged one, i.e. "
        "the Damaged cells do not form a contiguous bottom suffix. It reads "
        "False for every family in this data. PREREG §8's accuracy-level "
        "non-monotonicity (a lower-bits rung scoring above its higher-bits "
        "neighbor) is a separate, weaker anomaly, flagged per cell in "
        "cells.csv (`acc_inversion`/`inversion_above`) and listed in full "
        "in the next section -- earlier drafts' \"all 8 families "
        "monotonic\" statements referred only to the state-sequence "
        "definition."
    )
    lines.append("")
    lines.append("| run_id | suite | cliff rung | state-non-monotonic |")
    lines.append("|---|---|---|---|")
    for run_id in result.ladder_run_ids:
        for suite in result.suites:
            cr = result.cliffs.get((run_id, suite))
            if cr is None:
                continue
            rung_str = cr.cliff_rung if cr.cliff_rung is not None else "(none)"
            lines.append(f"| {run_id} | {suite} | {rung_str} | {cr.non_monotonic} |")
    lines.append("")

    # --- Accuracy-level non-monotonicity (PREREG §8 flag) ---
    lines.append("## Accuracy-level non-monotonicity (PREREG §8 flag)")
    lines.append("")
    lines.append(
        "PREREG §8: 'Non-monotonic rungs are flagged, never smoothed (with "
        "the IQ-vs-K ~2.5 bpw note where applicable).' Every adjacent-pair "
        "accuracy inversion -- a lower-bits rung scoring strictly above its "
        "higher-bits neighbor (F16 counts as the neighbor above the top "
        "rung; arm Q3_K_M files pair with their own uploader's Q4_K_M) -- "
        "is listed here and flagged per cell in cells.csv. Nothing is "
        "smoothed; the underlying accuracies stand unaltered in the tables "
        "above."
    )
    lines.append("")
    lines.append("| run_id | suite | flagged rung | scored above | acc (flagged) | acc (neighbor) | note |")
    lines.append("|---|---|---|---|---|---|---|")
    cell_by_key = {(c.run_id, c.suite, c.quant_label): c for c in result.cells}
    inv = result.inversions
    for (run_id, suite, label) in sorted(inv):
        hi = inv[(run_id, suite, label)]
        c = cell_by_key[(run_id, suite, label)]
        hi_acc = c.acc_f16 if hi == "F16" else cell_by_key[(run_id, suite, hi)].acc_quant
        note = ""
        if label == "IQ2_M" and hi == "Q2_K":
            note = (
                "IQ-vs-K ~2.5 bpw: the i-quant beats the k-quant at "
                "comparable bits -- the registered Q5 exploratory candidate "
                "pattern (PREREG §2/§5), here in confirmatory data"
            )
        lines.append(
            f"| {run_id} | {suite} | {label} | {hi} | {c.acc_quant:.4f} | {hi_acc:.4f} | {note} |"
        )
    n_ladder = sum(1 for k in inv if k[0] in result.ladder_run_ids)
    lines.append("")
    lines.append(
        f"Total: {len(inv)} inversions ({n_ladder} in ladder families, "
        f"{len(inv) - n_ladder} in arm families)."
    )
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
    for run_id in result.ladder_run_ids:
        for suite in result.suites:
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
    sign_sentence = _sign_convention_sentence(result)
    if sign_sentence is not None:
        lines.append(sign_sentence)
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

    # --- 0B-only annotations: seed sweep + Appendix A (see docstring) ---
    if is_0b and ratified and sweep is not None:
        lines += ["", "---", ""]
        lines += _seed_sweep_lines(result, sweep)
    if is_0b and ratified and first_pass is not None:
        lines += ["---", ""]
        lines += _appendix_a_lines(result, first_pass)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI (Task 7: takes run ids -- ladder runs, arm runs, and the factual_qa
# source substitution -- as arguments, with the 0B set as the documented
# default, so a future non-0B analysis (e.g. the 1.5B/0B-prime single-ladder-
# run, five-suite, no-arms shape) can reuse this script unchanged. With no
# arguments this is byte-identical to the pre-Task-7 hardcoded behavior.
# ---------------------------------------------------------------------------


def _parse_list(s: str) -> list[str]:
    """Comma-separated CLI list; "" -> [] (used to disable arm1/arm2 levels
    when no arm runs are given -- the shootout section then reports "not
    fired" instead of crashing)."""
    return [] if not s else s.split(",")


def _parse_kv(s: str) -> dict[str, str]:
    """Comma-separated ``k=v`` CLI mapping; "" -> {} (used to disable the
    factual_qa source substitution for runs that never sampled the
    wrong-order set, e.g. a 1.5B/0B-prime run)."""
    if not s:
        return {}
    out: dict[str, str] = {}
    for pair in s.split(","):
        k, _, v = pair.partition("=")
        out[k] = v
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=registered.BOOTSTRAP_SEED,
                        help="bootstrap seed prefix (default: ratified 8271)")
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default: analysis/0b)")
    parser.add_argument("--sweep-dir", default=None,
                        help="seed-sweep cells-seed*.csv directory feeding the "
                             "0B seed-sweep section (default: analysis/0b/sweep; "
                             "read only for the 0B run set at the ratified seed)")
    parser.add_argument("--run-ids", type=_parse_list, default=RUN_IDS,
                        help="comma-separated run ids to analyze "
                             "(default: the 0B set)")
    parser.add_argument("--ladder-run-ids", type=_parse_list, default=LADDER_RUN_IDS,
                        help="comma-separated subset of --run-ids treated as "
                             "ladder runs; every other run id is an 'arm' run "
                             "(default: the 0B ladder pair)")
    parser.add_argument("--suites", type=_parse_list, default=SUITES,
                        help="comma-separated suite keys to score "
                             "(default: the 0B four-suite set)")
    parser.add_argument("--ladder-order", type=_parse_list, default=LADDER_ORDER,
                        help="comma-separated ladder rung order "
                             "(default: the registered 7-rung order)")
    parser.add_argument("--arm1-levels", type=_parse_list, default=ARM1_LEVELS,
                        help="comma-separated Arm-1 shootout levels; pass "
                             "\"\" to disable Arm-1 (no arm runs given)")
    parser.add_argument("--arm2-levels", type=_parse_list, default=ARM2_LEVELS,
                        help="comma-separated Arm-2 shootout levels; pass "
                             "\"\" to disable Arm-2 (no arm runs given)")
    parser.add_argument("--run-model", type=_parse_kv, default=RUN_MODEL,
                        help="comma-separated run_id=model_id pairs "
                             "(default: the 0B mapping)")
    parser.add_argument("--factual-qa-source", type=_parse_kv, default=FACTUAL_QA_SOURCE,
                        help="comma-separated run_id=source_run_id factual_qa "
                             "substitution pairs (OPEN_QUESTIONS §8); pass "
                             "\"\" to disable the substitution")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    global SEED, SEED_RULE, PAIR_SEED_RULE
    SEED = args.seed
    SEED_RULE = seed_rule()
    PAIR_SEED_RULE = pair_seed_rule()

    pipeline_dir = Path(__file__).resolve().parents[1]
    runs_root = pipeline_dir / "runs-cloud" / "pipeline" / "runs"
    out_dir = Path(args.out_dir) if args.out_dir else pipeline_dir / "analysis" / "0b"

    result = run_analysis(
        runs_root,
        run_ids=args.run_ids,
        ladder_run_ids=args.ladder_run_ids,
        suites=args.suites,
        ladder_order=args.ladder_order,
        arm1_levels=args.arm1_levels,
        arm2_levels=args.arm2_levels,
        run_model=args.run_model,
        factual_qa_source=args.factual_qa_source,
    )
    # 0B-only annotation inputs (Task 8): gated exactly as write_findings_md
    # gates their sections, so a non-0B invocation never reads them.
    sweep = first_pass = None
    if is_0b_run_set(args.run_ids, args.ladder_run_ids) and str(SEED) == registered.BOOTSTRAP_SEED:
        sweep_dir = Path(args.sweep_dir) if args.sweep_dir else pipeline_dir / "analysis" / "0b" / "sweep"
        if sweep_dir.is_dir():
            sweep = load_sweep(sweep_dir) or None
        if "factual_qa" in args.suites and args.factual_qa_source:
            first_pass = compute_first_pass_factual_qa(
                runs_root, ladder_run_ids=args.ladder_run_ids,
                ladder_order=args.ladder_order, run_model=args.run_model,
            )
    write_cells_csv(result.cells, out_dir / "cells.csv", inversions=result.inversions)
    write_findings_md(result, out_dir / "FINDINGS_0B.md", sweep=sweep, first_pass=first_pass)
    print(f"wrote {len(result.cells)} cells to {out_dir / 'cells.csv'}")
    print(f"wrote findings to {out_dir / 'FINDINGS_0B.md'}")
    n_cliffs = sum(1 for cr in result.cliffs.values() if cr.cliff_rung is not None)
    print(f"cliffs found: {n_cliffs} / {len(result.cliffs)} (ladder run, suite) pairs")
    print(f"shootout trigger: {'FIRED' if result.shootout.triggered else 'not fired'}")


if __name__ == "__main__":
    main()
