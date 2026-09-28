"""Unit + integration tests for scripts/analyze_0b.py's driver logic:
pairing (incl. hard-error on id mismatch), the ladder-label-set gate,
cliff/Holm wiring, the §5 shootout trigger (incl. the cross-run "not
pairable" path), and determinism (run twice -> byte-identical outputs).

Loaded by file path (same pattern as tests/test_calibrate_f16.py): the
script has no package `__init__.py`.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "analyze_0b.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_0b", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # dataclasses with `from __future__ import annotations` resolve type
    # hints via sys.modules[cls.__module__] -- the module must be
    # registered there before exec_module runs the class bodies.
    sys.modules["analyze_0b"] = module
    spec.loader.exec_module(module)
    return module


a0b = _load_module()


# ---------------------------------------------------------------------------
# build_pairs
# ---------------------------------------------------------------------------


def test_build_pairs_matches_and_orders_by_item_id():
    map_a = {"b": (True, False), "a": (False, True)}
    map_b = {"a": (True, False), "b": (False, True)}
    pairs, trunc_a, trunc_b = a0b.build_pairs(map_a, map_b, context="ctx")
    # sorted by item id: "a" then "b"
    assert pairs == [(False, True), (True, False)]
    assert trunc_a == 1  # "a" truncated in map_a
    assert trunc_b == 1  # "b" truncated in map_b


def test_build_pairs_id_mismatch_raises():
    map_a = {"x": (True, False), "y": (True, False)}
    map_b = {"x": (True, False), "z": (True, False)}
    with pytest.raises(ValueError, match="item id mismatch"):
        a0b.build_pairs(map_a, map_b, context="ctx")


def test_build_pairs_empty_both_sides_ok():
    pairs, trunc_a, trunc_b = a0b.build_pairs({}, {}, context="ctx")
    assert pairs == []
    assert trunc_a == 0
    assert trunc_b == 0


# ---------------------------------------------------------------------------
# compute_cell_row (unit, no file I/O)
# ---------------------------------------------------------------------------


def _run_data(run_id, graded, item_ids=None):
    return a0b.RunData(run_id=run_id, graded=graded, item_ids=item_ids or {})


def test_compute_cell_row_equivalent_when_identical():
    suite = "arithmetic"
    ids = [f"i{n}" for n in range(10)]
    f16 = {i: (True, False) for i in ids}
    quant = {i: (True, False) for i in ids}
    rd = _run_data("run-x", {(suite, "F16"): f16, (suite, "Q4_K_M"): quant})
    row = a0b.compute_cell_row(rd, "some-model", suite, "Q4_K_M", margin=0.03, n_resamples=200)
    assert row.n == 10
    assert row.b_lost == 0 and row.c_gained == 0
    assert row.state == "equivalent"
    assert row.acc_f16 == 1.0 and row.acc_quant == 1.0
    assert row.delta == 0.0
    assert row.seed_rule == a0b.SEED_RULE


def test_compute_cell_row_damaged_when_all_discordant():
    suite = "arithmetic"
    ids = [f"i{n}" for n in range(20)]
    f16 = {i: (True, False) for i in ids}
    quant = {i: (False, False) for i in ids}
    rd = _run_data("run-x", {(suite, "F16"): f16, (suite, "Q2_K"): quant})
    row = a0b.compute_cell_row(rd, "some-model", suite, "Q2_K", margin=0.03, n_resamples=200)
    assert row.b_lost == 20 and row.c_gained == 0
    assert row.state == "damaged"
    assert row.delta == pytest.approx(-1.0)
    assert row.ci_lo == pytest.approx(-1.0) and row.ci_hi == pytest.approx(-1.0)


def test_compute_cell_row_tracks_truncation_independently_of_correctness():
    suite = "arithmetic"
    ids = [f"i{n}" for n in range(5)]
    f16 = {i: (True, i == "i0") for i in ids}
    quant = {i: (True, i == "i1" or i == "i2") for i in ids}
    rd = _run_data("run-x", {(suite, "F16"): f16, (suite, "Q4_K_M"): quant})
    row = a0b.compute_cell_row(rd, "m", suite, "Q4_K_M", margin=0.03, n_resamples=50)
    assert row.truncated_f16 == 1
    assert row.truncated_quant == 2


def test_compute_cell_row_seed_is_deterministic_per_run_quant_suite():
    suite = "arithmetic"
    ids = [f"i{n}" for n in range(20)]
    f16 = {i: (True, False) for i in ids}
    quant = {i: (i in ("i0", "i1"), False) for i in ids}
    rd = _run_data("same-run", {(suite, "F16"): f16, (suite, "Q4_K_M"): quant})
    row1 = a0b.compute_cell_row(rd, "m", suite, "Q4_K_M", margin=0.03, n_resamples=300)
    row2 = a0b.compute_cell_row(rd, "m", suite, "Q4_K_M", margin=0.03, n_resamples=300)
    assert row1 == row2  # frozen dataclass equality -> identical bootstrap draw


# ---------------------------------------------------------------------------
# compute_all_cells: ladder label-set gate
# ---------------------------------------------------------------------------


def test_compute_all_cells_raises_on_missing_ladder_rung():
    suite = "arithmetic"
    ids = [f"i{n}" for n in range(4)]
    f16 = {i: (True, False) for i in ids}
    # Only 6 of the 7 registered ladder rungs present (Q2_K missing).
    graded = {(suite, "F16"): f16}
    partial_ladder = [r for r in a0b.LADDER_ORDER if r != "Q2_K"]
    for rung in partial_ladder:
        graded[(suite, rung)] = {i: (True, False) for i in ids}
    runs_data = {"0b-llama-8b-ladder": _run_data("0b-llama-8b-ladder", graded)}
    with pytest.raises(ValueError, match="ladder label mismatch"):
        a0b.compute_all_cells(
            runs_data, margin=0.03, n_resamples=50, suites=[suite],
            run_ids=["0b-llama-8b-ladder"],
        )


def test_compute_all_cells_raises_on_extra_ladder_rung():
    suite = "arithmetic"
    ids = [f"i{n}" for n in range(4)]
    f16 = {i: (True, False) for i in ids}
    graded = {(suite, "F16"): f16}
    for rung in a0b.LADDER_ORDER:
        graded[(suite, rung)] = {i: (True, False) for i in ids}
    graded[(suite, "Q4_0")] = {i: (True, False) for i in ids}  # not a registered rung
    runs_data = {"0b-qwen-7b-ladder": _run_data("0b-qwen-7b-ladder", graded)}
    with pytest.raises(ValueError, match="ladder label mismatch"):
        a0b.compute_all_cells(
            runs_data, margin=0.03, n_resamples=50, suites=[suite],
            run_ids=["0b-qwen-7b-ladder"],
        )


# ---------------------------------------------------------------------------
# shootout_triggered: pure predicate, all four boolean branches
# ---------------------------------------------------------------------------


def _pair_result(pairable=True, excludes_zero=False, states_differ=False):
    return a0b.PairResult(
        arm="arm1", level="Q4_K_M", suite="arithmetic", name_a="a", name_b="b",
        pairable=pairable, reason=None if pairable else "not pairable (item sets differ)",
        delta=0.0, ci_lo=0.0, ci_hi=0.0,
        excludes_zero=excludes_zero if pairable else None,
        state_a="equivalent", state_b="equivalent",
        states_differ=states_differ if pairable else None,
    )


def test_shootout_triggered_false_when_nothing_fires():
    pairs = [_pair_result(), _pair_result()]
    assert a0b.shootout_triggered(pairs) is False


def test_shootout_triggered_true_on_ci_excludes_zero():
    pairs = [_pair_result(), _pair_result(excludes_zero=True)]
    assert a0b.shootout_triggered(pairs) is True


def test_shootout_triggered_true_on_states_differ():
    pairs = [_pair_result(), _pair_result(states_differ=True)]
    assert a0b.shootout_triggered(pairs) is True


def test_shootout_triggered_ignores_not_pairable_pairs():
    # A not-pairable entry must never itself count toward the trigger,
    # even though its excludes_zero/states_differ fields are None.
    pairs = [_pair_result(pairable=False)]
    assert a0b.shootout_triggered(pairs) is False


# ---------------------------------------------------------------------------
# Full synthetic runs tree: end-to-end driver wiring
# ---------------------------------------------------------------------------


def _write_run(run_dir: Path, item_ids_by_suite: dict, suite_label_states: dict, truncated=None):
    """suite_label_states: {suite: {quant_label: [state, ...]}} aligned to
    item_ids_by_suite[suite] order. truncated: optional
    {(suite, quant_label, item_id): True} sparse override (default False)."""
    truncated = truncated or {}
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "items.jsonl").open("w") as f:
        for suite, ids in item_ids_by_suite.items():
            for iid in ids:
                f.write(json.dumps({"id": iid, "suite": suite, "prompt": "p", "expected": ["x"]}) + "\n")
    with (run_dir / "grades.jsonl").open("w") as f:
        for suite, label_states in suite_label_states.items():
            ids = item_ids_by_suite[suite]
            for label, states in label_states.items():
                assert len(states) == len(ids), (run_dir, suite, label)
                for iid, state in zip(ids, states):
                    f.write(json.dumps({
                        "item_id": iid,
                        "suite": suite,
                        "quant_label": label,
                        "state": state,
                        "truncated": truncated.get((suite, label, iid), False),
                    }) + "\n")


SUITES = ["arithmetic", "longctx_retrieval"]
N = 20


@pytest.fixture
def synthetic_runs_root(tmp_path):
    root = tmp_path / "runs"

    # Shared item ids for the two "pairable" cross-run suites, and a
    # deliberately DIFFERENT id set for arm1's longctx_retrieval (the
    # "not pairable, item sets differ" path).
    shared_ids = {suite: [f"{suite}-{n:03d}" for n in range(N)] for suite in SUITES}
    arm1_ids = {
        "arithmetic": shared_ids["arithmetic"],  # matches llama ladder -> pairable
        "longctx_retrieval": [f"longctx_retrieval-alt-{n:03d}" for n in range(N)],  # differs -> NOT pairable
    }

    all_correct = {s: ["correct"] * N for s in SUITES}
    all_wrong = {s: ["wrong"] * N for s in SUITES}

    # --- 0b-llama-8b-ladder: F16 all-correct; top 4 rungs equivalent
    # (all-correct); bottom 3 rungs (Q3_K_M, Q2_K, IQ2_M) damaged
    # (all-wrong) -> clean monotone cliff at Q3_K_M.
    llama_labels = {"F16": all_correct}
    for rung in ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]:
        llama_labels[rung] = all_correct
    for rung in ["Q3_K_M", "Q2_K", "IQ2_M"]:
        llama_labels[rung] = all_wrong
    llama_suite_label_states = {
        s: {label: states[s] for label, states in llama_labels.items()} for s in SUITES
    }
    _write_run(
        root / "0b-llama-8b-ladder", shared_ids, llama_suite_label_states,
        truncated={("arithmetic", "Q3_K_M", "arithmetic-000"): True},
    )

    # --- 0b-qwen-7b-ladder: every rung all-correct -> no cliff anywhere.
    qwen_labels = {"F16": all_correct}
    for rung in a0b.LADDER_ORDER:
        qwen_labels[rung] = all_correct
    qwen_suite_label_states = {
        s: {label: states[s] for label, states in qwen_labels.items()} for s in SUITES
    }
    _write_run(root / "0b-qwen-7b-ladder", shared_ids, qwen_suite_label_states)

    # --- 0b-shootout-arm1: F16 all-correct; unsloth/mradermacher_static
    # all-correct at both levels; mradermacher_i1_Q4_K_M all-WRONG on
    # arithmetic only (the pairable suite) -> diverges sharply from
    # llama-ladder's bartowski Q4_K_M (all-correct), firing the trigger.
    arm1_labels = {
        "F16": all_correct,
        "unsloth_Q4_K_M": all_correct,
        "unsloth_Q3_K_M": all_correct,
        "mradermacher_static_Q4_K_M": all_correct,
        "mradermacher_static_Q3_K_M": all_correct,
        "mradermacher_i1_Q3_K_M": all_correct,
    }
    arm1_suite_label_states = {
        s: {label: states[s] for label, states in arm1_labels.items()} for s in SUITES
    }
    # mradermacher_i1_Q4_K_M: all-correct on longctx_retrieval, all-wrong on arithmetic
    arm1_suite_label_states["arithmetic"]["mradermacher_i1_Q4_K_M"] = ["wrong"] * N
    arm1_suite_label_states["longctx_retrieval"]["mradermacher_i1_Q4_K_M"] = ["correct"] * N
    _write_run(root / "0b-shootout-arm1", arm1_ids, arm1_suite_label_states)

    # --- 0b-arm2-official: F16 all-correct; official_Q4_K_M all-WRONG
    # (damaged vs its own F16, and diverges from qwen-ladder's bartowski
    # Q4_K_M which is all-correct/equivalent -> differing cell states);
    # official_Q3_K_M all-correct (equivalent, matches bartowski).
    arm2_labels = {
        "F16": all_correct,
        "Q4_K_M": all_wrong,
        "Q3_K_M": all_correct,
    }
    arm2_suite_label_states = {
        s: {label: states[s] for label, states in arm2_labels.items()} for s in SUITES
    }
    _write_run(root / "0b-arm2-official", shared_ids, arm2_suite_label_states)

    return root


def test_full_pipeline_cell_count(synthetic_runs_root):
    result = a0b.run_analysis(synthetic_runs_root, n_resamples=200, suites=SUITES)
    # llama ladder: 7 rungs * 2 suites = 14; qwen ladder: 14;
    # arm1: 6 labels * 2 suites = 12; arm2: 2 labels * 2 suites = 4.
    assert len(result.cells) == 14 + 14 + 12 + 4


def test_full_pipeline_cliff_llama_monotone(synthetic_runs_root):
    result = a0b.run_analysis(synthetic_runs_root, n_resamples=200, suites=SUITES)
    for suite in SUITES:
        cr = result.cliffs[("0b-llama-8b-ladder", suite)]
        assert cr.cliff_rung == "Q3_K_M"
        assert cr.non_monotonic is False


def test_full_pipeline_cliff_qwen_none(synthetic_runs_root):
    result = a0b.run_analysis(synthetic_runs_root, n_resamples=200, suites=SUITES)
    for suite in SUITES:
        cr = result.cliffs[("0b-qwen-7b-ladder", suite)]
        assert cr.cliff_rung is None
        assert cr.non_monotonic is False


def test_full_pipeline_holm_survives_for_llama(synthetic_runs_root):
    result = a0b.run_analysis(synthetic_runs_root, n_resamples=200, suites=SUITES)
    for suite in SUITES:
        verdict = result.holm_verdicts[("0b-llama-8b-ladder", suite)]
        assert verdict is not None
        assert verdict.cliff_rung == "Q3_K_M"
        assert verdict.rungs_at_or_below == ["Q3_K_M", "Q2_K", "IQ2_M"]
        # n=20, fully discordant (b=20,c=0) at each damaged rung ->
        # p = 2*0.5**20, comfortably beats every Holm threshold over m=7.
        assert verdict.survives is True
        assert verdict.failing_rungs == []


def test_full_pipeline_holm_none_for_qwen_no_cliff(synthetic_runs_root):
    result = a0b.run_analysis(synthetic_runs_root, n_resamples=200, suites=SUITES)
    for suite in SUITES:
        assert result.holm_verdicts[("0b-qwen-7b-ladder", suite)] is None


def test_full_pipeline_truncation_counted(synthetic_runs_root):
    result = a0b.run_analysis(synthetic_runs_root, n_resamples=50, suites=SUITES)
    row = next(
        c for c in result.cells
        if c.run_id == "0b-llama-8b-ladder" and c.suite == "arithmetic" and c.quant_label == "Q3_K_M"
    )
    assert row.truncated_f16 == 0
    assert row.truncated_quant == 1


def test_full_pipeline_shootout_not_pairable_for_cross_run_longctx(synthetic_runs_root):
    result = a0b.run_analysis(synthetic_runs_root, n_resamples=50, suites=SUITES)
    cross_run_pairs = [
        pr for pr in result.shootout.pairs
        if pr.suite == "longctx_retrieval"
        and pr.level == "Q4_K_M"
        and pr.arm == "arm1"
        and "bartowski_Q4_K_M" in (pr.name_a, pr.name_b)
    ]
    assert cross_run_pairs, "expected bartowski-involving arm1 Q4_K_M longctx pairs"
    for pr in cross_run_pairs:
        assert pr.pairable is False
        assert pr.reason == "not pairable (item sets differ)"
        assert pr.delta is None and pr.excludes_zero is None and pr.states_differ is None


def test_full_pipeline_shootout_same_run_pairs_still_pairable_when_cross_run_isnt(synthetic_runs_root):
    result = a0b.run_analysis(synthetic_runs_root, n_resamples=50, suites=SUITES)
    # unsloth vs mradermacher_static at Q4_K_M/longctx: both live in
    # 0b-shootout-arm1 -> pairable even though arm1's longctx item set
    # differs from the ladder run's.
    same_run_pair = next(
        pr for pr in result.shootout.pairs
        if pr.suite == "longctx_retrieval" and pr.level == "Q4_K_M" and pr.arm == "arm1"
        and {pr.name_a, pr.name_b} == {"unsloth_Q4_K_M", "mradermacher_static_Q4_K_M"}
    )
    assert same_run_pair.pairable is True


def test_full_pipeline_shootout_triggers(synthetic_runs_root):
    result = a0b.run_analysis(synthetic_runs_root, n_resamples=50, suites=SUITES)
    assert result.shootout.triggered is True
    # Confirm it's the expected pair carrying the signal: bartowski_Q4_K_M
    # (all-correct) vs mradermacher_i1_Q4_K_M (all-wrong) on arithmetic.
    firing_pair = next(
        pr for pr in result.shootout.pairs
        if pr.suite == "arithmetic" and pr.level == "Q4_K_M"
        and {pr.name_a, pr.name_b} == {"bartowski_Q4_K_M", "mradermacher_i1_Q4_K_M"}
    )
    assert firing_pair.pairable is True
    assert firing_pair.excludes_zero is True


def test_full_pipeline_shootout_arm2_states_differ(synthetic_runs_root):
    result = a0b.run_analysis(synthetic_runs_root, n_resamples=50, suites=SUITES)
    pair = next(
        pr for pr in result.shootout.pairs
        if pr.arm == "arm2" and pr.level == "Q4_K_M" and pr.suite == "arithmetic"
    )
    assert pair.pairable is True
    assert pair.state_a != pair.state_b
    assert pair.states_differ is True


# ---------------------------------------------------------------------------
# Determinism: running twice must produce byte-identical outputs
# ---------------------------------------------------------------------------


def test_determinism_cells_csv_and_findings_md_byte_identical(synthetic_runs_root, tmp_path):
    result_1 = a0b.run_analysis(synthetic_runs_root, n_resamples=100, suites=SUITES)
    result_2 = a0b.run_analysis(synthetic_runs_root, n_resamples=100, suites=SUITES)

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    a0b.write_cells_csv(result_1.cells, out1 / "cells.csv")
    a0b.write_findings_md(result_1, out1 / "FINDINGS_0B.md")
    a0b.write_cells_csv(result_2.cells, out2 / "cells.csv")
    a0b.write_findings_md(result_2, out2 / "FINDINGS_0B.md")

    assert (out1 / "cells.csv").read_bytes() == (out2 / "cells.csv").read_bytes()
    assert (out1 / "FINDINGS_0B.md").read_bytes() == (out2 / "FINDINGS_0B.md").read_bytes()


def test_cells_csv_header_matches_spec(synthetic_runs_root, tmp_path):
    result = a0b.run_analysis(synthetic_runs_root, n_resamples=50, suites=SUITES)
    out_path = tmp_path / "cells.csv"
    a0b.write_cells_csv(result.cells, out_path)
    header = out_path.read_text().splitlines()[0]
    assert header == ",".join(a0b.CELLS_CSV_FIELDS)


def test_findings_md_contains_seed_disclosure_and_trigger_verdict(synthetic_runs_root, tmp_path):
    result = a0b.run_analysis(synthetic_runs_root, n_resamples=50, suites=SUITES)
    out_path = tmp_path / "FINDINGS_0B.md"
    a0b.write_findings_md(result, out_path)
    text = out_path.read_text()
    assert "8271" in text
    assert "RATIFIED" in text  # Task 8: seed ratified 2026-09-04 (OPEN_QUESTIONS §7)
    assert "PROVISIONAL" not in text
    assert "OPEN_QUESTIONS.md" in text
    assert "runs-cloud/fingerprint.txt" in text
    assert "Trigger: FIRED" in text  # this synthetic tree is built to fire it
    assert "zero-discordance" in text


def test_factual_qa_cells_sourced_from_0b2_runs(tmp_path):
    """OPEN_QUESTIONS §8 resolution: when a 0b run carries factual_qa, the
    analyzer must load those grades/item-ids from the paired 0b2 rerun
    (registered item set) and stamp source_run_id accordingly; missing 0b2
    source -> hard error, never silent fallback to the wrong-set grades."""
    import pytest as _pytest

    root = tmp_path / "runs"
    n = 4
    ids_0b = {"factual_qa": [f"fq-old-{i}" for i in range(n)]}
    ids_0b2 = {"factual_qa": [f"fq-new-{i}" for i in range(n)]}
    states_0b = {"factual_qa": {"F16": ["correct"] * n, "Q8_0": ["correct"] * n}}
    states_0b2 = {"factual_qa": {"F16": ["correct"] * n, "Q8_0": ["wrong"] * n}}
    _write_run(root / "0b-llama-8b-ladder", ids_0b, states_0b)

    with _pytest.raises(FileNotFoundError, match="0b2-llama-8b-ladder"):
        a0b.load_all_runs(root, run_ids=["0b-llama-8b-ladder"])

    _write_run(root / "0b2-llama-8b-ladder", ids_0b2, states_0b2)
    runs = a0b.load_all_runs(root, run_ids=["0b-llama-8b-ladder"])
    run = runs["0b-llama-8b-ladder"]
    assert run.suite_source["factual_qa"] == "0b2-llama-8b-ladder"
    assert run.item_ids["factual_qa"] == set(ids_0b2["factual_qa"])
    assert all(not ok for ok, _ in run.graded[("factual_qa", "Q8_0")].values())

    cell = a0b.compute_cell_row(run, "m", "factual_qa", "Q8_0", margin=0.03, n_resamples=50)
    assert cell.source_run_id == "0b2-llama-8b-ladder"
    assert cell.run_id == "0b-llama-8b-ladder"


# ---------------------------------------------------------------------------
# Accuracy-level non-monotonicity flag (PREREG §8 accuracy reading, user
# ruling 2026-09-06 / OPEN_QUESTIONS §9): a cell is flagged when its rung
# scores ABOVE its higher-bits neighbor (F16 for the top rung; the previous
# registered-ladder rung otherwise; for arm files, the same-uploader Q4_K_M
# is Q3_K_M's neighbor and F16 is Q4_K_M's).
# ---------------------------------------------------------------------------


def _mk_cell(run_id, suite, quant_label, acc_f16, acc_quant):
    return a0b.CellRow(
        run_id=run_id, source_run_id=run_id, model="m", suite=suite,
        quant_label=quant_label, n=10, acc_f16=acc_f16, acc_quant=acc_quant,
        delta=acc_quant - acc_f16, ci_lo=-1, ci_hi=1, p_mcnemar=1.0,
        b_lost=0, c_gained=0, truncated_f16=0, truncated_quant=0,
        state="indeterminate", seed_rule="t",
    )


def test_accuracy_inversions_ladder_chain_including_f16_top():
    cells = [
        _mk_cell("0b-llama-8b-ladder", "arithmetic", "Q8_0", 0.80, 0.85),   # > F16 -> flag
        _mk_cell("0b-llama-8b-ladder", "arithmetic", "Q6_K", 0.80, 0.84),   # < Q8_0 -> no
        _mk_cell("0b-llama-8b-ladder", "arithmetic", "Q5_K_M", 0.80, 0.86), # > Q6_K -> flag
        _mk_cell("0b-llama-8b-ladder", "arithmetic", "Q4_K_M", 0.80, 0.86), # == Q5_K_M -> no (strict >)
    ]
    flags = a0b.accuracy_inversions(cells)
    assert flags[("0b-llama-8b-ladder", "arithmetic", "Q8_0")] == "F16"
    assert ("0b-llama-8b-ladder", "arithmetic", "Q6_K") not in flags
    assert flags[("0b-llama-8b-ladder", "arithmetic", "Q5_K_M")] == "Q6_K"
    assert ("0b-llama-8b-ladder", "arithmetic", "Q4_K_M") not in flags


def test_accuracy_inversions_arm_pairs_same_uploader_only():
    cells = [
        _mk_cell("0b-shootout-arm1", "arithmetic", "unsloth_Q4_K_M", 0.90, 0.80),
        _mk_cell("0b-shootout-arm1", "arithmetic", "unsloth_Q3_K_M", 0.90, 0.85),          # > own Q4 -> flag
        _mk_cell("0b-shootout-arm1", "arithmetic", "mradermacher_static_Q4_K_M", 0.90, 0.95),  # > F16 -> flag
        _mk_cell("0b-shootout-arm1", "arithmetic", "mradermacher_static_Q3_K_M", 0.90, 0.94),  # < own Q4 -> no
        _mk_cell("0b-arm2-official", "arithmetic", "Q4_K_M", 0.90, 0.80),
        _mk_cell("0b-arm2-official", "arithmetic", "Q3_K_M", 0.90, 0.81),                  # > own Q4 -> flag
    ]
    flags = a0b.accuracy_inversions(cells)
    assert flags[("0b-shootout-arm1", "arithmetic", "unsloth_Q3_K_M")] == "unsloth_Q4_K_M"
    assert flags[("0b-shootout-arm1", "arithmetic", "mradermacher_static_Q4_K_M")] == "F16"
    assert ("0b-shootout-arm1", "arithmetic", "mradermacher_static_Q3_K_M") not in flags
    assert flags[("0b-arm2-official", "arithmetic", "Q3_K_M")] == "Q4_K_M"


def test_cells_csv_carries_acc_inversion_columns(tmp_path):
    cells = [
        _mk_cell("0b-llama-8b-ladder", "arithmetic", "Q8_0", 0.80, 0.85),
        _mk_cell("0b-llama-8b-ladder", "arithmetic", "Q6_K", 0.80, 0.70),
    ]
    out = tmp_path / "cells.csv"
    a0b.write_cells_csv(cells, out, inversions=a0b.accuracy_inversions(cells))
    import csv as _csv
    rows = list(_csv.DictReader(out.open()))
    assert rows[0]["acc_inversion"] == "True" and rows[0]["inversion_above"] == "F16"
    assert rows[1]["acc_inversion"] == "False" and rows[1]["inversion_above"] == ""


# ---------------------------------------------------------------------------
# Task 7: analyze_0b.py takes run ids -- CLI arg parsing + defaults, and the
# generalized (run_ids/ladder_run_ids/suites/run_model/factual_qa_source)
# plumbing that lets a non-0B run (e.g. the 1.5B single-ladder-run, five
# -suite, no-arms 0B-prime analysis) reuse this script without editing it.
# ---------------------------------------------------------------------------


def test_parse_list_splits_on_comma():
    assert a0b._parse_list("a,b,c") == ["a", "b", "c"]


def test_parse_list_empty_string_is_empty_list():
    assert a0b._parse_list("") == []


def test_parse_kv_splits_pairs():
    assert a0b._parse_kv("x=1,y=2") == {"x": "1", "y": "2"}


def test_parse_kv_empty_string_is_empty_dict():
    assert a0b._parse_kv("") == {}


def test_build_arg_parser_defaults_are_the_0b_set():
    parser = a0b.build_arg_parser()
    args = parser.parse_args([])
    assert args.run_ids == a0b.RUN_IDS
    assert args.ladder_run_ids == a0b.LADDER_RUN_IDS
    assert args.suites == a0b.SUITES
    assert args.ladder_order == a0b.LADDER_ORDER
    assert args.arm1_levels == a0b.ARM1_LEVELS
    assert args.arm2_levels == a0b.ARM2_LEVELS
    assert args.run_model == a0b.RUN_MODEL
    assert args.factual_qa_source == a0b.FACTUAL_QA_SOURCE
    assert args.seed == a0b.registered.BOOTSTRAP_SEED
    assert args.out_dir is None


def test_build_arg_parser_overrides_for_a_single_ladder_run_no_arms():
    """The 1.5B (0B-prime) shape: one ladder run, five suites, no arm runs,
    no factual_qa substitution -- exactly what a future 1.5B invocation
    would pass, without editing this script."""
    parser = a0b.build_arg_parser()
    args = parser.parse_args([
        "--run-ids", "0b-prime-qwen-1.5b-ladder",
        "--ladder-run-ids", "0b-prime-qwen-1.5b-ladder",
        "--suites", "longctx_retrieval_2a,longctx_retrieval,arithmetic,arithmetic_twins,factual_qa",
        "--run-model", "0b-prime-qwen-1.5b-ladder=qwen2.5-1.5b-instruct",
        "--factual-qa-source", "",
        "--arm1-levels", "",
        "--arm2-levels", "",
    ])
    assert args.run_ids == ["0b-prime-qwen-1.5b-ladder"]
    assert args.ladder_run_ids == ["0b-prime-qwen-1.5b-ladder"]
    assert args.suites == [
        "longctx_retrieval_2a", "longctx_retrieval", "arithmetic", "arithmetic_twins", "factual_qa",
    ]
    assert args.run_model == {"0b-prime-qwen-1.5b-ladder": "qwen2.5-1.5b-instruct"}
    assert args.factual_qa_source == {}
    assert args.arm1_levels == []
    assert args.arm2_levels == []


def test_run_analysis_single_ladder_run_no_arms_does_not_crash(tmp_path):
    """Shootout/trigger logic only applies when arm runs are given; with
    none (arm1_levels=[] and arm2_levels=[]), the shootout section must be
    empty and the §5 trigger must read "not applicable" (not "not fired",
    which would imply a comparison was made), never crash."""
    root = tmp_path / "runs"
    suite = "arithmetic"
    ids = [f"i{n}" for n in range(10)]
    labels = {"F16": ["correct"] * 10}
    for rung in a0b.LADDER_ORDER:
        labels[rung] = ["correct"] * 10
    _write_run(root / "1p5b-ladder", {suite: ids}, {suite: labels})
    (root / "1p5b-ladder" / "fingerprint.txt").write_text("test fixture fingerprint\n")

    result = a0b.run_analysis(
        root,
        n_resamples=50,
        run_ids=["1p5b-ladder"],
        ladder_run_ids=["1p5b-ladder"],
        suites=[suite],
        arm1_levels=[],
        arm2_levels=[],
        run_model={"1p5b-ladder": "qwen2.5-1.5b-instruct"},
        factual_qa_source={},
    )
    assert result.shootout.pairs == []
    assert result.shootout.triggered is False
    assert len(result.cells) == len(a0b.LADDER_ORDER)  # one suite, one ladder run

    out_path = tmp_path / "FINDINGS_0B.md"
    a0b.write_findings_md(result, out_path, runs_root=root, pipeline_dir=tmp_path)  # must not raise
    text = out_path.read_text()
    assert "### `1p5b-ladder` (qwen2.5-1.5b-instruct, ladder)" in text
    assert "**Trigger: not applicable.**" in text
    assert "Trigger: not fired" not in text


def test_write_findings_md_uses_result_run_model_not_module_global(tmp_path):
    """Regression: write_findings_md must read model/kind/suites off the
    AnalysisResult it's given, not off the module-level RUN_MODEL/RUN_KIND/
    SUITES globals -- otherwise a non-0B run_id crashes with a KeyError."""
    root = tmp_path / "runs"
    suite = "custom_suite"
    ids = [f"i{n}" for n in range(6)]
    labels = {"F16": ["correct"] * 6}
    for rung in a0b.LADDER_ORDER:
        labels[rung] = ["correct"] * 6
    custom_run_id = "not-a-0b-run-id"
    _write_run(root / custom_run_id, {suite: ids}, {suite: labels})
    (root / custom_run_id / "fingerprint.txt").write_text("test fixture fingerprint\n")

    result = a0b.run_analysis(
        root,
        n_resamples=50,
        run_ids=[custom_run_id],
        ladder_run_ids=[custom_run_id],
        suites=[suite],
        arm1_levels=[],
        arm2_levels=[],
        run_model={custom_run_id: "some-custom-model"},
        factual_qa_source={},
    )
    out_path = tmp_path / "FINDINGS_0B.md"
    a0b.write_findings_md(result, out_path, runs_root=root, pipeline_dir=tmp_path)
    text = out_path.read_text()
    assert f"### `{custom_run_id}` (some-custom-model, ladder)" in text
    assert "**custom_suite**" in text


# ---------------------------------------------------------------------------
# Task 8: the generator emits every post-generation annotation that used to
# be hand-added to analysis/0b/FINDINGS_0B.md (ratified seed wording, Arm 2
# imatrix footnote, pair-delta sign convention, seed-robustness sweep,
# Appendix A), each gated on the inputs that make it applicable, plus the
# "PREREG SS2/SS5" -> "PREREG §2/§5" encoding fix.
# ---------------------------------------------------------------------------

import dataclasses as _dc
import itertools as _it

PIPELINE_DIR = Path(__file__).resolve().parent.parent
COMMITTED_0B = PIPELINE_DIR / "analysis" / "0b"
REAL_RUNS_ROOT = PIPELINE_DIR / "runs-cloud" / "pipeline" / "runs"


def _findings(result, tmp_path, **kw):
    out = tmp_path / "F.md"
    if "runs_root" not in kw and not a0b.is_0b_run_set(result.run_ids, result.ladder_run_ids):
        # Test convenience: every non-0B result-builder below writes its run
        # dirs (including each run's fingerprint.txt) under one of these
        # fixed tmp_path subdirs; default to whichever exists so call sites
        # don't need to repeat it.
        for candidate in (tmp_path / "runs1", tmp_path / "prime-runs"):
            if candidate.is_dir():
                kw["runs_root"] = candidate
                break
    if not a0b.is_0b_run_set(result.run_ids, result.ladder_run_ids):
        kw.setdefault("pipeline_dir", tmp_path)
    a0b.write_findings_md(result, out, **kw)
    return out.read_text()


def _single_run_result(tmp_path, run_id="1p5b-ladder"):
    root = tmp_path / "runs1"
    ids = [f"i{n}" for n in range(10)]
    labels = {"F16": ["correct"] * 10}
    for rung in a0b.LADDER_ORDER:
        labels[rung] = ["correct"] * 10
    _write_run(root / run_id, {"arithmetic": ids}, {"arithmetic": labels})
    (root / run_id / "fingerprint.txt").write_text("test fixture fingerprint\n")
    return a0b.run_analysis(
        root, n_resamples=50, run_ids=[run_id], ladder_run_ids=[run_id],
        suites=["arithmetic"], arm1_levels=[], arm2_levels=[],
        run_model={run_id: "qwen2.5-1.5b-instruct"}, factual_qa_source={},
    )


def test_seed_disclosure_ratified_wording_for_0b_set(synthetic_runs_root, tmp_path):
    text = _findings(a0b.run_analysis(synthetic_runs_root, n_resamples=50, suites=SUITES), tmp_path)
    assert "## Seed disclosure (seed 8271 — RATIFIED 2026-09-04, OPEN_QUESTIONS.md §7)\n" in text
    assert (
        "**Seed 8271 was provisional when this analysis first ran (2026-09-03) and was "
        "RATIFIED by user ruling on 2026-09-04 (OPEN_QUESTIONS.md §7), together with the "
        "per-cell and per-pair rules above; these numbers stand. Any other seed would "
        "require a cheap, local, deterministic re-run of this script.** Registered values "
        "used: margin M = 0.03, α = 0.05, n_resamples = 10000"
    ) in text


def test_seed_disclosure_non_0b_run_set_omits_0b_history(tmp_path):
    text = _findings(_single_run_result(tmp_path), tmp_path)
    assert "RATIFIED 2026-09-04" in text
    assert "first ran (2026-09-03)" not in text


def test_seed_disclosure_non_ratified_seed_says_so(synthetic_runs_root, tmp_path, monkeypatch):
    monkeypatch.setattr(a0b, "SEED", "3")
    text = _findings(a0b.run_analysis(synthetic_runs_root, n_resamples=50, suites=SUITES), tmp_path)
    assert "NOT the ratified seed" in text
    assert "RATIFIED 2026-09-04, OPEN_QUESTIONS.md §7)" not in text


ARM2_FOOTNOTE = (
    "Footnote (2026-09-08, OPEN_QUESTIONS §10): the two official Qwen files ran and are "
    "reported as `imatrix: false`; confirmed from their GGUF headers at the pinned revision "
    "(no `quantize.imatrix.*` keys) — evidence: "
    "`reference-manifests/evidence/qwen2.5-7b-official-gguf-headers.md`."
)


def test_arm2_imatrix_footnote_directly_under_arm2_heading(synthetic_runs_root, tmp_path):
    text = _findings(a0b.run_analysis(synthetic_runs_root, n_resamples=50, suites=SUITES), tmp_path)
    assert (
        "### `0b-arm2-official` (qwen2.5-7b-instruct, arm)\n\n" + ARM2_FOOTNOTE + "\n\n**"
    ) in text
    assert text.count("Footnote (2026-09-08") == 1


def test_arm2_imatrix_footnote_absent_without_arm2_run(tmp_path):
    assert "Footnote (2026-09-08" not in _findings(_single_run_result(tmp_path), tmp_path)


SIGN_0B = (
    'Sign convention (OPEN_QUESTIONS §15(c); emitted by `scripts/analyze_0b.py` '
    'since 2026-09-28): for a row "A vs B", delta = acc(B) minus acc(A). All {n} '
    "rows below were checked against the per-file accuracies in `cells.csv`, and "
    "none contradicts it."
)


def _pairable_only(result):
    pairs = [p for p in result.shootout.pairs if p.pairable]
    return _dc.replace(result, shootout=a0b.ShootoutReport(pairs=pairs, triggered=result.shootout.triggered))


def test_sign_convention_0b_sentence_when_every_row_checked(synthetic_runs_root, tmp_path):
    result = _pairable_only(a0b.run_analysis(synthetic_runs_root, n_resamples=50, suites=SUITES))
    text = _findings(result, tmp_path)
    n = len(result.shootout.pairs)
    assert "is reported not pairable and excluded below.\n\n" + SIGN_0B.format(n=n) + "\n\n| arm |" in text


def test_sign_convention_generic_when_some_rows_not_pairable(synthetic_runs_root, tmp_path):
    result = a0b.run_analysis(synthetic_runs_root, n_resamples=50, suites=SUITES)
    text = _findings(result, tmp_path)
    assert 'Sign convention (OPEN_QUESTIONS §15(c)): for a row "A vs B", delta = acc(B) minus acc(A).' in text
    assert "added 2026-09-27" not in text


def test_sign_convention_contradiction_raises(synthetic_runs_root, tmp_path):
    result = _pairable_only(a0b.run_analysis(synthetic_runs_root, n_resamples=50, suites=SUITES))
    pairs = list(result.shootout.pairs)
    bad = next(i for i, p in enumerate(pairs) if p.delta)
    pairs[bad] = _dc.replace(pairs[bad], delta=-pairs[bad].delta)
    result = _dc.replace(result, shootout=a0b.ShootoutReport(pairs=pairs, triggered=True))
    with pytest.raises(ValueError, match="sign convention"):
        _findings(result, tmp_path)


def test_sign_convention_absent_without_shootout_pairs(tmp_path):
    assert "Sign convention" not in _findings(_single_run_result(tmp_path), tmp_path)


def _committed_result():
    """AnalysisResult rebuilt from the committed cells.csv (ratified seed)
    -- no bootstrap, so it is fast. Shootout pairs are rebuilt with
    pairable=True (all 56 are pairable in 0B) and delta = acc(B) - acc(A);
    enough to exercise every annotation that reads them."""
    cells = a0b.read_cells_csv(COMMITTED_0B / "cells.csv")
    cliffs = a0b.compute_cliffs(cells)
    by_key = {(c.run_id, c.suite, c.quant_label): c for c in cells}
    pairs = []
    for arm, levels in (("arm1", a0b.ARM1_LEVELS), ("arm2", a0b.ARM2_LEVELS)):
        for level in levels:
            files = a0b.shootout_files(arm, level)
            for suite in a0b.SUITES:
                for (na, ra, la), (nb, rb, lb) in _it.combinations(files, 2):
                    ca, cb = by_key[(ra, suite, la)], by_key[(rb, suite, lb)]
                    pairs.append(a0b.PairResult(
                        arm=arm, level=level, suite=suite, name_a=na, name_b=nb,
                        pairable=True, reason=None, delta=cb.acc_quant - ca.acc_quant,
                        ci_lo=0.0, ci_hi=0.0, excludes_zero=False,
                        state_a=ca.state, state_b=cb.state, states_differ=ca.state != cb.state,
                    ))
    return a0b.AnalysisResult(
        cells=cells, cliffs=cliffs, holm_verdicts=a0b.compute_holm_verdicts(cells, cliffs),
        shootout=a0b.ShootoutReport(pairs=pairs, triggered=a0b.shootout_triggered(pairs)),
        inversions=a0b.accuracy_inversions(cells),
    )


def _committed_section(start: str, end: str | None) -> str:
    text = (COMMITTED_0B / "FINDINGS_0B.md").read_text()
    i = text.index(start)
    return text[i:] if end is None else text[i:text.index(end, i)]


def test_iq_vs_k_note_uses_section_sign_encoding(tmp_path):
    text = _findings(_committed_result(), tmp_path)
    assert "pattern (PREREG §2/§5), here in confirmatory data |" in text
    assert "SS2/SS5" not in text


def test_real_0b_sign_convention_sentence_is_the_committed_one(tmp_path):
    text = _findings(_committed_result(), tmp_path)
    assert SIGN_0B.format(n=56) in text


def test_real_0b_seed_sweep_section_matches_committed_text(tmp_path):
    sweep = a0b.load_sweep(COMMITTED_0B / "sweep")
    assert sorted(sweep) == ["1", "2", "3", "4"]
    text = _findings(_committed_result(), tmp_path, sweep=sweep)
    section = _committed_section("\n\n---\n\n## Seed robustness sweep", "---\n\n## Appendix A")
    assert "Two cells are seed-marginal" in section
    assert section in text


def test_seed_sweep_raises_when_marginal_cells_change(tmp_path):
    sweep = a0b.load_sweep(COMMITTED_0B / "sweep")
    rows = list(sweep["2"])
    i = next(k for k, c in enumerate(rows) if (c.run_id, c.suite, c.quant_label)
             == ("0b-arm2-official", "arithmetic", "Q4_K_M"))
    # An arm cell: enters no cliff/Holm family, so only the marginal set moves.
    rows[i] = _dc.replace(rows[i], state="small_real_loss" if rows[i].state != "small_real_loss" else "equivalent")
    sweep["2"] = rows
    with pytest.raises(ValueError, match="seed-marginal"):
        _findings(_committed_result(), tmp_path, sweep=sweep)


def test_seed_sweep_and_appendix_gated_off_for_non_0b_or_non_ratified(tmp_path, monkeypatch):
    sweep = a0b.load_sweep(COMMITTED_0B / "sweep")
    single = _single_run_result(tmp_path)
    text = _findings(single, tmp_path, sweep=sweep, first_pass=single)
    assert "## Seed robustness sweep" not in text and "## Appendix A" not in text
    monkeypatch.setattr(a0b, "SEED", "1")
    text = _findings(_committed_result(), tmp_path, sweep=sweep)
    assert "## Seed robustness sweep" not in text


@pytest.mark.skipif(not (REAL_RUNS_ROOT / "0b-llama-8b-ladder").exists(), reason="run records absent")
def test_real_0b_appendix_a_matches_committed_text(tmp_path):
    first_pass = a0b.compute_first_pass_factual_qa(REAL_RUNS_ROOT)
    text = _findings(_committed_result(), tmp_path, first_pass=first_pass)
    section = _committed_section("---\n\n## Appendix A", None)
    assert section.endswith("in kind and now unchanged in value.\n\n")
    assert text.endswith(section)


def test_appendix_a_raises_when_cliff_comparison_prose_no_longer_holds(tmp_path):
    result = _committed_result()
    # First pass == registered: Qwen's cliff no longer "moves one rung up".
    first_pass = _dc.replace(
        result,
        cells=[c for c in result.cells if c.suite == "factual_qa" and c.run_id in a0b.LADDER_RUN_IDS],
        cliffs={k: v for k, v in result.cliffs.items() if k[1] == "factual_qa"},
        holm_verdicts={k: v for k, v in result.holm_verdicts.items() if k[1] == "factual_qa"},
    )
    with pytest.raises(ValueError, match="Appendix A"):
        _findings(result, tmp_path, first_pass=first_pass)


# ---------------------------------------------------------------------------
# Final review (code turn 2026-09-27), Important 1 + 2: a non-0B run set
# (the 1.5B/0B-prime shape: one ladder run, five suites, no arms) gets a
# neutral title, no 0B scope note, no unverified 0B data claims, and the §5
# trigger "not applicable" with no Arm prose; a non-0B CLI invocation must
# name its own --out-dir and never writes analysis/0b/.
# ---------------------------------------------------------------------------

PRIME_RUN = "0b-prime-qwen-1.5b-ladder"
PRIME_SUITES = ["longctx_retrieval_2a", "longctx_retrieval", "arithmetic", "arithmetic_twins", "factual_qa"]

# Every string below is 0B-only prose (or a 0B data claim) that must never
# appear for a non-0B run set. (The cliff-flag "reads False for every
# family" sentence is data-derived, so it is checked per data, not here.)
ZERO_B_ONLY_STRINGS = [
    "# 0B Confirmatory Findings",
    "## Scope note (PREREG §4)",
    "0B covers only",
    "The 1.5B spectacle model",
    "this analysis touches none of them",
    "all 8 families",
    "Arm 1",
    "Arm 2",
    "Trigger: not fired",
    "Trigger: FIRED",
    "Absent the trigger",
    "no 7B uploader shootout runs",
    "near-ceiling",
    "0/96",
    "~3.7%",
    "first ran (2026-09-03)",
    "Footnote (2026-09-08",
    "## Seed robustness sweep",
    "## Appendix A",
]


def _prime_result(tmp_path, n=10):
    """Synthetic single-ladder five-suite no-arm run. F16 and the top four
    rungs are all correct (zero discordance), the bottom three all wrong
    (a cliff in every suite)."""
    root = tmp_path / "prime-runs"
    ids = {s: [f"{s}-{k}" for k in range(n)] for s in PRIME_SUITES}
    top = set(a0b.LADDER_ORDER[:4])
    labels = {
        s: {"F16": ["correct"] * n, **{
            r: (["correct"] if r in top else ["incorrect"]) * n for r in a0b.LADDER_ORDER
        }}
        for s in PRIME_SUITES
    }
    _write_run(root / PRIME_RUN, ids, labels)
    (root / PRIME_RUN / "fingerprint.txt").write_text("test fixture fingerprint\n")
    return a0b.run_analysis(
        root, n_resamples=50, run_ids=[PRIME_RUN], ladder_run_ids=[PRIME_RUN],
        suites=PRIME_SUITES, arm1_levels=[], arm2_levels=[],
        run_model={PRIME_RUN: "qwen2.5-1.5b-instruct"}, factual_qa_source={},
    )


def test_non_0b_single_ladder_five_suite_run_has_neutral_title_and_scope(tmp_path):
    text = _findings(_prime_result(tmp_path), tmp_path)
    assert text.startswith(f"# Confirmatory Findings — `{PRIME_RUN}` (PREREG §8)\n")
    assert "## Scope\n\nRuns analyzed: `0b-prime-qwen-1.5b-ladder` (qwen2.5-1.5b-instruct, ladder). " in text
    assert "Suites scored: `longctx_retrieval_2a`, `longctx_retrieval`, `arithmetic`, " in text


def test_non_0b_run_prints_none_of_the_0b_only_strings(tmp_path):
    text = _findings(_prime_result(tmp_path), tmp_path)
    for s in ZERO_B_ONLY_STRINGS:
        assert s not in text, s


def test_non_0b_shootout_section_is_not_applicable_without_arm_prose(tmp_path):
    text = _findings(_prime_result(tmp_path), tmp_path)
    section = text[text.index("## Uploader shootout trigger (PREREG §5)"):
                   text.index("## Disclosure: zero-discordance")]
    assert section == (
        "## Uploader shootout trigger (PREREG §5)\n\n"
        "**Trigger: not applicable.** This run set contains no uploader-shootout "
        "arm runs, so no same-label quant-vs-quant pair is computed and the "
        "PREREG §5 trigger is not evaluated here.\n\n"
    )
    assert "| arm | level |" not in text


def test_non_0b_cliff_and_zero_discordance_claims_are_data_derived(tmp_path):
    text = _findings(_prime_result(tmp_path), tmp_path)
    # Bottom-three-rungs-Damaged is a contiguous suffix: flag False everywhere.
    assert "It reads False for every family in this data." in text
    assert "the Damaged cells do not form a contiguous bottom suffix. " in text
    assert "listed in full in the next section.\n" in text
    # 4 zero-discordance rungs x 5 suites, counted per suite.
    assert (
        "This is disclosed here because 20 cell(s) in this data hit this condition "
        "(`longctx_retrieval_2a`: 4, `longctx_retrieval`: 4, `arithmetic`: 4, "
        "`arithmetic_twins`: 4, `factual_qa`: 4)"
    ) in text


def test_cliff_flag_sentence_counts_non_monotonic_families(tmp_path):
    result = _prime_result(tmp_path)
    k = (PRIME_RUN, "arithmetic")
    result.cliffs[k] = _dc.replace(result.cliffs[k], non_monotonic=True)
    assert "It reads True for 1 of 5 families in this data." in a0b._cliff_flag_sentence(result)


def test_zero_discordance_paragraph_when_no_cell_hits_it(tmp_path):
    result = _prime_result(tmp_path)
    result = _dc.replace(result, cells=[c for c in result.cells if c.b_lost + c.c_gained > 0])
    assert "No cell in this data hits this condition" in a0b._zero_discordance_paragraph(result)


def test_real_0b_gated_prose_matches_committed_text(tmp_path):
    """The 0B run set keeps the committed title, scope note, cliff sentence
    and zero-discordance paragraph byte for byte."""
    text = _findings(_committed_result(), tmp_path)
    assert text.startswith("# 0B Confirmatory Findings (PREREG §8)\n")
    for start, end in [
        ("## Scope note (PREREG §4)", "## Per-cell results"),
        ("## Cliffs (ladder runs only)", "| run_id | suite | cliff rung |"),
        ("## Disclosure: zero-discordance", "\n\n---"),
    ]:
        assert _committed_section(start, end) in text, start


def _run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["analyze_0b.py", *argv])
    a0b.main()


NON_0B_ARGV = [
    "--run-ids", PRIME_RUN, "--ladder-run-ids", PRIME_RUN,
    "--suites", ",".join(PRIME_SUITES),
    "--run-model", f"{PRIME_RUN}=qwen2.5-1.5b-instruct",
    "--factual-qa-source", "", "--arm1-levels", "", "--arm2-levels", "",
]


def _forbid_work(monkeypatch):
    calls = []
    for name in ("run_analysis", "write_cells_csv", "write_findings_md"):
        monkeypatch.setattr(a0b, name, lambda *a, _n=name, **k: calls.append(_n))
    return calls


def test_main_non_0b_without_out_dir_errors_and_writes_nothing(monkeypatch, capsys):
    calls = _forbid_work(monkeypatch)
    before = {p: p.stat().st_mtime_ns for p in COMMITTED_0B.iterdir() if p.is_file()}
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, NON_0B_ARGV)
    assert exc.value.code == 2
    assert "--out-dir is required" in capsys.readouterr().err
    assert calls == []
    assert {p: p.stat().st_mtime_ns for p in COMMITTED_0B.iterdir() if p.is_file()} == before


def test_main_non_0b_refuses_analysis_0b_as_out_dir(monkeypatch, capsys):
    calls = _forbid_work(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, [*NON_0B_ARGV, "--out-dir", str(COMMITTED_0B)])
    assert exc.value.code == 2
    assert "must not be analysis/0b/" in capsys.readouterr().err
    assert calls == []


def test_main_non_0b_with_out_dir_writes_findings_md_there(monkeypatch, tmp_path, capsys):
    prime = _prime_result(tmp_path)
    monkeypatch.setattr(a0b, "run_analysis", lambda *a, **k: prime)
    # main() derives runs_root from its own __file__ (pipeline_dir /
    # "runs-cloud" / "pipeline" / "runs"); point that at an isolated fake
    # pipeline dir carrying just this run's fingerprint file, so the real
    # (non-mocked) write_findings_md fingerprint gate is satisfied without
    # touching the real runs-cloud/ tree.
    fake_pipeline_dir = tmp_path / "fake_pipeline"
    fp_dir = fake_pipeline_dir / "runs-cloud" / "pipeline" / "runs" / PRIME_RUN
    fp_dir.mkdir(parents=True)
    (fp_dir / "fingerprint.txt").write_text("test fixture fingerprint\n")
    monkeypatch.setattr(a0b, "__file__", str(fake_pipeline_dir / "scripts" / "analyze_0b.py"))
    out = tmp_path / "out"
    _run_main(monkeypatch, [*NON_0B_ARGV, "--out-dir", str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["FINDINGS.md", "cells.csv"]
    assert (out / "FINDINGS.md").read_text().startswith("# Confirmatory Findings — ")
    assert "shootout trigger: not applicable" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Deferred items (code turn 2026-09-28): --sweep-dir fails loudly instead of
# silently dropping the seed-robustness sweep section for the 0B run set at
# the ratified seed; the fingerprint section reads each run's own
# fingerprint file for a non-0B run set instead of ever pointing at the
# shared runs-cloud/fingerprint.txt (which documents only the 0B machine).
# ---------------------------------------------------------------------------


def test_main_0b_missing_default_sweep_dir_errors_and_writes_nothing(monkeypatch, tmp_path, capsys):
    calls = _forbid_work(monkeypatch)
    missing_sweep = tmp_path / "analysis" / "0b" / "sweep"
    assert not missing_sweep.exists()
    monkeypatch.setattr(a0b, "__file__", str(tmp_path / "scripts" / "analyze_0b.py"))
    before = {p: p.stat().st_mtime_ns for p in COMMITTED_0B.iterdir() if p.is_file()}
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, [])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--sweep-dir" in err
    assert str(missing_sweep) in err
    assert "write_cells_csv" not in calls
    assert "write_findings_md" not in calls
    assert {p: p.stat().st_mtime_ns for p in COMMITTED_0B.iterdir() if p.is_file()} == before


def test_main_0b_missing_explicit_sweep_dir_errors_and_writes_nothing(monkeypatch, tmp_path, capsys):
    calls = _forbid_work(monkeypatch)
    missing_sweep = tmp_path / "no-such-sweep-dir"
    assert not missing_sweep.exists()
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, ["--sweep-dir", str(missing_sweep)])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--sweep-dir" in err
    assert str(missing_sweep) in err
    assert "write_cells_csv" not in calls
    assert "write_findings_md" not in calls


def test_main_0b_present_sweep_dir_still_writes(monkeypatch, tmp_path, capsys):
    """Sanity: an existing --sweep-dir is unaffected by the new gate (the
    prior silent-drop path only mattered for a *missing* directory)."""
    calls = []
    monkeypatch.setattr(
        a0b, "run_analysis", lambda *a, **k: (calls.append("run_analysis"), _committed_result())[1]
    )
    monkeypatch.setattr(a0b, "write_cells_csv", lambda *a, **k: calls.append("write_cells_csv"))
    monkeypatch.setattr(a0b, "write_findings_md", lambda *a, **k: calls.append("write_findings_md"))
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    _run_main(monkeypatch, ["--sweep-dir", str(sweep_dir), "--factual-qa-source", ""])
    assert calls == ["run_analysis", "write_cells_csv", "write_findings_md"]


def test_fingerprint_section_0b_ignores_runs_root(tmp_path):
    """0B run set: the section text is the committed one regardless of
    runs_root (even None), and never reads a per-run fingerprint file."""
    text = _findings(_committed_result(), tmp_path, runs_root=None)
    assert "recorded at\n`runs-cloud/fingerprint.txt`" in text or "recorded at `runs-cloud/fingerprint.txt`" in text
    assert "## Machine / software fingerprint" in text


def test_fingerprint_section_non_0b_requires_runs_root(tmp_path):
    result = _single_run_result(tmp_path)
    with pytest.raises(RuntimeError, match="runs_root"):
        a0b.write_findings_md(result, tmp_path / "F.md")


def test_fingerprint_section_non_0b_missing_per_run_file_fails_loudly(tmp_path):
    result = _single_run_result(tmp_path, run_id="missing-fp-run")
    runs_root = tmp_path / "runs1"
    # _single_run_result already wrote fingerprint.txt for its run id;
    # delete it to exercise the missing-file path.
    (runs_root / "missing-fp-run" / "fingerprint.txt").unlink()
    expected_path = "runs1/missing-fp-run/fingerprint.txt"  # relative to the pipeline dir
    with pytest.raises(RuntimeError, match=re.escape(expected_path)):
        a0b.write_findings_md(result, tmp_path / "F.md", runs_root=runs_root, pipeline_dir=tmp_path)


def test_fingerprint_section_non_0b_reads_own_fingerprint_never_shared_file(tmp_path):
    result = _single_run_result(tmp_path, run_id="own-fp-run")
    runs_root = tmp_path / "runs1"
    fp_path = runs_root / "own-fp-run" / "fingerprint.txt"
    assert fp_path.is_file()  # written by _single_run_result
    text = _findings(result, tmp_path, runs_root=runs_root)
    # Final review Minor 3: the path is printed relative to the pipeline
    # dir, never as an absolute path.
    assert "`runs1/own-fp-run/fingerprint.txt`" in text
    assert str(tmp_path) not in text
    assert "Not restated here to avoid a second copy drifting out of sync." not in text


def test_fingerprint_section_non_0b_empty_file_fails_loudly(tmp_path):
    result = _single_run_result(tmp_path, run_id="empty-fp-run")
    runs_root = tmp_path / "runs1"
    fp_path = runs_root / "empty-fp-run" / "fingerprint.txt"
    for content in ("", "  \n\n"):
        fp_path.write_text(content)
        with pytest.raises(RuntimeError, match="empty") as exc:
            a0b.write_findings_md(result, tmp_path / "F.md", runs_root=runs_root, pipeline_dir=tmp_path)
        assert "runs1/empty-fp-run/fingerprint.txt" in str(exc.value)
    assert not (tmp_path / "F.md").exists()


def test_fingerprint_section_non_0b_runs_root_outside_pipeline_dir_fails(tmp_path):
    result = _single_run_result(tmp_path, run_id="outside-run")
    elsewhere = tmp_path / "pipeline"
    elsewhere.mkdir()
    with pytest.raises(RuntimeError, match="pipeline dir"):
        a0b.write_findings_md(
            result, tmp_path / "F.md", runs_root=tmp_path / "runs1", pipeline_dir=elsewhere
        )


def test_main_non_0b_fingerprint_path_relative_to_pipeline_dir(monkeypatch, tmp_path):
    prime = _prime_result(tmp_path)
    monkeypatch.setattr(a0b, "run_analysis", lambda *a, **k: prime)
    fake_pipeline_dir = tmp_path / "fake_pipeline"
    fp_dir = fake_pipeline_dir / "runs-cloud" / "pipeline" / "runs" / PRIME_RUN
    fp_dir.mkdir(parents=True)
    (fp_dir / "fingerprint.txt").write_text("instance=i-test\n")
    monkeypatch.setattr(a0b, "__file__", str(fake_pipeline_dir / "scripts" / "analyze_0b.py"))
    out = tmp_path / "out"
    _run_main(monkeypatch, [*NON_0B_ARGV, "--out-dir", str(out)])
    text = (out / "FINDINGS.md").read_text()
    assert f"`runs-cloud/pipeline/runs/{PRIME_RUN}/fingerprint.txt`" in text
    assert str(tmp_path) not in text
