"""Unit + integration tests for scripts/analyze_0b.py's driver logic:
pairing (incl. hard-error on id mismatch), the ladder-label-set gate,
cliff/Holm wiring, the §5 shootout trigger (incl. the cross-run "not
pairable" path), and determinism (run twice -> byte-identical outputs).

Loaded by file path (same pattern as tests/test_calibrate_f16.py): the
script has no package `__init__.py`.
"""

import importlib.util
import json
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
    assert "PROVISIONAL" in text
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
