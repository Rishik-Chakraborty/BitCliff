"""Tests for scripts/nll_gate_compare.py (RUN_0B_PRIME.md §4 tiered match rule)."""

import dataclasses
import importlib.util
import sys
from pathlib import Path

from bitcliff_pipeline import nll_scorer
from bitcliff_pipeline.nll_scorer import NLLRecord

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "nll_gate_compare.py"
_spec = importlib.util.spec_from_file_location("nll_gate_compare", SCRIPT)
gc = importlib.util.module_from_spec(_spec)
sys.modules["nll_gate_compare"] = gc
_spec.loader.exec_module(gc)


def _rec(i, pt=(1.0, 2.0), dig=1.5, machine="m1"):
    return NLLRecord(
        item_id=f"it{i}", suite="longctx_retrieval", quant_label="Q4_K_M",
        model_sha256="ab" * 32, per_token_nll=list(pt),
        span_mean_nll=sum(pt) / len(pt), digits_only_mean_nll=dig,
        n_answer_tokens=len(pt), machine=machine,
    )


def _base():
    return [_rec(0), _rec(1, (0.5, 0.25, 0.125)), _rec(2, (3.0,), dig=None)]


def _write(tmp_path, name, recs):
    p = tmp_path / name
    nll_scorer.write_records(recs, p)
    return p


def test_tier1_identical():
    r = gc.compare_records(_base(), _base())
    assert r.outcome == "tier1" and r.passed


def test_machine_only_difference_is_tier1():
    other = [dataclasses.replace(x, machine="other-box") for x in _base()]
    assert gc.compare_records(other, _base()).outcome == "tier1"


def test_tier2_reports_maxima():
    prod = _base()
    prod[1] = dataclasses.replace(
        prod[1], per_token_nll=[0.5 + 4e-4, 0.25, 0.125 - 2e-4],
        span_mean_nll=prod[1].span_mean_nll + 7e-5, digits_only_mean_nll=1.5 + 3e-5,
    )
    r = gc.compare_records(prod, _base())
    assert r.outcome == "tier2" and r.passed
    assert abs(r.max_per_token_diff - 4e-4) < 1e-12
    assert abs(r.max_span_mean_diff - 7e-5) < 1e-12
    assert abs(r.max_digits_only_diff - 3e-5) < 1e-12
    assert "within floating-point tolerance" in gc.format_result(r)
    assert "compared in recorded (file) order" in gc.format_result(r)


def test_per_token_over_tolerance_is_mismatch():
    prod = _base()
    prod[0] = dataclasses.replace(prod[0], per_token_nll=[1.0, 2.0 + 2e-3])
    r = gc.compare_records(prod, _base())
    assert r.outcome == "mismatch" and (r.offending_item, r.offending_field) == ("it0", "per_token_nll")


def test_order_mismatch():
    prod = _base()
    prod[0], prod[1] = prod[1], prod[0]
    r = gc.compare_records(prod, _base())
    assert r.outcome == "mismatch" and r.offending_field == "item_id" and r.offending_item == "it0"


def test_count_mismatch():
    r = gc.compare_records(_base()[:2], _base())
    assert r.outcome == "mismatch" and r.offending_field == "record_count"


def test_field_mismatches():
    for field, val in [("model_sha256", "cd" * 32), ("suite", "x"), ("quant_label", "Q3"),
                       ("n_answer_tokens", 9)]:
        prod = _base()
        prod[1] = dataclasses.replace(prod[1], **{field: val})
        r = gc.compare_records(prod, _base())
        assert r.outcome == "mismatch" and r.offending_field == field and r.offending_item == "it1"


def test_token_length_and_none_mismatch():
    prod = _base()
    prod[1] = dataclasses.replace(prod[1], per_token_nll=[0.5, 0.25])
    assert gc.compare_records(prod, _base()).offending_field == "per_token_nll"
    prod = _base()
    prod[2] = dataclasses.replace(prod[2], digits_only_mean_nll=1.0)
    r = gc.compare_records(prod, _base())
    assert r.outcome == "mismatch" and r.offending_field == "digits_only_mean_nll"


def test_main_exit_codes(tmp_path, capsys):
    a = _write(tmp_path, "a.jsonl", _base())
    b = _write(tmp_path, "b.jsonl", [dataclasses.replace(x, machine="z") for x in _base()])
    assert gc.main([str(b), str(a)]) == 0
    assert "reproduced exactly" in capsys.readouterr().out
    c = _write(tmp_path, "c.jsonl", _base()[:1])
    assert gc.main([str(c), str(a)]) == 1
    out = capsys.readouterr().out
    assert "MISMATCH" in out and "recorded (file) order" in out


def _with_pt(vals_prod, vals_com):
    prod, com = _base(), _base()
    prod[0] = dataclasses.replace(prod[0], per_token_nll=vals_prod)
    com[0] = dataclasses.replace(com[0], per_token_nll=vals_com)
    return gc.compare_records(prod, com)


def test_nan_per_token_produced_side_is_mismatch():
    r = _with_pt([1.0, float("nan")], [1.0, 2.0])
    assert r.outcome == "mismatch" and (r.offending_item, r.offending_field) == ("it0", "per_token_nll")


def test_nan_per_token_both_sides_is_mismatch():
    nan = float("nan")
    assert _with_pt([1.0, nan], [1.0, nan]).outcome == "mismatch"


def test_inf_vs_inf_per_token_is_mismatch():
    inf = float("inf")
    assert _with_pt([1.0, inf], [1.0, inf]).outcome == "mismatch"


def test_nan_span_mean_is_mismatch():
    prod = _base()
    prod[1] = dataclasses.replace(prod[1], span_mean_nll=float("nan"))
    r = gc.compare_records(prod, _base())
    assert r.outcome == "mismatch" and (r.offending_item, r.offending_field) == ("it1", "span_mean_nll")


def test_nan_digits_only_is_mismatch():
    prod = _base()
    prod[0] = dataclasses.replace(prod[0], digits_only_mean_nll=float("nan"))
    assert gc.compare_records(prod, _base()).offending_field == "digits_only_mean_nll"
