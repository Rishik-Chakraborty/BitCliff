import csv
import json

from bitcliff_pipeline.grading import GradeResult
from bitcliff_pipeline.report import add_retention, aggregate, plot_retention, write_csv, write_json


def g(quant, suite, state, truncated=False, loop=False, i=[0]):
    i[0] += 1
    return GradeResult(f"item-{i[0]:03d}", suite, quant, state, truncated, loop)


GRADES = (
    [g("F16", "longctx_retrieval", "correct") for _ in range(9)]
    + [g("F16", "longctx_retrieval", "wrong")]
    + [g("Q2_K", "longctx_retrieval", "correct") for _ in range(3)]
    + [g("Q2_K", "longctx_retrieval", "partial")]
    + [g("Q2_K", "longctx_retrieval", "wrong", truncated=True, loop=True) for _ in range(6)]
    + [g("F16", "spectacle", "unscored")]
)


def row(rows, quant, suite):
    return next(r for r in rows if r["quant_label"] == quant and r["suite"] == suite)


def test_aggregate_counts_and_accuracy():
    rows = aggregate(GRADES)
    assert all(r["suite"] != "spectacle" for r in rows)  # unscored excluded
    f16 = row(rows, "F16", "longctx_retrieval")
    assert (f16["n"], f16["correct"], f16["accuracy"]) == (10, 9, 0.9)
    q2 = row(rows, "Q2_K", "longctx_retrieval")
    assert q2["n"] == 10
    assert q2["accuracy"] == (3 + 0.5) / 10
    assert q2["truncated"] == 6
    assert q2["loops"] == 6


def test_add_retention():
    rows = add_retention(aggregate(GRADES))
    assert row(rows, "F16", "longctx_retrieval")["retention"] == 1.0
    assert abs(row(rows, "Q2_K", "longctx_retrieval")["retention"] - 0.35 / 0.9) < 1e-9


def test_write_csv_and_json(tmp_path):
    rows = add_retention(aggregate(GRADES))
    write_csv(rows, tmp_path / "r.csv")
    write_json(rows, tmp_path / "r.json")
    with open(tmp_path / "r.csv") as f:
        parsed = list(csv.DictReader(f))
    assert len(parsed) == len(rows)
    assert json.loads((tmp_path / "r.json").read_text()) == rows


def test_plot_retention_writes_png(tmp_path):
    rows = add_retention(aggregate(GRADES))
    out = tmp_path / "retention.png"
    plot_retention(rows, ["F16", "Q2_K"], out)
    assert out.exists() and out.stat().st_size > 0


def test_write_csv_empty_rows(tmp_path):
    write_csv([], tmp_path / "empty.csv")
    assert (tmp_path / "empty.csv").exists()


def test_write_json_empty_rows(tmp_path):
    write_json([], tmp_path / "empty.json")
    assert json.loads((tmp_path / "empty.json").read_text()) == []


def test_aggregate_with_spectacle_only_labels():
    """Test that aggregate adds spectacle_only flag based on quant_label."""
    rows = aggregate(GRADES, spectacle_only_labels=frozenset({"Q2_K"}))
    f16_row = row(rows, "F16", "longctx_retrieval")
    q2_row = row(rows, "Q2_K", "longctx_retrieval")
    assert f16_row["spectacle_only"] is False
    assert q2_row["spectacle_only"] is True


def test_aggregate_default_spectacle_only_all_false():
    """Test that without spectacle_only_labels, all rows have spectacle_only False."""
    rows = aggregate(GRADES)
    assert all(r["spectacle_only"] is False for r in rows)


def test_add_retention_preserves_spectacle_only():
    """Test that add_retention passes spectacle_only through unchanged."""
    rows_agg = aggregate(GRADES, spectacle_only_labels=frozenset({"Q2_K"}))
    rows_ret = add_retention(rows_agg)
    f16_row = row(rows_ret, "F16", "longctx_retrieval")
    q2_row = row(rows_ret, "Q2_K", "longctx_retrieval")
    assert f16_row["spectacle_only"] is False
    assert q2_row["spectacle_only"] is True


def test_write_csv_includes_spectacle_only_column(tmp_path):
    """Test that CSV output includes the spectacle_only column."""
    rows = add_retention(aggregate(GRADES, spectacle_only_labels=frozenset({"Q2_K"})))
    write_csv(rows, tmp_path / "r.csv")
    with open(tmp_path / "r.csv") as f:
        reader = csv.DictReader(f)
        parsed = list(reader)
    assert len(parsed) > 0
    assert "spectacle_only" in parsed[0].keys()
    f16_row = next(r for r in parsed if r["quant_label"] == "F16")
    q2_row = next(r for r in parsed if r["quant_label"] == "Q2_K")
    assert f16_row["spectacle_only"] == "False"
    assert q2_row["spectacle_only"] == "True"


def test_plot_retention_with_mixed_spectacle_only(tmp_path):
    """Test that plot_retention handles mixed spectacle_only rungs."""
    rows = add_retention(aggregate(GRADES, spectacle_only_labels=frozenset({"Q2_K"})))
    out = tmp_path / "retention.png"
    plot_retention(rows, ["F16", "Q2_K"], out)
    assert out.exists() and out.stat().st_size > 0
