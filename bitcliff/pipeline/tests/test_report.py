import csv
import json

from bitcliff_pipeline.grading import GradeResult
from bitcliff_pipeline.report import add_retention, aggregate, plot_retention, write_csv, write_json


def g(quant, suite, state, truncated=False, loop=False, i=[0]):
    i[0] += 1
    return GradeResult(f"item-{i[0]:03d}", suite, quant, state, truncated, loop)


GRADES = (
    [g("F16", "retrieval", "correct") for _ in range(9)]
    + [g("F16", "retrieval", "wrong")]
    + [g("Q2_K", "retrieval", "correct") for _ in range(3)]
    + [g("Q2_K", "retrieval", "partial")]
    + [g("Q2_K", "retrieval", "wrong", truncated=True, loop=True) for _ in range(6)]
    + [g("F16", "spectacle", "unscored")]
)


def row(rows, quant, suite):
    return next(r for r in rows if r["quant_label"] == quant and r["suite"] == suite)


def test_aggregate_counts_and_accuracy():
    rows = aggregate(GRADES)
    assert all(r["suite"] != "spectacle" for r in rows)  # unscored excluded
    f16 = row(rows, "F16", "retrieval")
    assert (f16["n"], f16["correct"], f16["accuracy"]) == (10, 9, 0.9)
    q2 = row(rows, "Q2_K", "retrieval")
    assert q2["n"] == 10
    assert q2["accuracy"] == (3 + 0.5) / 10
    assert q2["truncated"] == 6
    assert q2["loops"] == 6


def test_add_retention():
    rows = add_retention(aggregate(GRADES))
    assert row(rows, "F16", "retrieval")["retention"] == 1.0
    assert abs(row(rows, "Q2_K", "retrieval")["retention"] - 0.35 / 0.9) < 1e-9


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
