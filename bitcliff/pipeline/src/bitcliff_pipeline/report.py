import csv
import json
from collections import defaultdict
from pathlib import Path

from .grading import GradeResult


def aggregate(grades: list[GradeResult]) -> list[dict]:
    groups: dict[tuple[str, str], list[GradeResult]] = defaultdict(list)
    for g in grades:
        if g.state != "unscored":
            groups[(g.quant_label, g.suite)].append(g)
    rows = []
    for (quant_label, suite), gs in sorted(groups.items()):
        n = len(gs)
        correct = sum(1 for g in gs if g.state == "correct")
        partial = sum(1 for g in gs if g.state == "partial")
        rows.append(
            {
                "quant_label": quant_label,
                "suite": suite,
                "n": n,
                "correct": correct,
                "partial": partial,
                "wrong": sum(1 for g in gs if g.state == "wrong"),
                "truncated": sum(1 for g in gs if g.truncated),
                "loops": sum(1 for g in gs if g.loop),
                "accuracy": (correct + 0.5 * partial) / n,
            }
        )
    return rows


def add_retention(rows: list[dict], baseline_label: str = "F16") -> list[dict]:
    baseline = {
        r["suite"]: r["accuracy"] for r in rows if r["quant_label"] == baseline_label
    }
    out = []
    for r in rows:
        base = baseline.get(r["suite"])
        retention = r["accuracy"] / base if base else None
        out.append({**r, "retention": retention})
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))


def plot_retention(rows: list[dict], ladder_order: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    suites = sorted({r["suite"] for r in rows})
    for suite in suites:
        by_label = {r["quant_label"]: r["retention"] for r in rows if r["suite"] == suite}
        ys = [by_label.get(label) for label in ladder_order]
        ax.plot(ladder_order, ys, marker="o", label=suite)
    ax.set_xlabel("quant level (exploratory pilot — numbers are thrown away)")
    ax.set_ylabel("retention vs F16")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
