import argparse
import dataclasses
import json
from pathlib import Path

from . import generate as gen_mod
from .config import LadderConfig, load_config
from .divergence import divergence_for_records
from .grading import grade_record, read_grades
from .hashing import build_manifest, load_manifest, verify_manifest, write_manifest
from .items import EvalItem
from .models import ensure_quants, resolve_all
from .report import add_retention, aggregate, plot_retention, write_csv, write_json


def build_items(config: LadderConfig, base_dir: Path) -> list[EvalItem]:
    from .suites import arithmetic, retrieval, spectacle

    items: list[EvalItem] = []
    suites = config.suites
    if "retrieval" in suites:
        s = suites["retrieval"]
        items += retrieval.generate_items(s["n_items"], s["n_pairs"], s["seed"])
    if "arithmetic" in suites:
        s = suites["arithmetic"]
        items += arithmetic.load_gsm8k_items(s["n_items"], s["seed"])
    if "spectacle" in suites:
        items += spectacle.load_items(base_dir / suites["spectacle"]["path"])
    return items


def run_pipeline(
    config: LadderConfig,
    run_id: str,
    models_dir: Path,
    runs_dir: Path,
    stage: str,
    llm_factory=gen_mod.make_llm,
    base_dir: Path = Path("."),
) -> None:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"

    if stage in ("download", "all"):
        ensure_quants(config, models_dir)
        paths = resolve_all(config, models_dir)
        write_manifest(build_manifest(paths), manifest_path)
        print(f"manifest written: {manifest_path}")

    if stage in ("generate", "all"):
        paths = resolve_all(config, models_dir)
        manifest = load_manifest(manifest_path)
        verify_manifest(manifest, paths)
        items = build_items(config, base_dir)
        items_path = run_dir / "items.jsonl"
        items_path.write_text(
            "".join(json.dumps(dataclasses.asdict(i)) + "\n" for i in items)
        )
        for label, path in paths.items():
            out_path = run_dir / "outputs" / f"{label}.jsonl"
            if out_path.exists():
                print(f"skip {label}: {out_path} exists")
                continue
            print(f"generating {label} ({len(items)} items)...")
            llm = llm_factory(path, config.generation)
            records = gen_mod.run_items(
                llm, items, label, manifest[label]["sha256"], config.generation
            )
            gen_mod.write_records(records, out_path)

    if stage in ("grade", "all"):
        items = [
            EvalItem(**{**d, "expected": tuple(d["expected"]) if d["expected"] else None})
            for d in map(json.loads, (run_dir / "items.jsonl").read_text().splitlines())
        ]
        items_by_id = {i.id: i for i in items}
        baseline = {
            r.item_id: r for r in gen_mod.read_records(run_dir / "outputs" / "F16.jsonl")
        }
        graded_dicts = []
        for out_path in sorted((run_dir / "outputs").glob("*.jsonl")):
            for record in gen_mod.read_records(out_path):
                expected_prompt = items_by_id[record.item_id].prompt
                if record.prompt != expected_prompt:
                    raise RuntimeError(
                        f"{out_path.name}: prompt mismatch for {record.item_id} — "
                        "outputs are stale relative to items.jsonl; delete "
                        "runs/<run-id>/outputs/ and regenerate"
                    )
                g = grade_record(items_by_id, record)
                div = divergence_for_records(
                    baseline[record.item_id], record, tokenize=str.split
                )
                graded_dicts.append({**dataclasses.asdict(g), "divergence": div})
        with open(run_dir / "grades.jsonl", "w") as f:
            for d in graded_dicts:
                f.write(json.dumps(d) + "\n")
        print(f"graded {len(graded_dicts)} outputs")

    if stage in ("report", "all"):
        grades = read_grades(run_dir / "grades.jsonl")
        rows = add_retention(aggregate(grades))
        write_csv(rows, run_dir / "results.csv")
        write_json(rows, run_dir / "results.json")
        ladder_order = ["F16"] + [q.label for q in config.quants]
        plot_retention(rows, ladder_order, run_dir / "retention.png")
        print(f"report written under {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="bitcliff_pipeline")
    parser.add_argument("config")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage", default="all",
        choices=["download", "generate", "grade", "report", "all"],
    )
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()
    config = load_config(args.config)
    run_pipeline(
        config,
        run_id=args.run_id,
        models_dir=Path(args.models_dir),
        runs_dir=Path(args.runs_dir),
        stage=args.stage,
        base_dir=Path(args.config).resolve().parent.parent,
    )


if __name__ == "__main__":
    main()
