#!/usr/bin/env python3
"""Task 6: reproduction gate.

Given a committed run directory, a rung (quant label), a suite and an N,
regenerates the first N items (id order) of that suite with the exact
settings the committed run itself used, then diffs the new output `text` /
`finish_reason` and the newly graded `state` / `truncated` against the
committed run's `outputs/<rung>.jsonl` and `grades.jsonl` (or, with
`--fixture`, against a small merged fixture file extracted from those same
committed files -- see `tests/data/repro_gate/`). Exits nonzero and prints
every difference found; exits 0 with a one-line PASS otherwise.

No item is rebuilt from its suite builder (no tokenizer/corpus rebuild): the
run's own `items.jsonl` already carries each item's exact `prompt_tokens`
(pipeline `__main__.py`'s grade stage reconstructs `EvalItem` the same way,
straight off `items.jsonl` -- this script mirrors that, not a copy of the
builders). Generation reuses `generate.run_items` / `generate.make_llm`;
grading reuses `grading.grade_record` / `grading.GRADERS` -- never a
reimplemented copy of either.

Settings provenance: `manifest.json`'s `_run_config` (written by the
`download`/`all` stage) records only the config file's PER-SUITE overrides
(n_items, seed, and -- for some suites -- max_tokens); it does NOT record
the run's base `GenSettings` (seed/temperature/top_k/max_tokens/n_ctx) that
`config.generation` set, because that block was never copied into the
manifest. Those five fields, plus the per-suite `max_tokens_effective`
that was actually passed to the model for this rung+suite, ARE recorded on
every output record's `gen_settings` (Task 5's per-item settings field) --
so this script reads them from there (the first output record matching
the target rung+suite in `outputs/<rung>.jsonl`) instead of guessing or
hand-copying a registered value.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from bitcliff_pipeline import generate as gen_mod
from bitcliff_pipeline.config import GenSettings
from bitcliff_pipeline.grading import grade_record, read_grades
from bitcliff_pipeline.hashing import load_manifest
from bitcliff_pipeline.items import EvalItem

_GEN_SETTINGS_FIELDS = ("seed", "temperature", "top_k", "max_tokens", "n_ctx")


@dataclass(frozen=True)
class ExpectedRecord:
    """The four fields this gate compares, for one item id, from whichever
    source (`outputs/<rung>.jsonl` + `grades.jsonl`, or a merged fixture)."""

    text: str
    finish_reason: str
    state: str
    truncated: bool


def read_items(run_dir: Path) -> list[EvalItem]:
    """Rebuild `EvalItem`s straight from the run's committed `items.jsonl`
    -- the same reconstruction `__main__.py`'s grade stage performs -- so
    no tokenizer/corpus rebuild is needed to regenerate (items.jsonl
    already holds the exact `prompt_tokens` for token-space suites like
    longctx_retrieval)."""
    items = []
    for line in (run_dir / "items.jsonl").read_text().splitlines():
        d = json.loads(line)
        items.append(
            EvalItem(
                id=d["id"],
                suite=d["suite"],
                prompt=d["prompt"],
                expected=tuple(d["expected"]) if d.get("expected") else None,
                prompt_tokens=tuple(d["prompt_tokens"]) if d.get("prompt_tokens") else None,
            )
        )
    return items


def select_first_n(items: list[EvalItem], suite: str, n: int) -> list[EvalItem]:
    """The first `n` items of `suite`, in id order (lexicographic -- every
    suite's item ids are zero-padded so this agrees with numeric order)."""
    suite_items = sorted((i for i in items if i.suite == suite), key=lambda i: i.id)
    if len(suite_items) < n:
        raise ValueError(
            f"suite {suite!r} has only {len(suite_items)} items in this "
            f"run, cannot select the first {n}"
        )
    return suite_items[:n]


def _read_output_dicts(path: Path) -> list[dict]:
    """Raw JSON rows of an `outputs/<rung>.jsonl` file, as plain dicts
    rather than `generate.OutputRecord` instances. Deliberately NOT
    `generate.read_records`: this run's committed `outputs/Q4_K_M.jsonl`
    predates Task 5's `gen_wall_seconds`/`gen_tokens` fields, and
    `OutputRecord(**json.loads(line))` (what `read_records` does) requires
    every dataclass field -- it would raise on that older, still-valid
    file. Nothing this script reads from a row (`item_id`, `suite`,
    `text`, `finish_reason`, `gen_settings`) depends on those two fields,
    so plain dict access reads any generation of this file."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def gen_settings_from_output(
    run_dir: Path, rung: str, suite: str
) -> tuple[GenSettings, int]:
    """The `GenSettings` this rung's model actually ran with, plus the
    per-suite `max_tokens_effective` that was in effect for `suite` --
    read from the FIRST output record for (rung, suite) in
    `outputs/<rung>.jsonl` (see module docstring: `manifest._run_config`
    does not carry the base GenSettings block at all)."""
    out_path = run_dir / "outputs" / f"{rung}.jsonl"
    for d in _read_output_dicts(out_path):
        if d["suite"] != suite:
            continue
        gs = d["gen_settings"]
        max_tokens_effective = gs["max_tokens_effective"]
        gen = GenSettings(**{k: gs[k] for k in _GEN_SETTINGS_FIELDS})
        return gen, max_tokens_effective
    raise ValueError(f"no output records for suite {suite!r} in {out_path}")


def load_expected_from_run(
    run_dir: Path, rung: str, suite: str, item_ids: list[str]
) -> dict[str, ExpectedRecord]:
    """Expected text/finish_reason/state/truncated for `item_ids`, read
    from the committed run's `outputs/<rung>.jsonl` (text, finish_reason)
    and `grades.jsonl` (state, truncated) via `grading.read_grades` (grade
    reading is reused as-is) and `_read_output_dicts` (plain-dict output
    reading -- see its docstring for why not `generate.read_records`)."""
    wanted = set(item_ids)
    outputs = {
        d["item_id"]: d
        for d in _read_output_dicts(run_dir / "outputs" / f"{rung}.jsonl")
        if d["suite"] == suite and d["item_id"] in wanted
    }
    missing_outputs = wanted - outputs.keys()
    if missing_outputs:
        raise ValueError(
            f"missing output records in outputs/{rung}.jsonl for item ids: "
            f"{sorted(missing_outputs)}"
        )
    grades = {
        g.item_id: g
        for g in read_grades(run_dir / "grades.jsonl")
        if g.suite == suite and g.quant_label == rung and g.item_id in wanted
    }
    missing_grades = wanted - grades.keys()
    if missing_grades:
        raise ValueError(
            f"missing grade records in grades.jsonl for item ids: "
            f"{sorted(missing_grades)}"
        )
    return {
        item_id: ExpectedRecord(
            text=outputs[item_id]["text"],
            finish_reason=outputs[item_id]["finish_reason"],
            state=grades[item_id].state,
            truncated=grades[item_id].truncated,
        )
        for item_id in wanted
    }


def load_expected_from_fixture(fixture_path: Path) -> dict[str, ExpectedRecord]:
    """Expected records from a merged fixture file (one JSON object per
    line: item_id/text/finish_reason/state/truncated). An optional first
    line carrying only `_extraction_note` (this fixture's provenance
    header) is skipped."""
    expected = {}
    for line in fixture_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if "_extraction_note" in d:
            continue
        expected[d["item_id"]] = ExpectedRecord(
            text=d["text"],
            finish_reason=d["finish_reason"],
            state=d["state"],
            truncated=d["truncated"],
        )
    return expected


def diff_against_expected(
    expected: dict[str, ExpectedRecord],
    items_by_id: dict[str, EvalItem],
    output_records: list,
) -> list[str]:
    """Diffs each freshly-generated `output_records` entry against its
    `expected` counterpart: `text` and `finish_reason` directly off the
    output record, `state` and `truncated` off `grading.grade_record`
    (reused, never reimplemented) applied to that same output record."""
    diffs = []
    for record in output_records:
        exp = expected.get(record.item_id)
        if exp is None:
            diffs.append(
                f"{record.item_id}: no expected record available "
                "(fixture/run out of sync with the selected items)"
            )
            continue
        if record.text != exp.text:
            diffs.append(
                f"{record.item_id}: text differs\n"
                f"  expected: {exp.text!r}\n"
                f"  actual:   {record.text!r}"
            )
        if record.finish_reason != exp.finish_reason:
            diffs.append(
                f"{record.item_id}: finish_reason differs: "
                f"expected {exp.finish_reason!r}, got {record.finish_reason!r}"
            )
        grade = grade_record(items_by_id, record)
        if grade.state != exp.state:
            diffs.append(
                f"{record.item_id}: state differs: "
                f"expected {exp.state!r}, got {grade.state!r}"
            )
        if grade.truncated != exp.truncated:
            diffs.append(
                f"{record.item_id}: truncated differs: "
                f"expected {exp.truncated!r}, got {grade.truncated!r}"
            )
    return diffs


def run_repro_gate(
    run_dir: Path,
    rung: str,
    suite: str,
    n: int,
    model_path: Path,
    llm_factory=gen_mod.make_llm,
    fixture_path: Path | None = None,
) -> list[str]:
    """Regenerates the first `n` `suite` items of `rung` in `run_dir` with
    the recorded settings and diffs them against either the committed
    run's own files (default) or `fixture_path` when given. `llm_factory`
    is injectable (`(model_path, GenSettings) -> llm`) so tests can pass a
    fake provider instead of `generate.make_llm`; no real model run
    happens in this repo's test suite."""
    manifest = load_manifest(run_dir / "manifest.json")
    model_sha256 = manifest[rung]["sha256"]

    items = read_items(run_dir)
    items_by_id = {i.id: i for i in items}
    selected = select_first_n(items, suite, n)
    item_ids = [i.id for i in selected]

    gen, max_tokens_effective = gen_settings_from_output(run_dir, rung, suite)

    expected = (
        load_expected_from_fixture(fixture_path)
        if fixture_path is not None
        else load_expected_from_run(run_dir, rung, suite, item_ids)
    )

    llm = llm_factory(model_path, gen)
    records = gen_mod.run_items(
        llm, selected, rung, model_sha256, gen,
        max_tokens_by_suite={suite: max_tokens_effective},
    )
    return diff_against_expected(expected, items_by_id, records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task 6: reproduction gate -- regenerate the first N "
        "items of a suite/rung and diff against the committed run."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--rung", required=True, help="quant label, e.g. Q4_K_M")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument(
        "--fixture", type=Path, default=None,
        help="compare against a merged fixture file instead of the "
        "committed run's outputs/<rung>.jsonl + grades.jsonl",
    )
    args = parser.parse_args(argv)

    diffs = run_repro_gate(
        args.run_dir, args.rung, args.suite, args.n, args.model_path,
        llm_factory=gen_mod.make_llm, fixture_path=args.fixture,
    )
    if diffs:
        for d in diffs:
            print(d)
        print(
            f"FAIL: {len(diffs)} difference(s) reproducing the first "
            f"{args.n} {args.suite} items of {args.rung} in {args.run_dir}"
        )
        return 1
    print(
        f"PASS: first {args.n} {args.suite} items of {args.rung} in "
        f"{args.run_dir} reproduce exactly"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
