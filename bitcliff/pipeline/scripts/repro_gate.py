#!/usr/bin/env python3
"""Task 6: reproduction gate (replay form by default, RUN_0B_PRIME §4).

Generated text depends on call history: llama-cpp-python reuses the longest
shared KV-cache token prefix with the previous call (llama.py `generate`),
so the same item can decode differently after different predecessors
(OPEN_QUESTIONS §17, diagnostic 2026-10-07: fresh-object runs differed from
0B on item 0005; replaying 0B's whole call sequence on one object matched
25/25 gate items and 1094/1094 earlier calls). The reproduction gate is
therefore the REPLAY form (default): one model object, built with the source
manifest's `_run_config.execution` when present; the truncation preflight
first (`generate.assert_truncation_finish_reason`, the first call 0B made on
every model object); then every record of the source run's
`outputs/<rung>.jsonl`, in that file's record order (= the recorded call
order, asserted equal to `items.jsonl` order), up to and including the last
of the N gate items, each with its own recorded `max_tokens_effective`. Only
the N gate items are compared. `--fresh` is the old fresh-object, gate-items-
only form; it is NOT a reproduction test (see above) and is kept only for
diagnosis.

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
from bitcliff_pipeline.config import ExecutionSettings, GenSettings
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
    rather than `generate.OutputRecord` instances. This run's committed
    `outputs/Q4_K_M.jsonl` predates Task 5's `gen_wall_seconds`/`gen_tokens`
    fields; since 9546a93 those fields default to None, so
    `generate.read_records` also reads such legacy files. Plain dicts are
    kept because nothing this script reads from a row (`item_id`, `suite`,
    `text`, `finish_reason`, `gen_settings`) needs the dataclass, and dict
    access reads any generation of this file without depending on
    `OutputRecord`'s field set."""
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
    reading -- see its docstring for why plain dicts)."""
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


def _replay_plan(
    run_dir: Path, rung: str, items: list[EvalItem], gate_ids: list[str]
) -> tuple[list[EvalItem], dict[str, int]]:
    """The recorded call sequence up to and including the last gate item, and
    the per-suite budget. Raises if `outputs/<rung>.jsonl` order differs from
    `items.jsonl` order, a gate item is absent, or a suite's records disagree
    on `max_tokens_effective`."""
    out_path = run_dir / "outputs" / f"{rung}.jsonl"
    if not out_path.exists():
        raise ValueError(f"replay needs the source run's outputs, missing: {out_path}")
    rows = _read_output_dicts(out_path)
    recorded_ids = [d["item_id"] for d in rows]
    items_order = [i.id for i in items if i.id in set(recorded_ids)]
    if recorded_ids != items_order:
        raise ValueError(
            f"{out_path} record order differs from items.jsonl order; the "
            "recorded call order cannot be replayed from items.jsonl"
        )
    absent = [g for g in gate_ids if g not in set(recorded_ids)]
    if absent:
        raise ValueError(f"gate items absent from {out_path}: {sorted(absent)}")
    last = max(recorded_ids.index(g) for g in gate_ids)
    by_id = {i.id: i for i in items}
    budgets: dict[str, int] = {}
    for d in rows[: last + 1]:
        b = d["gen_settings"]["max_tokens_effective"]
        if budgets.setdefault(d["suite"], b) != b:
            raise ValueError(
                f"suite {d['suite']!r} records disagree on max_tokens_effective "
                f"({budgets[d['suite']]} vs {b}) in {out_path}"
            )
    return [by_id[i] for i in recorded_ids[: last + 1]], budgets


def run_repro_gate(
    run_dir: Path,
    rung: str,
    suite: str,
    n: int,
    model_path: Path,
    llm_factory=gen_mod.make_llm,
    fixture_path: Path | None = None,
    fresh: bool = False,
) -> list[str]:
    """Regenerates the first `n` `suite` items of `rung` in `run_dir` with
    the recorded settings and diffs them against either the committed
    run's own files (default) or `fixture_path` when given. Default is the
    replay form (module docstring); `fresh=True` generates the gate items
    only on a fresh object, no preflight (NOT a reproduction test).
    `llm_factory` is injectable (`(model_path, GenSettings[, execution])
    -> llm`; the third argument is passed only when the source manifest
    records `_run_config.execution`) so tests can pass a fake provider; no
    real model run happens in this repo's test suite."""
    manifest = load_manifest(run_dir / "manifest.json")
    model_sha256 = manifest[rung]["sha256"]

    items = read_items(run_dir)
    items_by_id = {i.id: i for i in items}
    selected = select_first_n(items, suite, n)
    item_ids = [i.id for i in selected]

    if not fresh and not (run_dir / "outputs" / f"{rung}.jsonl").exists():
        raise ValueError(
            f"replay needs the source run's outputs, missing: "
            f"{run_dir / 'outputs' / f'{rung}.jsonl'}"
        )
    gen, max_tokens_effective = gen_settings_from_output(run_dir, rung, suite)

    expected = (
        load_expected_from_fixture(fixture_path)
        if fixture_path is not None
        else load_expected_from_run(run_dir, rung, suite, item_ids)
    )

    execution = (manifest.get("_run_config") or {}).get("execution")
    if execution is not None:
        llm = llm_factory(model_path, gen, ExecutionSettings(**execution))
    else:
        llm = llm_factory(model_path, gen)

    if fresh:
        records = gen_mod.run_items(
            llm, selected, rung, model_sha256, gen,
            max_tokens_by_suite={suite: max_tokens_effective},
        )
    else:
        sequence, budgets = _replay_plan(run_dir, rung, items, item_ids)
        gen_mod.assert_truncation_finish_reason(llm)
        replayed = gen_mod.run_items(
            llm, sequence, rung, model_sha256, gen, max_tokens_by_suite=budgets,
        )
        gate = set(item_ids)
        records = [r for r in replayed if r.item_id in gate]
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
    parser.add_argument(
        "--fresh", action="store_true",
        help="old form: fresh model object, gate items only, no preflight. "
        "NOT a reproduction test: generated text depends on call history "
        "through llama-cpp-python's KV-prefix reuse (OPEN_QUESTIONS §17, "
        "diagnostic 2026-10-07: fresh runs differed from 0B on item 0005; "
        "replay matched 25/25).",
    )
    args = parser.parse_args(argv)

    diffs = run_repro_gate(
        args.run_dir, args.rung, args.suite, args.n, args.model_path,
        llm_factory=gen_mod.make_llm, fixture_path=args.fixture,
        fresh=args.fresh,
    )
    form = "fresh (NOT a reproduction test)" if args.fresh else "replay"
    if diffs:
        for d in diffs:
            print(d)
        print(
            f"FAIL: {len(diffs)} difference(s) reproducing the first "
            f"{args.n} {args.suite} items of {args.rung} in {args.run_dir} "
            f"[{form} form]"
        )
        return 1
    print(
        f"PASS: first {args.n} {args.suite} items of {args.rung} in "
        f"{args.run_dir} reproduce exactly "
        f"[{form} form]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
