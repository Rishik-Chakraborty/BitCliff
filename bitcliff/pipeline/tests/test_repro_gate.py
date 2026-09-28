"""Unit tests for scripts/repro_gate.py (Task 6: reproduction gate).

Loaded by file path (same pattern as tests/test_analyze_0b.py /
tests/test_calibrate_f16.py): the script has no package `__init__.py`.

Covers, with a fake llm (no real model run):
- items.jsonl reconstruction (tuples restored, prompt_tokens preserved)
- first-N-in-id-order selection, scoped to one suite
- GenSettings + per-suite max_tokens_effective sourced from the recorded
  output records (manifest._run_config has no base GenSettings block)
- expected-record loading from a committed run's outputs/<rung>.jsonl +
  grades.jsonl, and from a merged fixture file
- the diff logic itself: identical -> no diffs; changed text/finish_reason
  -> reported; changed grade (state/truncated) -> reported
- the end-to-end `run_repro_gate` orchestration with an injected llm
  factory (fake provider), against both a committed-run-shaped tmp_path
  fixture and a small merged-fixture file
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from bitcliff_pipeline.config import GenSettings
from bitcliff_pipeline.generate import OutputRecord, write_records
from bitcliff_pipeline.grading import GradeResult, write_grades
from bitcliff_pipeline.hashing import write_manifest
from bitcliff_pipeline.items import EvalItem

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "repro_gate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("repro_gate", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["repro_gate"] = module
    spec.loader.exec_module(module)
    return module


rg = _load_module()


GEN = GenSettings(seed=42, temperature=0.0, top_k=1, max_tokens=1024, n_ctx=16384)


def _output_record(item_id, suite, quant_label, text, finish_reason="stop", max_tokens_effective=32):
    return OutputRecord(
        item_id=item_id,
        suite=suite,
        quant_label=quant_label,
        model_sha256="deadbeef",
        prompt=f"prompt for {item_id}",
        text=text,
        finish_reason=finish_reason,
        gen_settings={**{
            "seed": GEN.seed, "temperature": GEN.temperature, "top_k": GEN.top_k,
            "max_tokens": GEN.max_tokens, "n_ctx": GEN.n_ctx,
        }, "max_tokens_effective": max_tokens_effective},
        machine="test-machine",
        gen_wall_seconds=0.01,
        gen_tokens=len(text.split()),
    )


def _write_run_dir(tmp_path, rung="Q4_K_M", suite="longctx_retrieval", n=3):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    items = [
        EvalItem(
            id=f"{suite}-{i:04d}",
            suite=suite,
            prompt=f"prompt {i}",
            expected=("4404",),
            prompt_tokens=(1, 2, 3),
        )
        for i in range(n)
    ]
    # unrelated suite item, to prove suite filtering works
    items.append(EvalItem("arithmetic-000", "arithmetic", "2+2?", ("4",)))
    (run_dir / "items.jsonl").write_text(
        "".join(json.dumps({
            "id": it.id, "suite": it.suite, "prompt": it.prompt,
            "expected": list(it.expected) if it.expected else None,
            "prompt_tokens": list(it.prompt_tokens) if it.prompt_tokens else None,
        }) + "\n" for it in items)
    )
    manifest = {rung: {"filename": f"{rung}.gguf", "sha256": "deadbeef", "size_bytes": 1}}
    manifest["_run_config"] = {"model_id": "qwen2.5-7b-instruct", "suites": {suite: {"max_tokens": 32}}}
    write_manifest(manifest, run_dir / "manifest.json")

    output_records = [
        _output_record(it.id, it.suite, rung, text="The code is 4404.")
        for it in items
    ]
    (run_dir / "outputs").mkdir()
    write_records(output_records, run_dir / "outputs" / f"{rung}.jsonl")

    grades = [
        GradeResult(item_id=it.id, suite=it.suite, quant_label=rung, state="correct", truncated=False, loop=False)
        for it in items
    ]
    write_grades(grades, run_dir / "grades.jsonl")
    return run_dir, items


class FakeLlm:
    """Echoes a fixed completion text for every prompt (token-id path,
    since longctx items in this pipeline always carry prompt_tokens)."""

    def __init__(self, text="The code is 4404.", finish_reason="stop"):
        self._text = text
        self._finish_reason = finish_reason
        self.calls = []

    def create_completion(self, prompt, max_tokens, temperature, top_k, seed):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        return {
            "choices": [{"text": self._text, "finish_reason": self._finish_reason}],
            "usage": {"completion_tokens": len(self._text.split())},
        }


def _fake_llm_factory(text="The code is 4404.", finish_reason="stop"):
    def factory(model_path, gen):
        factory.model_path = model_path
        factory.gen = gen
        return FakeLlm(text=text, finish_reason=finish_reason)

    return factory


# ---------------------------------------------------------------------------
# read_items
# ---------------------------------------------------------------------------


def test_read_items_reconstructs_tuples(tmp_path):
    run_dir, items = _write_run_dir(tmp_path)
    loaded = rg.read_items(run_dir)
    assert len(loaded) == len(items)
    by_id = {i.id: i for i in loaded}
    assert isinstance(by_id["longctx_retrieval-0000"].expected, tuple)
    assert by_id["longctx_retrieval-0000"].expected == ("4404",)
    assert by_id["longctx_retrieval-0000"].prompt_tokens == (1, 2, 3)
    assert by_id["arithmetic-000"].prompt_tokens is None


# ---------------------------------------------------------------------------
# select_first_n
# ---------------------------------------------------------------------------


def test_select_first_n_sorts_by_id_and_filters_suite(tmp_path):
    run_dir, items = _write_run_dir(tmp_path, n=5)
    loaded = rg.read_items(run_dir)
    selected = rg.select_first_n(loaded, "longctx_retrieval", 3)
    assert [i.id for i in selected] == [
        "longctx_retrieval-0000", "longctx_retrieval-0001", "longctx_retrieval-0002",
    ]
    assert all(i.suite == "longctx_retrieval" for i in selected)


def test_select_first_n_raises_if_not_enough_items(tmp_path):
    run_dir, items = _write_run_dir(tmp_path, n=2)
    loaded = rg.read_items(run_dir)
    with pytest.raises(ValueError, match="only 2 items"):
        rg.select_first_n(loaded, "longctx_retrieval", 5)


# ---------------------------------------------------------------------------
# gen_settings_from_output
# ---------------------------------------------------------------------------


def test_gen_settings_from_output_reads_first_matching_record(tmp_path):
    run_dir, items = _write_run_dir(tmp_path)
    gen, max_tokens_effective = rg.gen_settings_from_output(run_dir, "Q4_K_M", "longctx_retrieval")
    assert gen == GEN
    assert max_tokens_effective == 32


def test_gen_settings_from_output_raises_when_suite_absent(tmp_path):
    run_dir, items = _write_run_dir(tmp_path)
    with pytest.raises(ValueError, match="no output records"):
        rg.gen_settings_from_output(run_dir, "Q4_K_M", "factual_qa")


# ---------------------------------------------------------------------------
# load_expected_from_run / load_expected_from_fixture
# ---------------------------------------------------------------------------


def test_load_expected_from_run_merges_outputs_and_grades(tmp_path):
    run_dir, items = _write_run_dir(tmp_path, n=3)
    item_ids = ["longctx_retrieval-0000", "longctx_retrieval-0001"]
    expected = rg.load_expected_from_run(run_dir, "Q4_K_M", "longctx_retrieval", item_ids)
    assert set(expected) == set(item_ids)
    rec = expected["longctx_retrieval-0000"]
    assert rec.text == "The code is 4404."
    assert rec.finish_reason == "stop"
    assert rec.state == "correct"
    assert rec.truncated is False


def test_load_expected_from_run_raises_on_missing_output(tmp_path):
    run_dir, items = _write_run_dir(tmp_path, n=3)
    with pytest.raises(ValueError, match="missing output records"):
        rg.load_expected_from_run(run_dir, "Q4_K_M", "longctx_retrieval", ["nonexistent-id"])


def test_load_expected_from_fixture_skips_header(tmp_path):
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        json.dumps({"_extraction_note": "explains provenance"}) + "\n"
        + json.dumps({"item_id": "a", "text": "hi", "finish_reason": "stop", "state": "correct", "truncated": False}) + "\n"
    )
    expected = rg.load_expected_from_fixture(fixture)
    assert set(expected) == {"a"}
    assert expected["a"].text == "hi"


# ---------------------------------------------------------------------------
# diff_against_expected
# ---------------------------------------------------------------------------


def test_diff_against_expected_identical_is_empty(tmp_path):
    run_dir, items = _write_run_dir(tmp_path, n=2)
    items_by_id = {i.id: i for i in rg.read_items(run_dir)}
    item_ids = ["longctx_retrieval-0000", "longctx_retrieval-0001"]
    expected = rg.load_expected_from_run(run_dir, "Q4_K_M", "longctx_retrieval", item_ids)
    records = [_output_record(iid, "longctx_retrieval", "Q4_K_M", "The code is 4404.") for iid in item_ids]
    diffs = rg.diff_against_expected(expected, items_by_id, records)
    assert diffs == []


def test_diff_against_expected_reports_text_change(tmp_path):
    run_dir, items = _write_run_dir(tmp_path, n=1)
    items_by_id = {i.id: i for i in rg.read_items(run_dir)}
    item_ids = ["longctx_retrieval-0000"]
    expected = rg.load_expected_from_run(run_dir, "Q4_K_M", "longctx_retrieval", item_ids)
    records = [_output_record(item_ids[0], "longctx_retrieval", "Q4_K_M", "totally different text")]
    diffs = rg.diff_against_expected(expected, items_by_id, records)
    assert any("text differs" in d for d in diffs)


def test_diff_against_expected_reports_finish_reason_change(tmp_path):
    run_dir, items = _write_run_dir(tmp_path, n=1)
    items_by_id = {i.id: i for i in rg.read_items(run_dir)}
    item_ids = ["longctx_retrieval-0000"]
    expected = rg.load_expected_from_run(run_dir, "Q4_K_M", "longctx_retrieval", item_ids)
    records = [_output_record(item_ids[0], "longctx_retrieval", "Q4_K_M", "The code is 4404.", finish_reason="length")]
    diffs = rg.diff_against_expected(expected, items_by_id, records)
    assert any("finish_reason differs" in d for d in diffs)


def test_diff_against_expected_reports_grade_change(tmp_path):
    run_dir, items = _write_run_dir(tmp_path, n=1)
    items_by_id = {i.id: i for i in rg.read_items(run_dir)}
    item_ids = ["longctx_retrieval-0000"]
    expected = rg.load_expected_from_run(run_dir, "Q4_K_M", "longctx_retrieval", item_ids)
    # text no longer contains the expected "4404" substring -> grade flips to "wrong"
    records = [_output_record(item_ids[0], "longctx_retrieval", "Q4_K_M", "no idea what the code is")]
    diffs = rg.diff_against_expected(expected, items_by_id, records)
    assert any("state differs" in d for d in diffs)


# ---------------------------------------------------------------------------
# run_repro_gate (end to end, fake llm factory injected)
# ---------------------------------------------------------------------------


def test_run_repro_gate_passes_when_fake_llm_reproduces_committed_run(tmp_path):
    run_dir, items = _write_run_dir(tmp_path, n=3)
    factory = _fake_llm_factory(text="The code is 4404.", finish_reason="stop")
    diffs = rg.run_repro_gate(
        run_dir, "Q4_K_M", "longctx_retrieval", 3, Path("/fake/model.gguf"),
        llm_factory=factory,
    )
    assert diffs == []
    # settings sourced from the recorded output records, not guessed
    assert factory.gen == GEN
    assert factory.model_path == Path("/fake/model.gguf")


def test_run_repro_gate_fails_when_fake_llm_diverges(tmp_path):
    run_dir, items = _write_run_dir(tmp_path, n=3)
    factory = _fake_llm_factory(text="something else entirely", finish_reason="stop")
    diffs = rg.run_repro_gate(
        run_dir, "Q4_K_M", "longctx_retrieval", 3, Path("/fake/model.gguf"),
        llm_factory=factory,
    )
    assert len(diffs) >= 3  # one item's text (+ likely grade) differs, for each of 3 items


def test_run_repro_gate_against_fixture(tmp_path):
    run_dir, items = _write_run_dir(tmp_path, n=2)
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        "".join(
            json.dumps({
                "item_id": it.id, "text": "The code is 4404.",
                "finish_reason": "stop", "state": "correct", "truncated": False,
            }) + "\n"
            for it in items if it.suite == "longctx_retrieval"
        )
    )
    factory = _fake_llm_factory(text="The code is 4404.", finish_reason="stop")
    diffs = rg.run_repro_gate(
        run_dir, "Q4_K_M", "longctx_retrieval", 2, Path("/fake/model.gguf"),
        llm_factory=factory, fixture_path=fixture,
    )
    assert diffs == []


# ---------------------------------------------------------------------------
# main() CLI, exit codes
# ---------------------------------------------------------------------------


def test_main_exits_zero_and_prints_pass(tmp_path, capsys, monkeypatch):
    run_dir, items = _write_run_dir(tmp_path, n=2)
    factory = _fake_llm_factory(text="The code is 4404.", finish_reason="stop")
    monkeypatch.setattr(rg.gen_mod, "make_llm", factory)
    rc = rg.main([
        "--run-dir", str(run_dir), "--rung", "Q4_K_M", "--suite", "longctx_retrieval",
        "--n", "2", "--model-path", "/fake/model.gguf",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out


# ---------------------------------------------------------------------------
# Integration: the real committed run + the real committed fixture
# ---------------------------------------------------------------------------

REAL_RUN_DIR = (
    Path(__file__).resolve().parent.parent
    / "runs-cloud" / "pipeline" / "runs" / "0b-qwen-7b-ladder"
)
REAL_FIXTURE = (
    Path(__file__).resolve().parent
    / "data" / "repro_gate" / "0b-qwen-7b-ladder-Q4_K_M-longctx-first25.jsonl"
)


class SequentialFakeLlm:
    """Returns the fixture's (text, finish_reason) pairs in order -- valid
    because `run_items` calls the llm once per item, in the exact order
    `select_first_n` (id order) hands it items, which is the same order
    the fixture rows are in (both built by sorting on item id)."""

    def __init__(self, texts_and_finish_reasons):
        self._rows = iter(texts_and_finish_reasons)

    def create_completion(self, prompt, max_tokens, temperature, top_k, seed):
        text, finish_reason = next(self._rows)
        return {
            "choices": [{"text": text, "finish_reason": finish_reason}],
            "usage": {"completion_tokens": len(text.split())},
        }


def test_run_repro_gate_against_real_run_and_fixture_passes():
    fixture_rows = [
        json.loads(line)
        for line in REAL_FIXTURE.read_text().splitlines()
        if "_extraction_note" not in line
    ]
    rows = [(row["text"], row["finish_reason"]) for row in fixture_rows]

    def factory(model_path, gen):
        return SequentialFakeLlm(rows)

    diffs = rg.run_repro_gate(
        REAL_RUN_DIR, "Q4_K_M", "longctx_retrieval", len(rows),
        Path("/fake/model.gguf"), llm_factory=factory, fixture_path=REAL_FIXTURE,
    )
    assert diffs == []


def test_run_repro_gate_against_real_run_without_fixture_passes():
    """Same real run, but comparing straight against the committed
    outputs/Q4_K_M.jsonl + grades.jsonl (no --fixture) -- exercises the
    default (non-fixture) path end to end."""
    items = rg.read_items(REAL_RUN_DIR)
    selected = rg.select_first_n(items, "longctx_retrieval", 25)
    expected = rg.load_expected_from_run(
        REAL_RUN_DIR, "Q4_K_M", "longctx_retrieval", [i.id for i in selected]
    )
    rows = [(expected[i.id].text, expected[i.id].finish_reason) for i in selected]

    def factory(model_path, gen):
        return SequentialFakeLlm(rows)

    diffs = rg.run_repro_gate(
        REAL_RUN_DIR, "Q4_K_M", "longctx_retrieval", 25,
        Path("/fake/model.gguf"), llm_factory=factory,
    )
    assert diffs == []


def test_main_exits_nonzero_and_prints_diffs(tmp_path, capsys, monkeypatch):
    run_dir, items = _write_run_dir(tmp_path, n=2)
    factory = _fake_llm_factory(text="wrong text", finish_reason="stop")
    monkeypatch.setattr(rg.gen_mod, "make_llm", factory)
    rc = rg.main([
        "--run-dir", str(run_dir), "--rung", "Q4_K_M", "--suite", "longctx_retrieval",
        "--n", "2", "--model-path", "/fake/model.gguf",
    ])
    assert rc != 0
    out = capsys.readouterr().out
    assert "text differs" in out
