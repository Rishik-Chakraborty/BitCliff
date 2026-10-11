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


def _kv_usage(llm, prompt_tokens, completion_tokens):
    """Mimic llama-cpp's post-call state (OPEN_QUESTIONS §17):
    `_input_ids[:n_tokens]` is prompt + evaluated generation."""
    llm._input_ids = [0] * prompt_tokens + list(range(100, 100 + completion_tokens))
    llm.n_tokens = prompt_tokens + completion_tokens
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


class FakeLlm:
    """Echoes a fixed completion text for every prompt (token-id path,
    since longctx items in this pipeline always carry prompt_tokens)."""

    PREFLIGHT_TOKENS = [999]

    def __init__(self, text="The code is 4404.", finish_reason="stop"):
        self._text = text
        self._finish_reason = finish_reason
        self.calls = []

    def tokenize(self, data, add_bos=True, special=False):
        return list(self.PREFLIGHT_TOKENS)

    def create_completion(self, prompt, max_tokens, temperature, top_k, seed):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        finish = "length" if prompt == self.PREFLIGHT_TOKENS else self._finish_reason
        return {
            "choices": [{"text": self._text, "finish_reason": finish}],
            "usage": _kv_usage(self, len(prompt), len(self._text.split())),
        }

    def create_chat_completion(self, messages, max_tokens, temperature, top_k, seed):
        self.calls.append({"prompt": messages[0]["content"], "max_tokens": max_tokens})
        return {
            "choices": [{
                "message": {"content": self._text}, "finish_reason": self._finish_reason,
            }],
            "usage": _kv_usage(self, 3, len(self._text.split())),
        }


def _fake_llm_factory(text="The code is 4404.", finish_reason="stop"):
    def factory(model_path, gen, *execution):
        factory.model_path = model_path
        factory.gen = gen
        factory.execution_args = execution
        factory.llm = FakeLlm(text=text, finish_reason=finish_reason)
        return factory.llm

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
    assert "replay form" in out
    assert factory.llm.calls[0]["prompt"] == FakeLlm.PREFLIGHT_TOKENS


def test_main_fresh_flag_names_fresh_form(tmp_path, capsys, monkeypatch):
    run_dir, items = _write_run_dir(tmp_path, n=2)
    factory = _fake_llm_factory()
    monkeypatch.setattr(rg.gen_mod, "make_llm", factory)
    rc = rg.main([
        "--run-dir", str(run_dir), "--rung", "Q4_K_M", "--suite", "longctx_retrieval",
        "--n", "2", "--model-path", "/fake/model.gguf", "--fresh",
    ])
    assert rc == 0
    assert "fresh" in capsys.readouterr().out
    assert len(factory.llm.calls) == 2


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


class ReplayFakeLlm:
    """Answers each replayed call from the committed outputs, looked up by
    prompt (chat) or by the item's prompt_tokens (completion). The first
    call is the truncation preflight."""

    def __init__(self, run_dir):
        items = {i.id: i for i in rg.read_items(run_dir)}
        self._by_tokens, self._by_prompt = {}, {}
        for d in rg._read_output_dicts(run_dir / "outputs" / "Q4_K_M.jsonl"):
            it = items[d["item_id"]]
            ans = (d["text"], d["finish_reason"])
            if it.prompt_tokens is not None:
                self._by_tokens[tuple(it.prompt_tokens)] = ans
            else:
                self._by_prompt[it.prompt] = ans
        self.n_calls = 0

    def tokenize(self, data, add_bos=True, special=False):
        return [999_999_999]

    def create_completion(self, prompt, max_tokens, temperature, top_k, seed):
        self.n_calls += 1
        if prompt == [999_999_999]:
            text, finish = "x", "length"
        else:
            text, finish = self._by_tokens[tuple(prompt)]
        return {
            "choices": [{"text": text, "finish_reason": finish}],
            "usage": _kv_usage(self, len(prompt), len(text.split())),
        }

    def create_chat_completion(self, messages, max_tokens, temperature, top_k, seed):
        self.n_calls += 1
        text, finish = self._by_prompt[messages[0]["content"]]
        return {
            "choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": _kv_usage(self, 3, len(text.split())),
        }


def test_run_repro_gate_against_real_run_and_fixture_passes():
    fakes = []

    def factory(model_path, gen):
        fakes.append(ReplayFakeLlm(REAL_RUN_DIR))
        return fakes[0]

    n = sum(1 for line in REAL_FIXTURE.read_text().splitlines() if "_extraction_note" not in line)
    diffs = rg.run_repro_gate(
        REAL_RUN_DIR, "Q4_K_M", "longctx_retrieval", n,
        Path("/fake/model.gguf"), llm_factory=factory, fixture_path=REAL_FIXTURE,
    )
    assert diffs == []
    assert fakes[0].n_calls > n  # replayed the earlier calls too


def test_run_repro_gate_against_real_run_without_fixture_passes():
    """Same real run, comparing straight against the committed
    outputs/Q4_K_M.jsonl + grades.jsonl (no --fixture), replay form."""
    fakes = []

    def factory(model_path, gen):
        fakes.append(ReplayFakeLlm(REAL_RUN_DIR))
        return fakes[0]

    diffs = rg.run_repro_gate(
        REAL_RUN_DIR, "Q4_K_M", "longctx_retrieval", 25,
        Path("/fake/model.gguf"), llm_factory=factory,
    )
    assert diffs == []
    assert fakes[0].n_calls > 1000


def test_run_repro_gate_against_real_run_fresh_makes_only_gate_calls():
    fakes = []

    def factory(model_path, gen):
        fakes.append(ReplayFakeLlm(REAL_RUN_DIR))
        return fakes[0]

    diffs = rg.run_repro_gate(
        REAL_RUN_DIR, "Q4_K_M", "longctx_retrieval", 25,
        Path("/fake/model.gguf"), llm_factory=factory, fresh=True,
    )
    assert diffs == []
    assert fakes[0].n_calls == 25


# ---------------------------------------------------------------------------
# Replay form (OPEN_QUESTIONS §17)
# ---------------------------------------------------------------------------


def _write_replay_run(tmp_path, execution=None, order=None, budgets=None):
    """Source run whose recorded order puts 2 `arithmetic` items before the
    longctx gate items, with a different budget per suite."""
    rung = "Q4_K_M"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ids = order or [
        ("arithmetic", "arithmetic-000"), ("arithmetic", "arithmetic-001"),
        ("longctx_retrieval", "longctx_retrieval-0000"),
        ("longctx_retrieval", "longctx_retrieval-0001"),
        ("longctx_retrieval", "longctx_retrieval-0002"),
    ]
    budgets = budgets or {"arithmetic": 16, "longctx_retrieval": 32}
    items = [
        EvalItem(
            iid, suite, f"prompt {iid}", ("4404",),
            (1, 2, 3) if suite == "longctx_retrieval" else None,
        )
        for suite, iid in ids
    ]
    (run_dir / "items.jsonl").write_text("".join(
        json.dumps({
            "id": it.id, "suite": it.suite, "prompt": it.prompt,
            "expected": list(it.expected),
            "prompt_tokens": list(it.prompt_tokens) if it.prompt_tokens else None,
        }) + "\n" for it in items
    ))
    cfg = {"model_id": "m", "suites": {}}
    if execution is not None:
        cfg["execution"] = execution
    write_manifest(
        {rung: {"filename": "x.gguf", "sha256": "deadbeef", "size_bytes": 1}, "_run_config": cfg},
        run_dir / "manifest.json",
    )
    (run_dir / "outputs").mkdir()
    write_records(
        [_output_record(it.id, it.suite, rung, "The code is 4404.",
                        max_tokens_effective=budgets[it.suite]) for it in items],
        run_dir / "outputs" / f"{rung}.jsonl",
    )
    write_grades(
        [GradeResult(it.id, it.suite, rung, "correct", False, False) for it in items],
        run_dir / "grades.jsonl",
    )
    return run_dir


def _run(run_dir, factory, n=2, **kw):
    return rg.run_repro_gate(
        run_dir, "Q4_K_M", "longctx_retrieval", n, Path("/fake/model.gguf"),
        llm_factory=factory, **kw,
    )


def test_replay_calls_preflight_then_recorded_order_through_last_gate_item(tmp_path):
    run_dir = _write_replay_run(tmp_path)
    factory = _fake_llm_factory()
    assert _run(run_dir, factory, n=2) == []
    calls = factory.llm.calls
    # preflight, 2 arithmetic (chat), 2 gate items; the 3rd longctx item is not replayed
    assert [c["prompt"] for c in calls] == [
        FakeLlm.PREFLIGHT_TOKENS, "prompt arithmetic-000", "prompt arithmetic-001", [1, 2, 3], [1, 2, 3],
    ]
    assert [c["max_tokens"] for c in calls][1:] == [16, 16, 32, 32]
    assert factory.execution_args == ()


def test_replay_fresh_makes_only_gate_calls(tmp_path):
    run_dir = _write_replay_run(tmp_path)
    factory = _fake_llm_factory()
    assert _run(run_dir, factory, n=2, fresh=True) == []
    assert [c["max_tokens"] for c in factory.llm.calls] == [32, 32]
    assert all(c["prompt"] == [1, 2, 3] for c in factory.llm.calls)


def test_replay_passes_execution_from_manifest(tmp_path):
    ex = dict(n_ctx=16384, n_batch=512, n_ubatch=512, n_threads=4, n_threads_batch=4,
              flash_attn=True, type_k=1, type_v=1, offload_kqv=True, op_offload=False,
              rope_scaling_type=0, rope_freq_base=1e6, rope_freq_scale=1.0,
              yarn_ext_factor=-1.0, n_gpu_layers=-1, seed=42)
    run_dir = _write_replay_run(tmp_path, execution=ex)
    factory = _fake_llm_factory()
    assert _run(run_dir, factory) == []
    assert factory.execution_args == (rg.ExecutionSettings(**ex),)


def test_replay_gate_items_not_contiguous_replays_through_last(tmp_path):
    order = [
        ("longctx_retrieval", "longctx_retrieval-0000"),
        ("arithmetic", "arithmetic-000"),
        ("longctx_retrieval", "longctx_retrieval-0001"),
        ("arithmetic", "arithmetic-001"),
    ]
    run_dir = _write_replay_run(tmp_path, order=order)
    factory = _fake_llm_factory()
    assert _run(run_dir, factory, n=2) == []
    assert len(factory.llm.calls) == 1 + 3


def test_replay_raises_when_outputs_order_differs_from_items_order(tmp_path):
    run_dir = _write_replay_run(tmp_path)
    out = run_dir / "outputs" / "Q4_K_M.jsonl"
    lines = out.read_text().splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    out.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="order differs"):
        _run(run_dir, _fake_llm_factory())


def test_replay_raises_when_gate_item_absent_from_outputs(tmp_path):
    run_dir = _write_replay_run(tmp_path)
    out = run_dir / "outputs" / "Q4_K_M.jsonl"
    lines = out.read_text().splitlines()
    out.write_text("\n".join(lines[:-1]) + "\n")
    fixture = tmp_path / "f.jsonl"
    fixture.write_text("".join(
        json.dumps({"item_id": f"longctx_retrieval-000{i}", "text": "t",
                    "finish_reason": "stop", "state": "correct", "truncated": False}) + "\n"
        for i in range(3)))
    with pytest.raises(ValueError, match="absent"):
        _run(run_dir, _fake_llm_factory(), n=3, fixture_path=fixture)


def test_replay_raises_when_suite_budgets_disagree(tmp_path):
    run_dir = _write_replay_run(tmp_path)
    out = run_dir / "outputs" / "Q4_K_M.jsonl"
    lines = out.read_text().splitlines()
    d = json.loads(lines[3])
    d["gen_settings"]["max_tokens_effective"] = 99
    lines[3] = json.dumps(d)
    out.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="disagree"):
        _run(run_dir, _fake_llm_factory())


def test_replay_with_fixture_still_requires_run_outputs(tmp_path):
    run_dir = _write_replay_run(tmp_path)
    (run_dir / "outputs" / "Q4_K_M.jsonl").unlink()
    fixture = tmp_path / "f.jsonl"
    fixture.write_text("")
    with pytest.raises(ValueError, match="missing"):
        _run(run_dir, _fake_llm_factory(), fixture_path=fixture)


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
