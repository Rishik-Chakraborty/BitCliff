import dataclasses
import json
import re
from pathlib import Path

import pytest

import bitcliff_pipeline.__main__ as main_mod
import bitcliff_pipeline.generate as gen_mod
import bitcliff_pipeline.grading as grading_mod
from bitcliff_pipeline.__main__ import build_items, run_pipeline
from bitcliff_pipeline.config import GenSettings, LadderConfig, QuantFile
from bitcliff_pipeline.items import EvalItem
from bitcliff_pipeline.suites import arithmetic, factual_qa

# A tiny synthetic gsm8k-shaped record set, injected via
# `arithmetic.items_from_records` in place of the real (networked) GSM8K
# dataset — see the `fake_gsm8k` fixture below. Kept in the same shape as
# test_arithmetic.py's RECORDS so it exercises the real extraction/grading
# logic, not a mock of it.
ARITHMETIC_RECORDS = [
    {
        "question": f"Q{i}: If x = {i} and y = {i}, what is x + y?",
        "answer": f"x + y = {i} + {i} = {2 * i}\n#### {2 * i}",
    }
    for i in range(1, 21)
]


def make_config(tmp_path: Path, spectacle_only: bool = False) -> LadderConfig:
    spectacle = tmp_path / "configs" / "prompts_spectacle.yaml"
    spectacle.parent.mkdir(parents=True)
    spectacle.write_text(
        "prompts:\n  - {id: spec-001, prompt: 'Say hi.'}\n"
        "  - {id: spec-002, prompt: 'Say bye.'}\n"
    )
    return LadderConfig(
        model_id="test-model",
        hf_repo="fake/repo",
        f16_path=tmp_path / "models" / "f16.gguf",
        quants=(QuantFile("Q4_K_M", "m-Q4_K_M.gguf", "bartowski", True, spectacle_only=spectacle_only),),
        generation=GenSettings(42, 0.0, 1, 640, 4096),
        suites={
            "arithmetic": {"n_items": 4, "seed": 1},
            "spectacle": {"path": "configs/prompts_spectacle.yaml"},
        },
    )


class FakeLlm:
    """Echoes the prompt back for spectacle (unscored) items. For arithmetic
    items it solves the embedded "x = N and y = N" pattern so the scored-
    suite path (aggregate/retention) is still exercised end-to-end without a
    real model or network access.
    """

    _ARITH_RE = re.compile(r"x = (\d+) and y = (\d+)")

    def tokenize(self, text, add_bos=True, special=False):
        return list(range(len(text.split())))

    def create_chat_completion(self, messages, **kwargs):
        prompt = messages[0]["content"]
        m = self._ARITH_RE.search(prompt)
        content = f"#### {int(m.group(1)) + int(m.group(2))}" if m else prompt
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}

    def create_completion(self, prompt, max_tokens, **kwargs):
        # the truncation preflight (generate stage) probes via the token
        # path with a small budget and expects an honest "length"
        return {"choices": [{"text": "1, 2, 3", "finish_reason": "length"}]}


@pytest.fixture(autouse=True)
def fake_gsm8k(monkeypatch):
    """build_items()'s arithmetic branch normally calls arithmetic.load_gsm8k_items,
    which hits the network (HF `datasets`). Redirect it to a tiny local record
    set via items_from_records, exactly as suggested for testing that function.
    """
    monkeypatch.setattr(
        arithmetic,
        "load_gsm8k_items",
        lambda n_items, seed: arithmetic.items_from_records(
            ARITHMETIC_RECORDS, n_items, seed
        ),
    )


def test_pipeline_end_to_end(tmp_path):
    cfg = make_config(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "m-Q4_K_M.gguf").write_bytes(b"quant bytes")
    cfg.f16_path.write_bytes(b"f16 bytes")
    runs_dir = tmp_path / "runs"

    run_pipeline(
        cfg, run_id="pilot-test", models_dir=models_dir, runs_dir=runs_dir,
        stage="all", llm_factory=lambda path, gen: FakeLlm(), base_dir=tmp_path,
    )

    run = runs_dir / "pilot-test"
    assert (run / "manifest.json").exists()
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["Q4_K_M"]["uploader"] == "bartowski"
    assert manifest["Q4_K_M"]["imatrix"] is True
    assert manifest["Q4_K_M"]["spectacle_only"] is False
    assert manifest["F16"]["uploader"] == "bitcliff-local-f16-conversion"
    assert manifest["F16"]["spectacle_only"] is False
    assert (run / "outputs" / "F16.jsonl").exists()
    assert (run / "outputs" / "Q4_K_M.jsonl").exists()
    grades = [json.loads(l) for l in (run / "grades.jsonl").read_text().splitlines()]
    # (4 arithmetic + 2 spectacle) items, for 2 ladder rungs
    assert len(grades) == 12
    assert {g["quant_label"] for g in grades} == {"F16", "Q4_K_M"}
    assert all("divergence" in g for g in grades)
    results = json.loads((run / "results.json").read_text())
    assert results  # the arithmetic suite produced scored rows
    assert all(r["retention"] == 1.0 for r in results)  # identical fake outputs
    assert (run / "retention.png").exists()


def test_grade_stage_rejects_stale_outputs(tmp_path):
    cfg = make_config(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "m-Q4_K_M.gguf").write_bytes(b"quant bytes")
    cfg.f16_path.write_bytes(b"f16 bytes")
    runs_dir = tmp_path / "runs"

    run_pipeline(
        cfg, run_id="pilot-test", models_dir=models_dir, runs_dir=runs_dir,
        stage="all", llm_factory=lambda path, gen: FakeLlm(), base_dir=tmp_path,
    )

    # Simulate a stale items.jsonl: rewrite one item's prompt while keeping its id.
    # Outputs from the earlier "all" run still hold the old prompt.
    run = runs_dir / "pilot-test"
    items_path = run / "items.jsonl"
    items = [json.loads(l) for l in items_path.read_text().splitlines()]
    items[0]["prompt"] = items[0]["prompt"] + " (edited)"
    items_path.write_text("".join(json.dumps(i) + "\n" for i in items))

    with pytest.raises(RuntimeError, match="prompt mismatch"):
        run_pipeline(
            cfg, run_id="pilot-test", models_dir=models_dir, runs_dir=runs_dir,
            stage="grade", llm_factory=lambda path, gen: FakeLlm(), base_dir=tmp_path,
        )


def test_manifest_records_run_config(tmp_path):
    """Pre-0B ticket (freeze-plan §10): manifest.json carries the run's
    suite config so the packager can positively identify corpus provenance
    (e.g. a longctx run's corpus_sha256) instead of the seed heuristic."""
    cfg = make_config(tmp_path)
    cfg = dataclasses.replace(
        cfg,
        suites={
            **cfg.suites,
            "longctx_retrieval": {
                "variant": "multivalue4",
                "target_tokens": 8192,
                "seed": 2024,
                "n_items": 96,
                "corpus_sha256": "0a21a13834b5215876bd4019af8fbc436abbfbb61b2826db62223eb990071443",
                "max_tokens": 32,
            },
        },
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "m-Q4_K_M.gguf").write_bytes(b"quant bytes")
    cfg.f16_path.write_bytes(b"f16 bytes")
    runs_dir = tmp_path / "runs"

    run_pipeline(
        cfg, run_id="runconfig-test", models_dir=models_dir, runs_dir=runs_dir,
        stage="download", llm_factory=lambda path, gen: FakeLlm(), base_dir=tmp_path,
    )

    manifest = json.loads((runs_dir / "runconfig-test" / "manifest.json").read_text())
    rc = manifest["_run_config"]
    assert rc["model_id"] == "test-model"
    assert rc["suites"]["longctx_retrieval"]["corpus_sha256"] == (
        "0a21a13834b5215876bd4019af8fbc436abbfbb61b2826db62223eb990071443"
    )
    # metadata key must not look like a rung to downstream consumers
    assert set(manifest) - {"_run_config"} == {"F16", "Q4_K_M"}


def test_build_items_covers_configured_suites(tmp_path):
    cfg = make_config(tmp_path)
    items = build_items(cfg, base_dir=tmp_path)
    suites = [i.suite for i in items]
    assert suites.count("arithmetic") == 4
    assert suites.count("spectacle") == 2


def test_build_items_factual_qa_forwards_alias_augmentation_path(tmp_path, monkeypatch):
    """build_items' factual_qa branch reads an optional
    alias_augmentation_path key from the suite config, resolved against
    base_dir (same pattern as the spectacle suite's `path` key), and
    forwards it to load_popqa_items."""
    captured = {}

    def fake_load_popqa_items(n_items, seed, alias_augmentation_path=None):
        captured["args"] = (n_items, seed, alias_augmentation_path)
        return []

    monkeypatch.setattr(factual_qa, "load_popqa_items", fake_load_popqa_items)

    cfg = make_config(tmp_path)
    cfg = dataclasses.replace(
        cfg,
        suites={
            **cfg.suites,
            "factual_qa": {
                "n_items": 500,
                "seed": 7411,
                "alias_augmentation_path": "data/aliases.json",
            },
        },
    )
    build_items(cfg, base_dir=tmp_path)
    assert captured["args"] == (500, 7411, tmp_path / "data/aliases.json")


def test_build_items_factual_qa_without_alias_augmentation_path_passes_none(tmp_path, monkeypatch):
    captured = {}

    def fake_load_popqa_items(n_items, seed, alias_augmentation_path=None):
        captured["args"] = (n_items, seed, alias_augmentation_path)
        return []

    monkeypatch.setattr(factual_qa, "load_popqa_items", fake_load_popqa_items)

    cfg = make_config(tmp_path)
    cfg = dataclasses.replace(
        cfg,
        suites={**cfg.suites, "factual_qa": {"n_items": 500, "seed": 7411}},
    )
    build_items(cfg, base_dir=tmp_path)
    assert captured["args"] == (500, 7411, None)


class TokenAwareFakeLlm(FakeLlm):
    """Extends FakeLlm with create_completion for token-id items. Records
    the exact token list it was called with. Answers the truncation
    preflight (identified by its registered probe budget) honestly."""

    def create_completion(self, prompt, max_tokens=None, **kwargs):
        if max_tokens == gen_mod.TRUNCATION_PREFLIGHT_MAX_TOKENS:
            return super().create_completion(prompt, max_tokens, **kwargs)
        return {"choices": [{"text": "The passcode is 42.", "finish_reason": "stop"}]}


def test_token_id_item_roundtrips_through_items_jsonl_and_grades(tmp_path, monkeypatch):
    """Covers the Task-1 deferred gap: a token-id item (prompt_tokens set)
    must survive write -> items.jsonl -> read (reconstructed as a tuple) and
    the grade stage must actually run the longctx_retrieval grader over its
    generated record — not just parse it.
    """
    cfg = make_config(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "m-Q4_K_M.gguf").write_bytes(b"quant bytes")
    cfg.f16_path.write_bytes(b"f16 bytes")
    runs_dir = tmp_path / "runs"

    token_item = EvalItem(
        "longctx_retrieval-1-0000",
        "longctx_retrieval",
        "Alice's code?",
        ("42",),
        prompt_tokens=(11, 12, 13),
    )
    original_build_items = main_mod.build_items

    def build_items_with_token_item(config, base_dir):
        return original_build_items(config, base_dir) + [token_item]

    monkeypatch.setattr(main_mod, "build_items", build_items_with_token_item)

    # Spy on the real grader so we can assert, from inside the grade stage
    # itself, that item.prompt_tokens came back as a tuple after the
    # items.jsonl roundtrip — not just that grading happened to succeed.
    captured = {}
    original_grade = grading_mod.GRADERS["longctx_retrieval"]

    def spy_grade(item, text):
        captured["prompt_tokens"] = item.prompt_tokens
        return original_grade(item, text)

    monkeypatch.setitem(grading_mod.GRADERS, "longctx_retrieval", spy_grade)

    run_pipeline(
        cfg, run_id="token-test", models_dir=models_dir, runs_dir=runs_dir,
        stage="all", llm_factory=lambda path, gen: TokenAwareFakeLlm(), base_dir=tmp_path,
    )

    run = runs_dir / "token-test"
    items = [json.loads(l) for l in (run / "items.jsonl").read_text().splitlines()]
    on_disk = next(i for i in items if i["id"] == "longctx_retrieval-1-0000")
    assert on_disk["prompt_tokens"] == [11, 12, 13]  # JSON has no tuple type

    grades = [json.loads(l) for l in (run / "grades.jsonl").read_text().splitlines()]
    token_grades = [g for g in grades if g["item_id"] == "longctx_retrieval-1-0000"]
    assert len(token_grades) == 2  # one per ladder rung (F16, Q4_K_M)
    assert all(g["state"] == "correct" for g in token_grades)

    # the grade stage actually reconstructed prompt_tokens as a tuple, not a list
    assert captured["prompt_tokens"] == (11, 12, 13)
    assert isinstance(captured["prompt_tokens"], tuple)


def test_spectacle_only_flag_in_results(tmp_path):
    """Test that spectacle_only=True in config creates rows with spectacle_only=True."""
    cfg = make_config(tmp_path, spectacle_only=True)
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "m-Q4_K_M.gguf").write_bytes(b"quant bytes")
    cfg.f16_path.write_bytes(b"f16 bytes")
    runs_dir = tmp_path / "runs"

    run_pipeline(
        cfg, run_id="spectacle-test", models_dir=models_dir, runs_dir=runs_dir,
        stage="all", llm_factory=lambda path, gen: FakeLlm(), base_dir=tmp_path,
    )

    run = runs_dir / "spectacle-test"
    results = json.loads((run / "results.json").read_text())
    f16_rows = [r for r in results if r["quant_label"] == "F16"]
    q4_rows = [r for r in results if r["quant_label"] == "Q4_K_M"]

    # F16 should have spectacle_only=False
    assert all(r["spectacle_only"] is False for r in f16_rows)
    # Q4_K_M should have spectacle_only=True
    assert all(r["spectacle_only"] is True for r in q4_rows)
