import json
import re
from pathlib import Path

import pytest

from bitcliff_pipeline.__main__ import build_items, run_pipeline
from bitcliff_pipeline.config import GenSettings, LadderConfig, QuantFile
from bitcliff_pipeline.suites import arithmetic

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


def make_config(tmp_path: Path) -> LadderConfig:
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
        quants=(QuantFile("Q4_K_M", "m-Q4_K_M.gguf", "bartowski", True),),
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

    def create_chat_completion(self, messages, **kwargs):
        prompt = messages[0]["content"]
        m = self._ARITH_RE.search(prompt)
        content = f"#### {int(m.group(1)) + int(m.group(2))}" if m else prompt
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}


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


def test_build_items_covers_configured_suites(tmp_path):
    cfg = make_config(tmp_path)
    items = build_items(cfg, base_dir=tmp_path)
    suites = [i.suite for i in items]
    assert suites.count("arithmetic") == 4
    assert suites.count("spectacle") == 2
