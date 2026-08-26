import json
from pathlib import Path

import pytest

from bitcliff_pipeline.__main__ import build_items, run_pipeline
from bitcliff_pipeline.config import GenSettings, LadderConfig, QuantFile


def make_config(tmp_path: Path) -> LadderConfig:
    spectacle = tmp_path / "configs" / "prompts_spectacle.yaml"
    spectacle.parent.mkdir(parents=True)
    spectacle.write_text("prompts:\n  - {id: spec-001, prompt: 'Say hi.'}\n")
    return LadderConfig(
        model_id="test-model",
        hf_repo="fake/repo",
        f16_path=tmp_path / "models" / "f16.gguf",
        quants=(QuantFile("Q4_K_M", "m-Q4_K_M.gguf", "bartowski", True),),
        generation=GenSettings(42, 0.0, 1, 640, 4096),
        suites={
            "retrieval": {"n_items": 4, "n_pairs": 3, "seed": 1},
            "spectacle": {"path": "configs/prompts_spectacle.yaml"},
        },
    )


class FakeLlm:
    """Echoes the expected answer for retrieval items by reading the context."""

    def create_chat_completion(self, messages, **kwargs):
        prompt = messages[0]["content"]
        # answer with the whole context: retrieval grader will find both codes
        return {"choices": [{"message": {"content": prompt}, "finish_reason": "stop"}]}


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
    # 4 retrieval + 1 spectacle, for 2 ladder rungs
    assert len(grades) == 10
    assert {g["quant_label"] for g in grades} == {"F16", "Q4_K_M"}
    assert all("divergence" in g for g in grades)
    results = json.loads((run / "results.json").read_text())
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
    assert suites.count("retrieval") == 4
    assert suites.count("spectacle") == 1
