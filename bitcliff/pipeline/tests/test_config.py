from pathlib import Path

import pytest

from bitcliff_pipeline.config import load_config


VALID_YAML = """\
model_id: qwen2.5-1.5b-instruct
hf_repo: bartowski/Qwen2.5-1.5B-Instruct-GGUF
f16_path: models/f16/Qwen2.5-1.5B-Instruct-f16.gguf
quants:
  - {label: Q8_0, filename: Qwen2.5-1.5B-Instruct-Q8_0.gguf, uploader: bartowski, imatrix: true}
  - {label: Q4_K_M, filename: Qwen2.5-1.5B-Instruct-Q4_K_M.gguf, uploader: bartowski, imatrix: true}
generation:
  seed: 42
  temperature: 0.0
  top_k: 1
  max_tokens: 640
  n_ctx: 4096
suites:
  retrieval: {n_items: 40, n_pairs: 8, seed: 1301}
  arithmetic: {n_items: 40, seed: 1301}
  spectacle: {path: configs/prompts_spectacle.yaml}
"""


def write_yaml(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "ladder.yaml"
    p.write_text(text)
    return p


def test_loads_valid_config(tmp_path):
    cfg = load_config(write_yaml(tmp_path, VALID_YAML))
    assert cfg.model_id == "qwen2.5-1.5b-instruct"
    assert cfg.hf_repo == "bartowski/Qwen2.5-1.5B-Instruct-GGUF"
    assert cfg.f16_path == Path("models/f16/Qwen2.5-1.5B-Instruct-f16.gguf")
    assert [q.label for q in cfg.quants] == ["Q8_0", "Q4_K_M"]
    assert cfg.quants[0].uploader == "bartowski"
    assert cfg.quants[0].imatrix is True
    assert cfg.generation.seed == 42
    assert cfg.generation.temperature == 0.0
    assert cfg.suites["retrieval"]["n_pairs"] == 8


def test_rejects_duplicate_quant_labels(tmp_path):
    bad = VALID_YAML.replace("label: Q4_K_M", "label: Q8_0")
    with pytest.raises(ValueError, match="duplicate"):
        load_config(write_yaml(tmp_path, bad))


def test_rejects_missing_required_key(tmp_path):
    bad = VALID_YAML.replace("hf_repo: bartowski/Qwen2.5-1.5B-Instruct-GGUF\n", "")
    with pytest.raises(KeyError):
        load_config(write_yaml(tmp_path, bad))


def test_spectacle_only_defaults_false_and_parses(tmp_path):
    yaml_text = VALID_YAML.replace(
        "  - {label: Q4_K_M, filename: Qwen2.5-1.5B-Instruct-Q4_K_M.gguf, uploader: bartowski, imatrix: true}",
        "  - {label: IQ1_S, filename: x-IQ1_S.gguf, uploader: bitcliff-inhouse, imatrix: true, spectacle_only: true}",
    )
    cfg = load_config(write_yaml(tmp_path, yaml_text))
    assert cfg.quants[0].spectacle_only is False   # omitted -> default
    assert cfg.quants[1].spectacle_only is True
