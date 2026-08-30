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
  longctx_retrieval: {n_items: 40, seed: 1301}
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
    assert cfg.suites["longctx_retrieval"]["n_items"] == 40


def test_rejects_duplicate_quant_labels(tmp_path):
    bad = VALID_YAML.replace("label: Q4_K_M", "label: Q8_0")
    with pytest.raises(ValueError, match="duplicate"):
        load_config(write_yaml(tmp_path, bad))


def test_rejects_missing_required_key(tmp_path):
    bad = VALID_YAML.replace("hf_repo: bartowski/Qwen2.5-1.5B-Instruct-GGUF\n", "")
    with pytest.raises(KeyError):
        load_config(write_yaml(tmp_path, bad))


def test_suites_max_tokens_override_is_reachable(tmp_path):
    yaml_text = VALID_YAML.replace(
        "  longctx_retrieval: {n_items: 40, seed: 1301}\n",
        "  longctx_retrieval: {n_items: 10, seed: 1301, max_tokens: 32}\n",
    )
    cfg = load_config(write_yaml(tmp_path, yaml_text))
    assert cfg.suites["longctx_retrieval"]["max_tokens"] == 32


def test_spectacle_only_defaults_false_and_parses(tmp_path):
    yaml_text = VALID_YAML.replace(
        "  - {label: Q4_K_M, filename: Qwen2.5-1.5B-Instruct-Q4_K_M.gguf, uploader: bartowski, imatrix: true}",
        "  - {label: IQ1_S, filename: x-IQ1_S.gguf, uploader: bitcliff-inhouse, imatrix: true, spectacle_only: true}",
    )
    cfg = load_config(write_yaml(tmp_path, yaml_text))
    assert cfg.quants[0].spectacle_only is False   # omitted -> default
    assert cfg.quants[1].spectacle_only is True


# ---------------------------------------------------------------------------
# 0B P3: config-pinned sha256 / per-quant hf_repo override / extra_files
# (sharded GGUFs) -- all optional, default to None/None/() when omitted so
# every pre-0B config keeps loading unchanged.
# ---------------------------------------------------------------------------


def test_quant_sha256_and_hf_repo_default_to_none(tmp_path):
    cfg = load_config(write_yaml(tmp_path, VALID_YAML))
    assert cfg.quants[0].sha256 is None
    assert cfg.quants[0].hf_repo is None
    assert cfg.quants[0].extra_files == ()


def test_quant_sha256_and_hf_repo_override_parse(tmp_path):
    yaml_text = VALID_YAML.replace(
        "  - {label: Q8_0, filename: Qwen2.5-1.5B-Instruct-Q8_0.gguf, uploader: bartowski, imatrix: true}",
        "  - {label: Q8_0, filename: Qwen2.5-1.5B-Instruct-Q8_0.gguf, uploader: bartowski, imatrix: true, "
        "sha256: 'aa' , hf_repo: other/repo}",
    )
    cfg = load_config(write_yaml(tmp_path, yaml_text))
    assert cfg.quants[0].sha256 == "aa"
    assert cfg.quants[0].hf_repo == "other/repo"


def test_quant_extra_files_parse(tmp_path):
    yaml_text = VALID_YAML.replace(
        "  - {label: Q8_0, filename: Qwen2.5-1.5B-Instruct-Q8_0.gguf, uploader: bartowski, imatrix: true}",
        "  - {label: Q8_0, filename: shard1.gguf, uploader: qwen-official, imatrix: false, "
        "extra_files: [{filename: shard2.gguf, sha256: 'bb'}]}",
    )
    cfg = load_config(write_yaml(tmp_path, yaml_text))
    assert len(cfg.quants[0].extra_files) == 1
    assert cfg.quants[0].extra_files[0].filename == "shard2.gguf"
    assert cfg.quants[0].extra_files[0].sha256 == "bb"
