from pathlib import Path

import pytest

from bitcliff_pipeline.config import GenSettings, LadderConfig, QuantFile
from bitcliff_pipeline.models import ensure_quants, resolve_all


def make_config(tmp_path: Path) -> LadderConfig:
    return LadderConfig(
        model_id="test-model",
        hf_repo="fake/repo",
        f16_path=tmp_path / "f16.gguf",
        quants=(
            QuantFile("Q8_0", "m-Q8_0.gguf", "bartowski", True),
            QuantFile("Q4_K_M", "m-Q4_K_M.gguf", "bartowski", True),
        ),
        generation=GenSettings(42, 0.0, 1, 640, 4096),
        suites={},
    )


def test_ensure_quants_downloads_missing_and_skips_existing(tmp_path):
    cfg = make_config(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m-Q8_0.gguf").write_bytes(b"already here")
    calls = []

    def fake_downloader(repo_id, filename, local_dir):
        calls.append((repo_id, filename))
        Path(local_dir, filename).write_bytes(b"downloaded")

    paths = ensure_quants(cfg, models_dir, downloader=fake_downloader)
    assert calls == [("fake/repo", "m-Q4_K_M.gguf")]
    assert paths["Q8_0"].read_bytes() == b"already here"
    assert paths["Q4_K_M"].read_bytes() == b"downloaded"


def test_resolve_all_includes_f16(tmp_path):
    cfg = make_config(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for q in cfg.quants:
        (models_dir / q.filename).write_bytes(b"x")
    cfg.f16_path.write_bytes(b"f16 bytes")
    paths = resolve_all(cfg, models_dir)
    assert set(paths) == {"F16", "Q8_0", "Q4_K_M"}
    assert paths["F16"] == cfg.f16_path


def test_resolve_all_raises_when_f16_missing(tmp_path):
    cfg = make_config(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for q in cfg.quants:
        (models_dir / q.filename).write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="F16"):
        resolve_all(cfg, models_dir)
