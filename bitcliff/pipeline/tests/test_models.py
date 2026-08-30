import hashlib
from pathlib import Path

import pytest

from bitcliff_pipeline.config import ExtraFile, GenSettings, LadderConfig, QuantFile
from bitcliff_pipeline.hashing import ManifestMismatch
from bitcliff_pipeline.models import ensure_quants, resolve_all


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


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


# ---------------------------------------------------------------------------
# 0B P3: config-pinned sha256 (ties a quant file to the committed
# reference-manifests/*.json, distinct from hashing.verify_manifest's
# run-local check).
# ---------------------------------------------------------------------------


def test_ensure_quants_passes_when_downloaded_file_matches_config_sha256(tmp_path):
    content = b"downloaded bytes"
    cfg = LadderConfig(
        model_id="test-model",
        hf_repo="fake/repo",
        f16_path=tmp_path / "f16.gguf",
        quants=(QuantFile("Q8_0", "m-Q8_0.gguf", "bartowski", True, sha256=_sha256_bytes(content)),),
        generation=GenSettings(42, 0.0, 1, 640, 4096),
        suites={},
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    def fake_downloader(repo_id, filename, local_dir):
        Path(local_dir, filename).write_bytes(content)

    paths = ensure_quants(cfg, models_dir, downloader=fake_downloader)
    assert paths["Q8_0"].read_bytes() == content


def test_ensure_quants_raises_on_config_sha256_mismatch(tmp_path):
    cfg = LadderConfig(
        model_id="test-model",
        hf_repo="fake/repo",
        f16_path=tmp_path / "f16.gguf",
        quants=(QuantFile("Q8_0", "m-Q8_0.gguf", "bartowski", True, sha256="0" * 64),),
        generation=GenSettings(42, 0.0, 1, 640, 4096),
        suites={},
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    def fake_downloader(repo_id, filename, local_dir):
        Path(local_dir, filename).write_bytes(b"downloaded bytes")

    with pytest.raises(ManifestMismatch, match="m-Q8_0.gguf"):
        ensure_quants(cfg, models_dir, downloader=fake_downloader)


def test_ensure_quants_verifies_already_present_file_against_config_sha256(tmp_path):
    """A file already on disk (skip-download path) still gets checked
    against the pin -- catching a stale/wrong file left over from a prior
    session, not just a freshly-downloaded one."""
    cfg = LadderConfig(
        model_id="test-model",
        hf_repo="fake/repo",
        f16_path=tmp_path / "f16.gguf",
        quants=(QuantFile("Q8_0", "m-Q8_0.gguf", "bartowski", True, sha256="0" * 64),),
        generation=GenSettings(42, 0.0, 1, 640, 4096),
        suites={},
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m-Q8_0.gguf").write_bytes(b"stale bytes from a re-quantized upload")

    with pytest.raises(ManifestMismatch, match="m-Q8_0.gguf"):
        ensure_quants(cfg, models_dir, downloader=lambda **kw: None)


def test_ensure_quants_no_sha256_pin_skips_verification(tmp_path):
    cfg = make_config(tmp_path)  # QuantFile.sha256 defaults to None
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    def fake_downloader(repo_id, filename, local_dir):
        Path(local_dir, filename).write_bytes(b"anything at all")

    ensure_quants(cfg, models_dir, downloader=fake_downloader)  # no raise


# ---------------------------------------------------------------------------
# 0B P3: per-quant hf_repo override (RUN_0B.md's Arm 1 shootout spans three
# repos -- unsloth, mradermacher static, mradermacher i1 -- inside one
# config whose top-level hf_repo can only name one of them).
# ---------------------------------------------------------------------------


def test_ensure_quants_uses_per_quant_hf_repo_override(tmp_path):
    cfg = LadderConfig(
        model_id="test-model",
        hf_repo="default/repo",
        f16_path=tmp_path / "f16.gguf",
        quants=(
            QuantFile("unsloth_Q4_K_M", "u-Q4_K_M.gguf", "unsloth", False, hf_repo="unsloth/repo"),
            QuantFile("bartowski_Q4_K_M", "b-Q4_K_M.gguf", "bartowski", True),
        ),
        generation=GenSettings(42, 0.0, 1, 640, 4096),
        suites={},
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    calls = []

    def fake_downloader(repo_id, filename, local_dir):
        calls.append((repo_id, filename))
        Path(local_dir, filename).write_bytes(b"x")

    ensure_quants(cfg, models_dir, downloader=fake_downloader)
    assert ("unsloth/repo", "u-Q4_K_M.gguf") in calls
    assert ("default/repo", "b-Q4_K_M.gguf") in calls


# ---------------------------------------------------------------------------
# 0B P3: extra_files (sharded GGUFs -- e.g. Qwen official q4_k_m, 2 shards).
# `resolve_all`/generation only ever load the primary `filename` (llama-cpp
# discovers sibling shards itself from the first shard's path); extra_files
# exist so the config records + verifies every shard's provenance.
# ---------------------------------------------------------------------------


def test_ensure_quants_downloads_and_verifies_extra_shard_files(tmp_path):
    shard2 = b"second shard bytes"
    cfg = LadderConfig(
        model_id="test-model",
        hf_repo="qwen/official",
        f16_path=tmp_path / "f16.gguf",
        quants=(
            QuantFile(
                "Q4_K_M", "q-00001-of-00002.gguf", "qwen-official", False,
                extra_files=(ExtraFile("q-00002-of-00002.gguf", _sha256_bytes(shard2)),),
            ),
        ),
        generation=GenSettings(42, 0.0, 1, 640, 4096),
        suites={},
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    calls = []

    def fake_downloader(repo_id, filename, local_dir):
        calls.append((repo_id, filename))
        content = shard2 if "00002" in filename else b"first shard bytes"
        Path(local_dir, filename).write_bytes(content)

    paths = ensure_quants(cfg, models_dir, downloader=fake_downloader)
    # resolve_all/generation load only the primary (first) shard path
    assert paths["Q4_K_M"] == models_dir / "q-00001-of-00002.gguf"
    assert set(calls) == {
        ("qwen/official", "q-00001-of-00002.gguf"),
        ("qwen/official", "q-00002-of-00002.gguf"),
    }
    assert (models_dir / "q-00002-of-00002.gguf").exists()


def test_ensure_quants_raises_on_extra_shard_sha256_mismatch(tmp_path):
    cfg = LadderConfig(
        model_id="test-model",
        hf_repo="qwen/official",
        f16_path=tmp_path / "f16.gguf",
        quants=(
            QuantFile(
                "Q4_K_M", "q-00001-of-00002.gguf", "qwen-official", False,
                extra_files=(ExtraFile("q-00002-of-00002.gguf", "0" * 64),),
            ),
        ),
        generation=GenSettings(42, 0.0, 1, 640, 4096),
        suites={},
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    def fake_downloader(repo_id, filename, local_dir):
        Path(local_dir, filename).write_bytes(b"whatever")

    with pytest.raises(ManifestMismatch, match="q-00002-of-00002.gguf"):
        ensure_quants(cfg, models_dir, downloader=fake_downloader)
