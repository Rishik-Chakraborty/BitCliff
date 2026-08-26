import hashlib

import pytest

from bitcliff_pipeline.hashing import (
    ManifestMismatch,
    build_manifest,
    load_manifest,
    sha256_file,
    verify_manifest,
    write_manifest,
)


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "model.gguf"
    p.write_bytes(b"fake gguf bytes")
    assert sha256_file(p) == hashlib.sha256(b"fake gguf bytes").hexdigest()


def test_build_write_load_roundtrip(tmp_path):
    p = tmp_path / "a.gguf"
    p.write_bytes(b"aaaa")
    manifest = build_manifest({"Q8_0": p})
    assert manifest["Q8_0"]["filename"] == "a.gguf"
    assert manifest["Q8_0"]["size_bytes"] == 4
    out = tmp_path / "manifest.json"
    write_manifest(manifest, out)
    assert load_manifest(out) == manifest


def test_verify_manifest_passes_on_same_bytes(tmp_path):
    p = tmp_path / "a.gguf"
    p.write_bytes(b"aaaa")
    manifest = build_manifest({"Q8_0": p})
    verify_manifest(manifest, {"Q8_0": p})  # no raise


def test_verify_manifest_raises_on_changed_file(tmp_path):
    p = tmp_path / "a.gguf"
    p.write_bytes(b"aaaa")
    manifest = build_manifest({"Q8_0": p})
    p.write_bytes(b"bbbb")
    with pytest.raises(ManifestMismatch, match="Q8_0"):
        verify_manifest(manifest, {"Q8_0": p})
