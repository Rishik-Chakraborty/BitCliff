"""Tests for scripts/package_dataset.py — PREREG §11 dataset packager.

All synthetic run data lives under tmp_path; no models, no network. Loaded
by file path (same pattern as tests/test_calibrate_f16.py): the script has
no package `__init__.py`.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "package_dataset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("package_dataset", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # dataclasses needs the module registered in sys.modules (it looks up
    # cls.__module__ during class creation) before exec_module runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pkg = _load_module()


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

MANIFEST = {
    "F16": {
        "filename": "Fake-Instruct-f16.gguf",
        "imatrix": False,
        "sha256": "f16" + "0" * 61,
        "size_bytes": 1000,
        "spectacle_only": False,
        "uploader": "bitcliff-local-f16-conversion",
    },
    "Q4_K_M": {
        "filename": "Fake-Instruct-Q4_K_M.gguf",
        "imatrix": True,
        "sha256": "q4" + "0" * 62,
        "size_bytes": 500,
        "spectacle_only": False,
        "uploader": "bartowski",
    },
}

SENTINEL_PROMPT_TEXT = "SENTINEL_DOCUMENT_TOKEN_98efbe3c_do_not_publish"
SENTINEL_TOKENS = [11111, 22222, 33333, 44444, 55555]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _items():
    return [
        {
            "id": "retrieval-1301-000",
            "suite": "retrieval",
            "prompt": "retired placeholder prompt text",
            "expected": ["1234", "5678"],
        },
        {
            "id": "arithmetic-3141-000",
            "suite": "arithmetic",
            "prompt": "If Alice has 2 apples and buys 3 more, how many apples?",
            "expected": ["5"],
        },
        {
            "id": "arithmetic-3141-001",
            "suite": "arithmetic",
            "prompt": "A train travels 60 miles in 2 hours. Speed in mph?",
            "expected": ["30"],
        },
        {
            "id": "spec-001",
            "suite": "spectacle",
            "prompt": "Tell me a haiku about compilers.",
            "expected": None,
        },
        {
            "id": "longctx_retrieval-fakevariant-t100-s999-0000",
            "suite": "longctx_retrieval",
            "prompt": f"What is the secret passcode? {SENTINEL_PROMPT_TEXT}",
            "expected": ["4242"],
            "prompt_tokens": SENTINEL_TOKENS,
        },
    ]


def _items_with_twins():
    return _items() + [
        {
            "id": "arithmetic_twins-1301-orig-000",
            "suite": "arithmetic_twins",
            "prompt": "embargoed twin prompt",
            "expected": ["7"],
        }
    ]


def _outputs_for(items: list[dict], quant_label: str) -> list[dict]:
    records = []
    for item in items:
        records.append(
            {
                "item_id": item["id"],
                "suite": item["suite"],
                "quant_label": quant_label,
                "model_sha256": MANIFEST[quant_label]["sha256"],
                "prompt": item["prompt"] if item["suite"] != "longctx_retrieval" else item["prompt"],
                "text": f"answer-from-{quant_label}-for-{item['id']}",
                "finish_reason": "stop",
                "gen_settings": {"seed": 42, "temperature": 0.0, "top_k": 1, "max_tokens": 64},
                "machine": "test-machine",
            }
        )
    return records


def _grades_for(items: list[dict], quant_label: str) -> list[dict]:
    return [
        {
            "item_id": item["id"],
            "suite": item["suite"],
            "quant_label": quant_label,
            "state": "correct",
            "truncated": False,
            "loop": False,
            "divergence": None,
        }
        for item in items
    ]


def _build_run_dir(tmp_path: Path, name: str, items: list[dict]) -> Path:
    run_dir = tmp_path / "runs" / name
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps(MANIFEST, sort_keys=True))
    _write_jsonl(run_dir / "items.jsonl", items)

    grades = []
    for label in MANIFEST:
        _write_jsonl(run_dir / "outputs" / f"{label}.jsonl", _outputs_for(items, label))
        grades += _grades_for(items, label)
    _write_jsonl(run_dir / "grades.jsonl", grades)
    return run_dir


# ---------------------------------------------------------------------------
# Twins refusal
# ---------------------------------------------------------------------------


def test_twins_suite_raises_and_writes_nothing(tmp_path):
    run_dir = _build_run_dir(tmp_path, "with-twins", _items_with_twins())
    out_dir = tmp_path / "dist" / "dataset-with-twins"

    with pytest.raises(pkg.EmbargoViolation, match="arithmetic_twins"):
        pkg.package(run_dir, out_dir)

    assert not out_dir.exists()


def test_private_path_in_run_dir_raises(tmp_path):
    run_dir = _build_run_dir(tmp_path, "with-private", _items())
    (run_dir / "private" / "twins").mkdir(parents=True)
    (run_dir / "private" / "twins" / "secret.jsonl").write_text("{}\n")
    out_dir = tmp_path / "dist" / "dataset-with-private"

    with pytest.raises(pkg.EmbargoViolation, match="private"):
        pkg.package(run_dir, out_dir)

    assert not out_dir.exists()


# ---------------------------------------------------------------------------
# Full (non-twins) packaging run
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_run_dir(tmp_path):
    return _build_run_dir(tmp_path, "clean-run", _items())


def test_longctx_prompt_and_tokens_absent_from_every_produced_byte(tmp_path, clean_run_dir):
    out_dir = tmp_path / "dist" / "dataset-clean-run"
    pkg.package(clean_run_dir, out_dir)

    forbidden = [
        SENTINEL_PROMPT_TEXT,
        json.dumps(SENTINEL_TOKENS),
        ",".join(str(t) for t in SENTINEL_TOKENS),
    ]
    produced_files = [p for p in out_dir.rglob("*") if p.is_file()]
    assert produced_files, "packaging produced no files"
    for path in produced_files:
        content = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in content, f"{path} leaked embargoed longctx content"


def test_longctx_record_still_published_with_id_and_outputs(tmp_path, clean_run_dir):
    out_dir = tmp_path / "dist" / "dataset-clean-run"
    pkg.package(clean_run_dir, out_dir)

    lines = (out_dir / "longctx_retrieval.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["id"] == "longctx_retrieval-fakevariant-t100-s999-0000"
    assert "prompt" not in rec
    assert "prompt_tokens" not in rec
    assert "expected" not in rec
    assert set(rec["rungs"].keys()) == {"F16", "Q4_K_M"}
    assert rec["rungs"]["F16"]["text"] == "answer-from-F16-for-longctx_retrieval-fakevariant-t100-s999-0000"
    assert rec["rungs"]["F16"]["uploader"] == "bitcliff-local-f16-conversion"
    assert rec["rungs"]["Q4_K_M"]["sha256"] == MANIFEST["Q4_K_M"]["sha256"]


def test_arithmetic_and_spectacle_prompts_are_published(tmp_path, clean_run_dir):
    out_dir = tmp_path / "dist" / "dataset-clean-run"
    pkg.package(clean_run_dir, out_dir)

    arithmetic_lines = (out_dir / "arithmetic.jsonl").read_text().splitlines()
    assert len(arithmetic_lines) == 2
    for line in arithmetic_lines:
        rec = json.loads(line)
        assert rec["prompt"]
        assert rec["expected"]
        assert set(rec["rungs"].keys()) == {"F16", "Q4_K_M"}

    spectacle_lines = (out_dir / "spectacle.jsonl").read_text().splitlines()
    assert len(spectacle_lines) == 1
    spec_rec = json.loads(spectacle_lines[0])
    assert spec_rec["prompt"] == "Tell me a haiku about compilers."


def test_retired_retrieval_suite_excluded(tmp_path, clean_run_dir):
    out_dir = tmp_path / "dist" / "dataset-clean-run"
    stats = pkg.package(clean_run_dir, out_dir)

    assert not (out_dir / "retrieval.jsonl").exists()
    assert "retrieval" not in stats["suite_record_counts"]
    reasons = {e.get("suite") or e.get("field"): e["reason"] for e in stats["exclusions"]}
    assert "retrieval" in reasons
    assert "superseded" in reasons["retrieval"]


def test_manifest_counts_and_exclusions_correct(tmp_path, clean_run_dir):
    out_dir = tmp_path / "dist" / "dataset-clean-run"
    pkg.package(clean_run_dir, out_dir)

    manifest = json.loads((out_dir / "dataset-manifest.json").read_text())
    assert manifest["suite_record_counts"] == {
        "arithmetic": 2,
        "spectacle": 1,
        "longctx_retrieval": 1,
    }
    assert manifest["prereg_commit"] == pkg.PREREG_COMMIT
    excluded_targets = {e.get("suite") or e.get("field") for e in manifest["exclusions"]}
    assert "retrieval" in excluded_targets
    assert any("longctx_retrieval" in t for t in excluded_targets)
    assert "longctx_recipe" in manifest
    assert manifest["longctx_recipe"]["verification_digest"]
    assert (out_dir / "RECONSTRUCTION.md").exists()
    assert (out_dir / "PACKAGED_AT").exists()


def test_deterministic_double_run_byte_identical(tmp_path, clean_run_dir):
    out_dir_1 = tmp_path / "dist" / "dataset-run-1"
    out_dir_2 = tmp_path / "dist" / "dataset-run-2"
    pkg.package(clean_run_dir, out_dir_1)
    pkg.package(clean_run_dir, out_dir_2)

    files_1 = sorted(p.relative_to(out_dir_1) for p in out_dir_1.rglob("*") if p.is_file())
    files_2 = sorted(p.relative_to(out_dir_2) for p in out_dir_2.rglob("*") if p.is_file())
    assert files_1 == files_2

    for rel in files_1:
        if rel.name == "PACKAGED_AT":
            # PACKAGED_AT deliberately carries the generation date and is
            # excluded from the determinism guarantee (PREREG §11 task).
            continue
        assert (out_dir_1 / rel).read_bytes() == (out_dir_2 / rel).read_bytes(), (
            f"{rel} differs between two packaging runs of the same input"
        )


def test_2a_signature_run_is_refused(tmp_path):
    items = _items()[:1]  # keep retired suite out of the way, arbitrary base
    items = [
        {
            "id": "arithmetic-3141-000",
            "suite": "arithmetic",
            "prompt": "2+2?",
            "expected": ["4"],
        },
        {
            "id": "longctx_retrieval-multivalue2-t4096-s2024-0000",
            "suite": "longctx_retrieval",
            "prompt": "What is the secret passcode?",
            "expected": ["9999"],
            "prompt_tokens": [1, 2, 3],
        },
    ]
    run_dir = _build_run_dir(tmp_path, "config-2a-shaped", items)
    out_dir = tmp_path / "dist" / "dataset-config-2a-shaped"

    with pytest.raises(pkg.EmbargoViolation, match="config 2a"):
        pkg.package(run_dir, out_dir)

    assert not out_dir.exists()


def test_no_run_writes_no_output_on_twins_even_if_out_dir_preexists(tmp_path):
    run_dir = _build_run_dir(tmp_path, "with-twins-2", _items_with_twins())
    out_dir = tmp_path / "dist" / "dataset-with-twins-2"
    out_dir.mkdir(parents=True)
    (out_dir / "stale.txt").write_text("leftover from a previous, unrelated build\n")

    with pytest.raises(pkg.EmbargoViolation):
        pkg.package(run_dir, out_dir)

    # refusal happens before out_dir is touched at all
    assert (out_dir / "stale.txt").exists()
