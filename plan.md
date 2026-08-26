# BitCliff Phase 0A (Pilot) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the config-driven evaluation pipeline and run the exploratory Qwen2.5-1.5B quantization ladder end-to-end on local hardware, producing the retention curves that decide whether BitCliff exists.

**Architecture:** A single Python package (`bitcliff/pipeline/`) with pure, file-based stages: download quant files → generate deterministic outputs per quant level → grade outputs mechanically → aggregate into retention curves and CSV/JSON reports. Every stage reads and writes plain files (YAML config in, JSONL/CSV/JSON/PNG out); there are no services or databases. Inference runs through `llama-cpp-python` (Metal on this Mac), injected as a factory so every module is testable without a model.

**Tech Stack:** Python 3.11+, `uv`, `pytest`, `llama-cpp-python`, `huggingface_hub`, `datasets`, `PyYAML`, `matplotlib`.

**Spec:** `claude/IDEA.md` (the full product spec). This plan implements its Section 18 ("First move: 0A") and the pipeline pieces of Sections 4, 6, and 10. Per the spec, everything beyond the pilot is gated on the pilot's curves; later phases get their own plans (see "Out of scope" at the bottom).

## Global Constraints

Copied from the spec and `bitcliff/.claude/skills/aws-ops/SKILL.MD`; every task's requirements implicitly include these.

- **The pilot is exploratory.** Its numbers are thrown away; they inform curation and parameter choices only. Nothing confirmatory runs before the registration freeze (spec §10, "0A"/"The freeze").
- **Zero cloud spend.** Phase 0A runs on own hardware only. This plan must not create any AWS resource. ("no GPU money is spent before it either" — spec §10; aws-ops: stop and ask before any cash spend over $50.)
- **Determinism, honestly scoped.** All generation uses `temperature=0.0`, `top_k=1`, fixed `seed=42`, and every relevant setting plus a machine descriptor is recorded in every output record. Determinism is claimed only within this one hardware+software configuration (spec §6).
- **A quant level is a file, not a label.** Canonical ladder = bartowski's imatrix quants. Every artifact records uploader, exact filename, and SHA-256 hash. Results are pinned to hashes (spec §4).
- **F16 is the reference**, converted locally from the HF safetensors, and is never a download recommendation (spec §4).
- **Grade vocabulary:** `correct` / `partial` / `wrong` / `unscored`, plus separate mechanical flags `truncated` and `loop`. Truncation counts as wrong for accuracy but is always tracked and reported separately — "wrong" never silently absorbs "cut off" (spec §6).
- **Twin GSM8K problems are NOT in this plan.** They are built at the freeze, stay embargoed, and never enter the pilot (spec §6).
- **Pilot ladder (verified against the live HF repo `bartowski/Qwen2.5-1.5B-Instruct-GGUF` on 2026-08-26):** Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K, IQ2_M. The repo publishes nothing below IQ2_M, so the spec's "in-house bottom rungs for spectacle" exception applies — deferred to the runbook as an optional follow-up, not launch-pipeline work.
- **Tooling:** `uv` for env/deps, `pytest` for tests, TDD per task, commit after every green task.

## File Structure

```
bitcliff/pipeline/
├── pyproject.toml                      # uv project; deps + pytest config
├── configs/
│   ├── qwen2.5-1.5b-pilot.yaml         # the pilot ladder config (Task 1)
│   └── prompts_spectacle.yaml          # curated unscored spectacle prompts (Task 5)
├── src/bitcliff_pipeline/
│   ├── __init__.py
│   ├── __main__.py                     # CLI orchestrator: download|generate|grade|report|all (Task 10)
│   ├── config.py                       # YAML → typed LadderConfig (Task 1)
│   ├── items.py                        # EvalItem shared dataclass (Task 1)
│   ├── hashing.py                      # SHA-256 manifest build/write/verify (Task 2)
│   ├── models.py                       # HF quant download, path resolution (Task 3)
│   ├── generate.py                     # deterministic llama.cpp runner, JSONL records (Task 6)
│   ├── grading.py                      # grade dispatch + loop/truncation flags (Task 7)
│   ├── divergence.py                   # first-divergent-token vs F16 (Task 8)
│   ├── report.py                       # aggregates, retention, CSV/JSON/PNG (Task 9)
│   └── suites/
│       ├── __init__.py
│       ├── retrieval.py                # multivalue2-style key→two-values recall (Task 4)
│       ├── arithmetic.py               # GSM8K subset + numeric grading (Task 5)
│       └── spectacle.py                # curated prompts, always unscored (Task 5)
├── tests/
│   ├── test_config.py
│   ├── test_hashing.py
│   ├── test_models.py
│   ├── test_retrieval.py
│   ├── test_arithmetic.py
│   ├── test_spectacle.py
│   ├── test_generate.py
│   ├── test_grading.py
│   ├── test_divergence.py
│   ├── test_report.py
│   └── test_cli.py
├── models/                             # gitignored: downloaded ggufs + local F16
├── runs/                               # gitignored: per-run JSONL outputs, grades, reports
└── PILOT_RUNBOOK.md                    # the manual weekend procedure (Task 11)
```

Data flow between stages (all file-based):

```
configs/*.yaml ──> download ──> models/*.gguf + runs/<id>/manifest.json
                └> generate ──> runs/<id>/outputs/<QUANT>.jsonl
                └> grade    ──> runs/<id>/grades.jsonl (+ divergence fields)
                └> report   ──> runs/<id>/results.csv|results.json|retention.png
```

---

### Task 1: Project scaffold, config loader, and shared item type

**Files:**
- Create: `bitcliff/pipeline/pyproject.toml`
- Create: `bitcliff/pipeline/.gitignore`
- Create: `bitcliff/pipeline/src/bitcliff_pipeline/__init__.py`
- Create: `bitcliff/pipeline/src/bitcliff_pipeline/config.py`
- Create: `bitcliff/pipeline/src/bitcliff_pipeline/items.py`
- Create: `bitcliff/pipeline/configs/qwen2.5-1.5b-pilot.yaml`
- Test: `bitcliff/pipeline/tests/test_config.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `config.load_config(path: str | Path) -> LadderConfig`
  - `LadderConfig(model_id: str, hf_repo: str, f16_path: Path, quants: tuple[QuantFile, ...], generation: GenSettings, suites: dict)`
  - `QuantFile(label: str, filename: str, uploader: str, imatrix: bool)`
  - `GenSettings(seed: int, temperature: float, top_k: int, max_tokens: int, n_ctx: int)`
  - `items.EvalItem(id: str, suite: str, prompt: str, expected: tuple[str, ...] | None)` — frozen dataclass used by every suite, the generator, and grading.

- [ ] **Step 1: Initialize the uv project**

```bash
cd bitcliff/pipeline
uv init --package --name bitcliff-pipeline --python 3.11
uv add pyyaml huggingface_hub datasets matplotlib llama-cpp-python
uv add --dev pytest
```

Note: `llama-cpp-python` compiles llama.cpp on install (a few minutes on first build; Metal is enabled by default on macOS). If the build fails, continue — it is only needed at Task 6 and the runbook; retry there with `CMAKE_ARGS="-DGGML_METAL=on" uv add llama-cpp-python`.

- [ ] **Step 2: Write `.gitignore`**

```gitignore
models/
runs/
__pycache__/
*.egg-info/
.venv/
```

- [ ] **Step 3: Add pytest config to `pyproject.toml`** (append)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Write the failing tests**

`tests/test_config.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bitcliff_pipeline.config'`

- [ ] **Step 6: Write the implementation**

`src/bitcliff_pipeline/items.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalItem:
    """One evaluation prompt. expected is None for unscored (spectacle) items."""

    id: str
    suite: str
    prompt: str
    expected: tuple[str, ...] | None
```

`src/bitcliff_pipeline/config.py`:

```python
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class QuantFile:
    label: str
    filename: str
    uploader: str
    imatrix: bool


@dataclass(frozen=True)
class GenSettings:
    seed: int
    temperature: float
    top_k: int
    max_tokens: int
    n_ctx: int


@dataclass(frozen=True)
class LadderConfig:
    model_id: str
    hf_repo: str
    f16_path: Path
    quants: tuple[QuantFile, ...]
    generation: GenSettings
    suites: dict


def load_config(path: str | Path) -> LadderConfig:
    raw = yaml.safe_load(Path(path).read_text())
    quants = tuple(QuantFile(**q) for q in raw["quants"])
    labels = [q.label for q in quants]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate quant labels in {path}")
    return LadderConfig(
        model_id=raw["model_id"],
        hf_repo=raw["hf_repo"],
        f16_path=Path(raw["f16_path"]),
        quants=quants,
        generation=GenSettings(**raw["generation"]),
        suites=raw["suites"],
    )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 3 PASS

- [ ] **Step 8: Write the real pilot config**

`configs/qwen2.5-1.5b-pilot.yaml` — filenames below were verified against the live HF repo on 2026-08-26; the ladder is bartowski's imatrix set, top to bottom, and F16 is the locally converted reference:

```yaml
model_id: qwen2.5-1.5b-instruct
hf_repo: bartowski/Qwen2.5-1.5B-Instruct-GGUF
f16_path: models/f16/Qwen2.5-1.5B-Instruct-f16.gguf
quants:
  - {label: Q8_0,   filename: Qwen2.5-1.5B-Instruct-Q8_0.gguf,   uploader: bartowski, imatrix: true}
  - {label: Q6_K,   filename: Qwen2.5-1.5B-Instruct-Q6_K.gguf,   uploader: bartowski, imatrix: true}
  - {label: Q5_K_M, filename: Qwen2.5-1.5B-Instruct-Q5_K_M.gguf, uploader: bartowski, imatrix: true}
  - {label: Q4_K_M, filename: Qwen2.5-1.5B-Instruct-Q4_K_M.gguf, uploader: bartowski, imatrix: true}
  - {label: Q3_K_M, filename: Qwen2.5-1.5B-Instruct-Q3_K_M.gguf, uploader: bartowski, imatrix: true}
  - {label: Q2_K,   filename: Qwen2.5-1.5B-Instruct-Q2_K.gguf,   uploader: bartowski, imatrix: true}
  - {label: IQ2_M,  filename: Qwen2.5-1.5B-Instruct-IQ2_M.gguf,  uploader: bartowski, imatrix: true}
generation:
  seed: 42
  temperature: 0.0
  top_k: 1
  max_tokens: 640          # generous budget; the runbook checks F16 never hits it (spec §6)
  n_ctx: 4096
suites:
  retrieval: {n_items: 40, n_pairs: 8, seed: 1301}
  arithmetic: {n_items: 40, seed: 1301}
  spectacle: {path: configs/prompts_spectacle.yaml}
```

- [ ] **Step 9: Verify the real config loads**

Run: `uv run python -c "from bitcliff_pipeline.config import load_config; c = load_config('configs/qwen2.5-1.5b-pilot.yaml'); print(len(c.quants), 'quants OK')"`
Expected: `7 quants OK`

- [ ] **Step 10: Commit**

```bash
git add bitcliff/pipeline
git commit -m "feat(pipeline): scaffold, ladder config loader, pilot config"
```

---

### Task 2: Hash manifest (pin every result to exact files)

**Files:**
- Create: `bitcliff/pipeline/src/bitcliff_pipeline/hashing.py`
- Test: `bitcliff/pipeline/tests/test_hashing.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `hashing.sha256_file(path: Path) -> str` (hex digest)
  - `hashing.build_manifest(files: dict[str, Path]) -> dict` — maps quant label → `{"filename": str, "sha256": str, "size_bytes": int}`
  - `hashing.write_manifest(manifest: dict, path: Path) -> None` / `hashing.load_manifest(path: Path) -> dict`
  - `hashing.verify_manifest(manifest: dict, files: dict[str, Path]) -> None` — raises `hashing.ManifestMismatch` if any hash differs (this is the "uploader re-quantized" tripwire, spec §4/§14).

- [ ] **Step 1: Write the failing tests**

`tests/test_hashing.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hashing.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/bitcliff_pipeline/hashing.py`:

```python
import hashlib
import json
from pathlib import Path


class ManifestMismatch(Exception):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(files: dict[str, Path]) -> dict:
    return {
        label: {
            "filename": p.name,
            "sha256": sha256_file(p),
            "size_bytes": p.stat().st_size,
        }
        for label, p in files.items()
    }


def write_manifest(manifest: dict, path: Path) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_manifest(manifest: dict, files: dict[str, Path]) -> None:
    for label, entry in manifest.items():
        actual = sha256_file(files[label])
        if actual != entry["sha256"]:
            raise ManifestMismatch(
                f"{label}: expected {entry['sha256'][:12]}..., got {actual[:12]}..."
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hashing.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add bitcliff/pipeline
git commit -m "feat(pipeline): sha256 manifest build/verify for file pinning"
```

---

### Task 3: Model acquisition (download quants, resolve paths)

**Files:**
- Create: `bitcliff/pipeline/src/bitcliff_pipeline/models.py`
- Test: `bitcliff/pipeline/tests/test_models.py`

**Interfaces:**
- Consumes: `LadderConfig` from Task 1.
- Produces:
  - `models.ensure_quants(config: LadderConfig, models_dir: Path, downloader=...) -> dict[str, Path]` — returns quant label → local path, downloading any missing file via `huggingface_hub.hf_hub_download`; skips files already present. Does NOT include F16.
  - `models.resolve_all(config: LadderConfig, models_dir: Path) -> dict[str, Path]` — quant paths plus `"F16": config.f16_path`; raises `FileNotFoundError` if any file (including F16) is missing. This is what generate/grade stages call.

- [ ] **Step 1: Write the failing tests**

`tests/test_models.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/bitcliff_pipeline/models.py`:

```python
from pathlib import Path

from huggingface_hub import hf_hub_download

from .config import LadderConfig


def ensure_quants(
    config: LadderConfig, models_dir: Path, downloader=hf_hub_download
) -> dict[str, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for q in config.quants:
        dest = models_dir / q.filename
        if not dest.exists():
            downloader(repo_id=config.hf_repo, filename=q.filename, local_dir=str(models_dir))
        paths[q.label] = dest
    return paths


def resolve_all(config: LadderConfig, models_dir: Path) -> dict[str, Path]:
    paths = {"F16": config.f16_path}
    for q in config.quants:
        paths[q.label] = models_dir / q.filename
    missing = [label for label, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing model files for: {', '.join(missing)} "
            "(run the download stage; F16 must be converted locally, see PILOT_RUNBOOK.md)"
        )
    return paths
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add bitcliff/pipeline
git commit -m "feat(pipeline): quant download and model path resolution"
```

---

### Task 4: Retrieval suite (multivalue2-style)

**Files:**
- Create: `bitcliff/pipeline/src/bitcliff_pipeline/suites/__init__.py` (empty)
- Create: `bitcliff/pipeline/src/bitcliff_pipeline/suites/retrieval.py`
- Test: `bitcliff/pipeline/tests/test_retrieval.py`

**Interfaces:**
- Consumes: `items.EvalItem` from Task 1.
- Produces:
  - `retrieval.generate_items(n_items: int, n_pairs: int, seed: int) -> list[EvalItem]` — deterministic for a given seed; each item embeds `n_pairs` name → (code, code) associations and asks for one name's two codes; `expected` is that `(code_a, code_b)` tuple.
  - `retrieval.grade(item: EvalItem, text: str) -> str` — `"correct"` if both codes appear in the output, `"partial"` if exactly one, else `"wrong"`.

Design note: this mirrors the paper's multivalue2 task shape (recall two associated values for a queried key). Codes are unique 4-digit strings within an item so a code match cannot be a coincidence from another pair. Before the freeze, this generator gets reconciled item-for-item with the paper's exact task definition; for the pilot the shape is what matters.

- [ ] **Step 1: Write the failing tests**

`tests/test_retrieval.py`:

```python
from bitcliff_pipeline.suites.retrieval import generate_items, grade


def test_generation_is_deterministic():
    a = generate_items(n_items=5, n_pairs=4, seed=1301)
    b = generate_items(n_items=5, n_pairs=4, seed=1301)
    assert a == b


def test_different_seed_differs():
    a = generate_items(n_items=5, n_pairs=4, seed=1301)
    b = generate_items(n_items=5, n_pairs=4, seed=1302)
    assert a != b


def test_item_shape():
    (item,) = generate_items(n_items=1, n_pairs=4, seed=7)
    assert item.suite == "retrieval"
    assert item.id == "retrieval-7-000"
    assert len(item.expected) == 2
    for code in item.expected:
        assert code in item.prompt  # the answer is present in the context
    assert item.prompt.count(":") >= 4  # all pairs listed


def test_codes_unique_within_item():
    (item,) = generate_items(n_items=1, n_pairs=8, seed=7)
    # every code in the prompt context appears exactly once
    codes = [tok.strip(".,") for tok in item.prompt.split() if tok.strip(".,").isdigit()]
    assert len(codes) == len(set(codes))


def test_grade_correct_partial_wrong():
    (item,) = generate_items(n_items=1, n_pairs=4, seed=7)
    a, b = item.expected
    assert grade(item, f"The two codes are {a} and {b}.") == "correct"
    assert grade(item, f"I only remember {a}.") == "partial"
    assert grade(item, "No idea, maybe 0000 and 1111?" ) == "wrong"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/bitcliff_pipeline/suites/retrieval.py`:

```python
import random

from ..items import EvalItem

FIRST_NAMES = [
    "Alice", "Marcus", "Priya", "Chen", "Fatima", "Diego", "Yuki", "Omar",
    "Ingrid", "Kwame", "Lena", "Rafael", "Sofia", "Tariq", "Mei", "Anders",
    "Zara", "Viktor", "Nadia", "Jamal", "Elena", "Hiro", "Amara", "Luca",
]

PROMPT_TEMPLATE = (
    "Here is a list of people and their two ID codes.\n\n{pairs}\n\n"
    "Question: What are the two ID codes for {key}? "
    "Answer with just the two codes."
)


def generate_items(n_items: int, n_pairs: int, seed: int) -> list[EvalItem]:
    rng = random.Random(seed)
    items = []
    for i in range(n_items):
        names = rng.sample(FIRST_NAMES, n_pairs)
        codes = rng.sample(range(1000, 10000), n_pairs * 2)
        assoc = {
            name: (str(codes[2 * j]), str(codes[2 * j + 1]))
            for j, name in enumerate(names)
        }
        target = rng.choice(names)
        pairs = "\n".join(f"{n}: {a}, {b}" for n, (a, b) in assoc.items())
        items.append(
            EvalItem(
                id=f"retrieval-{seed}-{i:03d}",
                suite="retrieval",
                prompt=PROMPT_TEMPLATE.format(pairs=pairs, key=target),
                expected=assoc[target],
            )
        )
    return items


def grade(item: EvalItem, text: str) -> str:
    hits = sum(1 for code in item.expected if code in text)
    return {2: "correct", 1: "partial"}.get(hits, "wrong")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add bitcliff/pipeline
git commit -m "feat(pipeline): retrieval (multivalue2-style) suite with grading"
```

---

### Task 5: Arithmetic suite (GSM8K) and spectacle prompts

**Files:**
- Create: `bitcliff/pipeline/src/bitcliff_pipeline/suites/arithmetic.py`
- Create: `bitcliff/pipeline/src/bitcliff_pipeline/suites/spectacle.py`
- Create: `bitcliff/pipeline/configs/prompts_spectacle.yaml`
- Test: `bitcliff/pipeline/tests/test_arithmetic.py`
- Test: `bitcliff/pipeline/tests/test_spectacle.py`

**Interfaces:**
- Consumes: `items.EvalItem` from Task 1.
- Produces:
  - `arithmetic.extract_final_number(text: str) -> str | None` — prefers the `#### <number>` convention, falls back to the last number in the text; normalizes commas and trailing zeros (`"1,000"` → `"1000"`, `"6.0"` → `"6"`).
  - `arithmetic.items_from_records(records: list[dict], n_items: int, seed: int) -> list[EvalItem]` — pure, deterministic sample; each record is a GSM8K row `{"question": str, "answer": str}`; `expected` is the normalized gold number as a 1-tuple.
  - `arithmetic.load_gsm8k_items(n_items: int, seed: int) -> list[EvalItem]` — thin wrapper over `datasets.load_dataset("openai/gsm8k", "main", split="test")` calling `items_from_records` (not unit-tested; exercised in the runbook).
  - `arithmetic.grade(item: EvalItem, text: str) -> str` — `"correct"` iff the extracted final number equals `item.expected[0]`, else `"wrong"` (no partial credit in arithmetic).
  - `spectacle.load_items(path: str | Path) -> list[EvalItem]` — reads the YAML prompt list; `expected=None`.
  - `spectacle.grade(item: EvalItem, text: str) -> str` — always `"unscored"` (spec §5: spectacle prompts are labeled unscored entertainment).

- [ ] **Step 1: Write the failing tests**

`tests/test_arithmetic.py`:

```python
from bitcliff_pipeline.suites.arithmetic import (
    extract_final_number,
    grade,
    items_from_records,
)

RECORDS = [
    {"question": f"Q{i}: If x = {i} and y = {i}, what is x + y?",
     "answer": f"x + y = {i} + {i} = {2 * i}\n#### {2 * i}"}
    for i in range(1, 21)
]


def test_extract_prefers_hash_marker():
    assert extract_final_number("some steps 3 + 4\n#### 7") == "7"


def test_extract_normalizes_commas_and_floats():
    assert extract_final_number("#### 1,000") == "1000"
    assert extract_final_number("#### 6.0") == "6"
    assert extract_final_number("#### 2.5") == "2.5"


def test_extract_falls_back_to_last_number():
    assert extract_final_number("The answer is 12, no wait, 14.") == "14"


def test_extract_none_when_no_number():
    assert extract_final_number("I cannot solve this.") is None


def test_items_are_deterministic_and_shaped():
    a = items_from_records(RECORDS, n_items=5, seed=1301)
    b = items_from_records(RECORDS, n_items=5, seed=1301)
    assert a == b
    assert all(item.suite == "arithmetic" for item in a)
    assert all(item.expected is not None for item in a)
    assert "####" in a[0].prompt  # instructs the answer format


def test_grade():
    (item,) = items_from_records(RECORDS[:1], n_items=1, seed=1)
    gold = item.expected[0]
    assert grade(item, f"steps...\n#### {gold}") == "correct"
    assert grade(item, "#### 999999") == "wrong"
    assert grade(item, "no answer at all") == "wrong"
```

`tests/test_spectacle.py`:

```python
from bitcliff_pipeline.suites.spectacle import grade, load_items


def test_load_and_grade(tmp_path):
    p = tmp_path / "prompts.yaml"
    p.write_text(
        "prompts:\n"
        "  - {id: spec-001, prompt: 'Write a haiku about databases.'}\n"
        "  - {id: spec-002, prompt: 'Explain gravity to a pirate.'}\n"
    )
    items = load_items(p)
    assert [i.id for i in items] == ["spec-001", "spec-002"]
    assert all(i.suite == "spectacle" and i.expected is None for i in items)
    assert grade(items[0], "anything at all") == "unscored"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_arithmetic.py tests/test_spectacle.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementations**

`src/bitcliff_pipeline/suites/arithmetic.py`:

```python
import random
import re

from ..items import EvalItem

_FINAL = re.compile(r"####\s*([-+]?[\d,]*\.?\d+)")
_NUM = re.compile(r"[-+]?[\d,]*\.?\d+")

ANSWER_INSTRUCTION = (
    "\n\nSolve step by step, then give the final answer on its own line as: #### <number>"
)


def _normalize(s: str) -> str:
    value = float(s.replace(",", ""))
    return str(int(value)) if value == int(value) else str(value)


def extract_final_number(text: str) -> str | None:
    m = _FINAL.search(text)
    if m:
        return _normalize(m.group(1))
    nums = _NUM.findall(text)
    return _normalize(nums[-1]) if nums else None


def items_from_records(records, n_items: int, seed: int) -> list[EvalItem]:
    rng = random.Random(seed)
    picked = rng.sample(list(records), n_items)
    items = []
    for i, r in enumerate(picked):
        gold = extract_final_number(r["answer"])
        items.append(
            EvalItem(
                id=f"arithmetic-{seed}-{i:03d}",
                suite="arithmetic",
                prompt=r["question"] + ANSWER_INSTRUCTION,
                expected=(gold,),
            )
        )
    return items


def load_gsm8k_items(n_items: int, seed: int) -> list[EvalItem]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    return items_from_records(ds, n_items, seed)


def grade(item: EvalItem, text: str) -> str:
    got = extract_final_number(text)
    return "correct" if got is not None and got == item.expected[0] else "wrong"
```

`src/bitcliff_pipeline/suites/spectacle.py`:

```python
from pathlib import Path

import yaml

from ..items import EvalItem


def load_items(path: str | Path) -> list[EvalItem]:
    raw = yaml.safe_load(Path(path).read_text())
    return [
        EvalItem(id=p["id"], suite="spectacle", prompt=p["prompt"], expected=None)
        for p in raw["prompts"]
    ]


def grade(item: EvalItem, text: str) -> str:
    return "unscored"
```

`configs/prompts_spectacle.yaml` — 10 curated candidates for the pilot (the launch set of 50 is chosen AFTER seeing pilot outputs; these seed the search for drama):

```yaml
# Unscored entertainment prompts (spec §5). Pilot candidates; the launch 50
# get curated from whatever the pilot shows degrades most visibly.
prompts:
  - {id: spec-001, prompt: "Tell me the plot of Romeo and Juliet in exactly three sentences."}
  - {id: spec-002, prompt: "List the planets of the solar system in order from the Sun."}
  - {id: spec-003, prompt: "Write a limerick about a cat who codes in Python."}
  - {id: spec-004, prompt: "What year did the Berlin Wall fall, and what country was reunified afterward?"}
  - {id: spec-005, prompt: "Count backwards from 20 to 1, separated by commas."}
  - {id: spec-006, prompt: "Translate 'Good morning, how are you?' into French, Spanish, and German."}
  - {id: spec-007, prompt: "Name the four members of The Beatles and the instrument each mainly played."}
  - {id: spec-008, prompt: "Explain why the sky is blue in one short paragraph a child could understand."}
  - {id: spec-009, prompt: "Write the first 12 numbers of the Fibonacci sequence."}
  - {id: spec-010, prompt: "Give me a recipe for pancakes with exact measurements for 4 servings."}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_arithmetic.py tests/test_spectacle.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add bitcliff/pipeline
git commit -m "feat(pipeline): GSM8K arithmetic suite and unscored spectacle prompts"
```

---

### Task 6: Deterministic generation runner

**Files:**
- Create: `bitcliff/pipeline/src/bitcliff_pipeline/generate.py`
- Test: `bitcliff/pipeline/tests/test_generate.py`

**Interfaces:**
- Consumes: `EvalItem` (Task 1), `GenSettings` (Task 1).
- Produces:
  - `generate.OutputRecord(item_id: str, suite: str, quant_label: str, model_sha256: str, prompt: str, text: str, finish_reason: str, gen_settings: dict, machine: str)` — frozen dataclass; one generated output.
  - `generate.make_llm(model_path: Path, gen: GenSettings)` — real factory returning a `llama_cpp.Llama` (chat completions use the GGUF's embedded chat template).
  - `generate.run_items(llm, items: list[EvalItem], quant_label: str, model_sha256: str, gen: GenSettings) -> list[OutputRecord]` — `llm` is anything with `create_chat_completion(...)`, so tests inject a fake.
  - `generate.write_records(records: list[OutputRecord], path: Path) -> None` and `generate.read_records(path: Path) -> list[OutputRecord]` — JSONL roundtrip. Grading (Task 7) and reporting (Task 9) consume these files.

Determinism contract (Global Constraints): every record embeds the full `gen_settings` dict and a `machine` platform string, so any published output is traceable to its exact configuration.

- [ ] **Step 1: Write the failing tests**

`tests/test_generate.py`:

```python
from dataclasses import asdict

from bitcliff_pipeline.config import GenSettings
from bitcliff_pipeline.generate import (
    OutputRecord,
    read_records,
    run_items,
    write_records,
)
from bitcliff_pipeline.items import EvalItem

GEN = GenSettings(seed=42, temperature=0.0, top_k=1, max_tokens=640, n_ctx=4096)

ITEMS = [
    EvalItem("retrieval-1-000", "retrieval", "What is Alice's code?", ("1234", "5678")),
    EvalItem("spec-001", "spectacle", "Write a haiku.", None),
]


class FakeLlm:
    def __init__(self):
        self.calls = []

    def create_chat_completion(self, messages, max_tokens, temperature, top_k, seed):
        self.calls.append(
            {"messages": messages, "max_tokens": max_tokens,
             "temperature": temperature, "top_k": top_k, "seed": seed}
        )
        return {
            "choices": [
                {"message": {"content": f"echo: {messages[0]['content'][:10]}"},
                 "finish_reason": "stop"}
            ]
        }


def test_run_items_builds_records_with_settings():
    llm = FakeLlm()
    records = run_items(llm, ITEMS, quant_label="Q4_K_M", model_sha256="abc123", gen=GEN)
    assert len(records) == 2
    r = records[0]
    assert r.item_id == "retrieval-1-000"
    assert r.suite == "retrieval"
    assert r.quant_label == "Q4_K_M"
    assert r.model_sha256 == "abc123"
    assert r.finish_reason == "stop"
    assert r.gen_settings == asdict(GEN)
    assert r.machine  # non-empty platform string
    # deterministic settings actually passed through to the model
    assert llm.calls[0]["temperature"] == 0.0
    assert llm.calls[0]["top_k"] == 1
    assert llm.calls[0]["seed"] == 42


def test_jsonl_roundtrip(tmp_path):
    llm = FakeLlm()
    records = run_items(llm, ITEMS, "Q8_0", "def456", GEN)
    path = tmp_path / "Q8_0.jsonl"
    write_records(records, path)
    loaded = read_records(path)
    assert loaded == records
    assert isinstance(loaded[0], OutputRecord)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/bitcliff_pipeline/generate.py`:

```python
import dataclasses
import json
import platform
from dataclasses import dataclass
from pathlib import Path

from .config import GenSettings
from .items import EvalItem


@dataclass(frozen=True)
class OutputRecord:
    item_id: str
    suite: str
    quant_label: str
    model_sha256: str
    prompt: str
    text: str
    finish_reason: str
    gen_settings: dict
    machine: str


def make_llm(model_path: Path, gen: GenSettings):
    from llama_cpp import Llama

    return Llama(
        model_path=str(model_path),
        n_ctx=gen.n_ctx,
        seed=gen.seed,
        n_gpu_layers=-1,
        verbose=False,
    )


def run_items(
    llm,
    items: list[EvalItem],
    quant_label: str,
    model_sha256: str,
    gen: GenSettings,
) -> list[OutputRecord]:
    machine = f"{platform.platform()} / {platform.machine()}"
    records = []
    for item in items:
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": item.prompt}],
            max_tokens=gen.max_tokens,
            temperature=gen.temperature,
            top_k=gen.top_k,
            seed=gen.seed,
        )
        choice = out["choices"][0]
        records.append(
            OutputRecord(
                item_id=item.id,
                suite=item.suite,
                quant_label=quant_label,
                model_sha256=model_sha256,
                prompt=item.prompt,
                text=choice["message"]["content"],
                finish_reason=choice.get("finish_reason") or "stop",
                gen_settings=dataclasses.asdict(gen),
                machine=machine,
            )
        )
    return records


def write_records(records: list[OutputRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(dataclasses.asdict(r)) + "\n")


def read_records(path: Path) -> list[OutputRecord]:
    return [OutputRecord(**json.loads(line)) for line in path.read_text().splitlines()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_generate.py -v`
Expected: 2 PASS

- [ ] **Step 5: Smoke-test against a real model (manual, no assertion)**

Only if a small GGUF is already on disk; otherwise defer to the runbook. This verifies the `llama_cpp.Llama` kwargs are right before the weekend run:

```bash
uv run python - <<'EOF'
from pathlib import Path
from bitcliff_pipeline.config import GenSettings
from bitcliff_pipeline.generate import make_llm, run_items
from bitcliff_pipeline.items import EvalItem

gen = GenSettings(seed=42, temperature=0.0, top_k=1, max_tokens=64, n_ctx=2048)
path = Path("models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
if path.exists():
    llm = make_llm(path, gen)
    (r,) = run_items(llm, [EvalItem("smoke-1", "spectacle", "Say hello in five words.", None)], "Q4_K_M", "unhashed-smoke", gen)
    print(repr(r.text), r.finish_reason)
else:
    print("no local model; smoke test deferred to runbook")
EOF
```

Expected: a short greeting and `stop`, or the deferral message.

- [ ] **Step 6: Commit**

```bash
git add bitcliff/pipeline
git commit -m "feat(pipeline): deterministic generation runner with JSONL records"
```

---

### Task 7: Grading with mechanical truncation and loop flags

**Files:**
- Create: `bitcliff/pipeline/src/bitcliff_pipeline/grading.py`
- Test: `bitcliff/pipeline/tests/test_grading.py`

**Interfaces:**
- Consumes: `EvalItem` (Task 1), `OutputRecord` (Task 6), suite `grade` functions (Tasks 4–5).
- Produces:
  - `grading.GradeResult(item_id: str, suite: str, quant_label: str, state: str, truncated: bool, loop: bool)` — frozen dataclass; `state` ∈ {`correct`, `partial`, `wrong`, `unscored`}.
  - `grading.detect_loop(text: str, min_len: int = 10, min_repeats: int = 3) -> bool` — mechanical: True iff the text ends with the same chunk (≥ `min_len` chars) repeated ≥ `min_repeats` times consecutively.
  - `grading.grade_record(items_by_id: dict[str, EvalItem], record: OutputRecord) -> GradeResult` — dispatches to the suite grader; `truncated = (finish_reason == "length")`. Truncation does not change `state` (a truncated wrong answer is `wrong, truncated=True`) — the separation the spec demands.
  - `grading.write_grades(grades: list[GradeResult], path: Path) -> None` / `grading.read_grades(path: Path) -> list[GradeResult]` — JSONL roundtrip for the report stage.

- [ ] **Step 1: Write the failing tests**

`tests/test_grading.py`:

```python
from bitcliff_pipeline.config import GenSettings
from bitcliff_pipeline.generate import OutputRecord
from bitcliff_pipeline.grading import (
    GradeResult,
    detect_loop,
    grade_record,
    read_grades,
    write_grades,
)
from bitcliff_pipeline.items import EvalItem
import dataclasses

GEN = dataclasses.asdict(GenSettings(42, 0.0, 1, 640, 4096))


def record(item, text, finish_reason="stop", quant="Q4_K_M"):
    return OutputRecord(
        item_id=item.id, suite=item.suite, quant_label=quant,
        model_sha256="abc", prompt=item.prompt, text=text,
        finish_reason=finish_reason, gen_settings=GEN, machine="test",
    )


def test_detect_loop_on_repeated_tail():
    assert detect_loop("Setup text. " + "the answer is the answer is " * 6)
    assert detect_loop("na na na " * 20, min_len=5)


def test_detect_loop_false_on_normal_text():
    assert not detect_loop("The two codes are 1234 and 5678.")
    assert not detect_loop("")


def test_grade_record_retrieval_correct():
    item = EvalItem("retrieval-1-000", "retrieval", "codes?", ("1234", "5678"))
    g = grade_record({item.id: item}, record(item, "They are 1234 and 5678."))
    assert g == GradeResult("retrieval-1-000", "retrieval", "Q4_K_M", "correct", False, False)


def test_grade_record_truncation_is_separate_flag():
    item = EvalItem("arithmetic-1-000", "arithmetic", "2+2?", ("4",))
    g = grade_record({item.id: item}, record(item, "Let me think step by", finish_reason="length"))
    assert g.state == "wrong"       # never arrived => wrong (spec §6)
    assert g.truncated is True      # ...but tracked separately


def test_grade_record_spectacle_unscored():
    item = EvalItem("spec-001", "spectacle", "haiku", None)
    g = grade_record({item.id: item}, record(item, "five seven five"))
    assert g.state == "unscored"


def test_grades_jsonl_roundtrip(tmp_path):
    grades = [
        GradeResult("a", "retrieval", "Q8_0", "correct", False, False),
        GradeResult("b", "arithmetic", "Q2_K", "wrong", True, True),
    ]
    path = tmp_path / "grades.jsonl"
    write_grades(grades, path)
    assert read_grades(path) == grades
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_grading.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/bitcliff_pipeline/grading.py`:

```python
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from .generate import OutputRecord
from .items import EvalItem
from .suites import arithmetic, retrieval, spectacle

GRADERS = {
    "retrieval": retrieval.grade,
    "arithmetic": arithmetic.grade,
    "spectacle": spectacle.grade,
}


@dataclass(frozen=True)
class GradeResult:
    item_id: str
    suite: str
    quant_label: str
    state: str
    truncated: bool
    loop: bool


def detect_loop(text: str, min_len: int = 10, min_repeats: int = 3) -> bool:
    tail = text[-2000:].rstrip()
    max_size = len(tail) // min_repeats
    for size in range(min_len, max_size + 1):
        chunk = tail[-size:]
        if chunk * min_repeats == tail[-size * min_repeats :]:
            return True
    return False


def grade_record(items_by_id: dict[str, EvalItem], record: OutputRecord) -> GradeResult:
    item = items_by_id[record.item_id]
    state = GRADERS[item.suite](item, record.text)
    return GradeResult(
        item_id=item.id,
        suite=item.suite,
        quant_label=record.quant_label,
        state=state,
        truncated=record.finish_reason == "length",
        loop=detect_loop(record.text),
    )


def write_grades(grades: list[GradeResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for g in grades:
            f.write(json.dumps(dataclasses.asdict(g)) + "\n")


def read_grades(path: Path) -> list[GradeResult]:
    return [GradeResult(**json.loads(line)) for line in path.read_text().splitlines()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_grading.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add bitcliff/pipeline
git commit -m "feat(pipeline): grading with mechanical truncation and loop flags"
```

---

### Task 8: First-divergent-token vs F16

**Files:**
- Create: `bitcliff/pipeline/src/bitcliff_pipeline/divergence.py`
- Test: `bitcliff/pipeline/tests/test_divergence.py`

**Interfaces:**
- Consumes: `OutputRecord` (Task 6).
- Produces:
  - `divergence.first_divergence_index(a: list, b: list) -> int | None` — index of first mismatch between two token (or word) sequences; `min(len)` if one is a prefix of the other; `None` if identical.
  - `divergence.divergence_for_records(baseline: OutputRecord, other: OutputRecord, tokenize) -> int | None` — tokenizes both texts with the provided callable and compares. `tokenize` is any `str -> list`; in the runbook it is the F16 model's `llm.tokenize` (all quants of one model share a tokenizer), in tests it is `str.split`.

This powers the playground's "first token where it diverged, highlighted" card field (spec §5). For the pilot it is a per-item integer in the grade output; UI highlighting is a later plan.

- [ ] **Step 1: Write the failing tests**

`tests/test_divergence.py`:

```python
from bitcliff_pipeline.divergence import divergence_for_records, first_divergence_index
from bitcliff_pipeline.generate import OutputRecord


def rec(text):
    return OutputRecord(
        item_id="x", suite="retrieval", quant_label="Q", model_sha256="h",
        prompt="p", text=text, finish_reason="stop", gen_settings={}, machine="m",
    )


def test_identical_sequences_return_none():
    assert first_divergence_index([1, 2, 3], [1, 2, 3]) is None


def test_first_mismatch_index():
    assert first_divergence_index([1, 2, 3], [1, 9, 3]) == 1


def test_prefix_returns_shorter_length():
    assert first_divergence_index([1, 2], [1, 2, 3]) == 2


def test_divergence_for_records_with_word_tokenizer():
    base = rec("the cat sat on the mat")
    other = rec("the cat slept on the mat")
    assert divergence_for_records(base, other, tokenize=str.split) == 2
    assert divergence_for_records(base, rec("the cat sat on the mat"), tokenize=str.split) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_divergence.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/bitcliff_pipeline/divergence.py`:

```python
from .generate import OutputRecord


def first_divergence_index(a: list, b: list) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def divergence_for_records(baseline: OutputRecord, other: OutputRecord, tokenize) -> int | None:
    return first_divergence_index(tokenize(baseline.text), tokenize(other.text))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_divergence.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add bitcliff/pipeline
git commit -m "feat(pipeline): first-divergent-token computation vs F16 baseline"
```

---

### Task 9: Report — aggregates, retention, CSV/JSON/PNG

**Files:**
- Create: `bitcliff/pipeline/src/bitcliff_pipeline/report.py`
- Test: `bitcliff/pipeline/tests/test_report.py`

**Interfaces:**
- Consumes: `GradeResult` (Task 7).
- Produces:
  - `report.aggregate(grades: list[GradeResult]) -> list[dict]` — one row per (quant_label, suite), excluding `unscored`; keys: `quant_label, suite, n, correct, partial, wrong, truncated, loops, accuracy` where `accuracy = (correct + 0.5 * partial) / n` (pilot scoring; the registered scoring rule is fixed at the freeze).
  - `report.add_retention(rows: list[dict], baseline_label: str = "F16") -> list[dict]` — adds `retention = accuracy / baseline_accuracy` per suite (`None` when baseline accuracy is 0). Retention is within-model by construction (spec §16).
  - `report.write_csv(rows: list[dict], path: Path) -> None` and `report.write_json(rows: list[dict], path: Path) -> None`.
  - `report.plot_retention(rows: list[dict], ladder_order: list[str], path: Path) -> None` — one matplotlib line per suite, x = ladder order (F16 → lowest), y = retention; saved PNG. This is the exploratory curve the go/no-go gate reads.

- [ ] **Step 1: Write the failing tests**

`tests/test_report.py`:

```python
import csv
import json

from bitcliff_pipeline.grading import GradeResult
from bitcliff_pipeline.report import add_retention, aggregate, plot_retention, write_csv, write_json


def g(quant, suite, state, truncated=False, loop=False, i=[0]):
    i[0] += 1
    return GradeResult(f"item-{i[0]:03d}", suite, quant, state, truncated, loop)


GRADES = (
    [g("F16", "retrieval", "correct") for _ in range(9)]
    + [g("F16", "retrieval", "wrong")]
    + [g("Q2_K", "retrieval", "correct") for _ in range(3)]
    + [g("Q2_K", "retrieval", "partial")]
    + [g("Q2_K", "retrieval", "wrong", truncated=True, loop=True) for _ in range(6)]
    + [g("F16", "spectacle", "unscored")]
)


def row(rows, quant, suite):
    return next(r for r in rows if r["quant_label"] == quant and r["suite"] == suite)


def test_aggregate_counts_and_accuracy():
    rows = aggregate(GRADES)
    assert all(r["suite"] != "spectacle" for r in rows)  # unscored excluded
    f16 = row(rows, "F16", "retrieval")
    assert (f16["n"], f16["correct"], f16["accuracy"]) == (10, 9, 0.9)
    q2 = row(rows, "Q2_K", "retrieval")
    assert q2["n"] == 10
    assert q2["accuracy"] == (3 + 0.5) / 10
    assert q2["truncated"] == 6
    assert q2["loops"] == 6


def test_add_retention():
    rows = add_retention(aggregate(GRADES))
    assert row(rows, "F16", "retrieval")["retention"] == 1.0
    assert abs(row(rows, "Q2_K", "retrieval")["retention"] - 0.35 / 0.9) < 1e-9


def test_write_csv_and_json(tmp_path):
    rows = add_retention(aggregate(GRADES))
    write_csv(rows, tmp_path / "r.csv")
    write_json(rows, tmp_path / "r.json")
    with open(tmp_path / "r.csv") as f:
        parsed = list(csv.DictReader(f))
    assert len(parsed) == len(rows)
    assert json.loads((tmp_path / "r.json").read_text()) == rows


def test_plot_retention_writes_png(tmp_path):
    rows = add_retention(aggregate(GRADES))
    out = tmp_path / "retention.png"
    plot_retention(rows, ["F16", "Q2_K"], out)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/bitcliff_pipeline/report.py`:

```python
import csv
import json
from collections import defaultdict
from pathlib import Path

from .grading import GradeResult


def aggregate(grades: list[GradeResult]) -> list[dict]:
    groups: dict[tuple[str, str], list[GradeResult]] = defaultdict(list)
    for g in grades:
        if g.state != "unscored":
            groups[(g.quant_label, g.suite)].append(g)
    rows = []
    for (quant_label, suite), gs in sorted(groups.items()):
        n = len(gs)
        correct = sum(1 for g in gs if g.state == "correct")
        partial = sum(1 for g in gs if g.state == "partial")
        rows.append(
            {
                "quant_label": quant_label,
                "suite": suite,
                "n": n,
                "correct": correct,
                "partial": partial,
                "wrong": sum(1 for g in gs if g.state == "wrong"),
                "truncated": sum(1 for g in gs if g.truncated),
                "loops": sum(1 for g in gs if g.loop),
                "accuracy": (correct + 0.5 * partial) / n,
            }
        )
    return rows


def add_retention(rows: list[dict], baseline_label: str = "F16") -> list[dict]:
    baseline = {
        r["suite"]: r["accuracy"] for r in rows if r["quant_label"] == baseline_label
    }
    out = []
    for r in rows:
        base = baseline.get(r["suite"])
        retention = r["accuracy"] / base if base else None
        out.append({**r, "retention": retention})
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))


def plot_retention(rows: list[dict], ladder_order: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    suites = sorted({r["suite"] for r in rows})
    for suite in suites:
        by_label = {r["quant_label"]: r["retention"] for r in rows if r["suite"] == suite}
        ys = [by_label.get(label) for label in ladder_order]
        ax.plot(ladder_order, ys, marker="o", label=suite)
    ax.set_xlabel("quant level (exploratory pilot — numbers are thrown away)")
    ax.set_ylabel("retention vs F16")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_report.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add bitcliff/pipeline
git commit -m "feat(pipeline): retention aggregation, CSV/JSON export, curve plot"
```

---

### Task 10: CLI orchestrator

**Files:**
- Create: `bitcliff/pipeline/src/bitcliff_pipeline/__main__.py`
- Test: `bitcliff/pipeline/tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `python -m bitcliff_pipeline <config.yaml> --run-id <id> --stage {download,generate,grade,report,all}` with `--models-dir` (default `models/`) and `--runs-dir` (default `runs/`).
  - `__main__.build_items(config: LadderConfig, base_dir: Path) -> list[EvalItem]` — assembles retrieval + arithmetic + spectacle items from the config's `suites` block (spectacle path resolved relative to `base_dir`, the config file's parent's parent, i.e. the pipeline root).
  - `__main__.run_pipeline(config, run_id, models_dir, runs_dir, stage, llm_factory=..., base_dir=...) -> None` — the orchestrator; `llm_factory(path, gen)` defaults to `generate.make_llm`, injectable for the end-to-end test. Layout it produces:
    - `runs/<run_id>/manifest.json`
    - `runs/<run_id>/items.jsonl` (the exact prompts used, for reproducibility)
    - `runs/<run_id>/outputs/<LABEL>.jsonl` (one per ladder rung incl. F16)
    - `runs/<run_id>/grades.jsonl` (grade results, each with `divergence` field vs F16, word-level for the pilot)
    - `runs/<run_id>/results.csv`, `results.json`, `retention.png`

Stage behavior: `download` = fetch quants + write manifest; `generate` = verify manifest, then run every ladder rung (skipping any `outputs/<LABEL>.jsonl` that already exists, so an interrupted weekend run resumes); `grade` and `report` operate purely on files; `all` runs everything in order.

- [ ] **Step 1: Write the failing test (end-to-end with fake llm)**

`tests/test_cli.py`:

```python
import json
from pathlib import Path

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


def test_build_items_covers_configured_suites(tmp_path):
    cfg = make_config(tmp_path)
    items = build_items(cfg, base_dir=tmp_path)
    suites = [i.suite for i in items]
    assert suites.count("retrieval") == 4
    assert suites.count("spectacle") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/bitcliff_pipeline/__main__.py`:

```python
import argparse
import dataclasses
import json
from pathlib import Path

from . import generate as gen_mod
from .config import LadderConfig, load_config
from .divergence import divergence_for_records
from .grading import grade_record, write_grades
from .hashing import build_manifest, load_manifest, verify_manifest, write_manifest
from .items import EvalItem
from .models import ensure_quants, resolve_all
from .report import add_retention, aggregate, plot_retention, write_csv, write_json
from .grading import read_grades


def build_items(config: LadderConfig, base_dir: Path) -> list[EvalItem]:
    from .suites import arithmetic, retrieval, spectacle

    items: list[EvalItem] = []
    suites = config.suites
    if "retrieval" in suites:
        s = suites["retrieval"]
        items += retrieval.generate_items(s["n_items"], s["n_pairs"], s["seed"])
    if "arithmetic" in suites:
        s = suites["arithmetic"]
        items += arithmetic.load_gsm8k_items(s["n_items"], s["seed"])
    if "spectacle" in suites:
        items += spectacle.load_items(base_dir / suites["spectacle"]["path"])
    return items


def run_pipeline(
    config: LadderConfig,
    run_id: str,
    models_dir: Path,
    runs_dir: Path,
    stage: str,
    llm_factory=gen_mod.make_llm,
    base_dir: Path = Path("."),
) -> None:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"

    if stage in ("download", "all"):
        ensure_quants(config, models_dir)
        paths = resolve_all(config, models_dir)
        write_manifest(build_manifest(paths), manifest_path)
        print(f"manifest written: {manifest_path}")

    if stage in ("generate", "all"):
        paths = resolve_all(config, models_dir)
        manifest = load_manifest(manifest_path)
        verify_manifest(manifest, paths)
        items = build_items(config, base_dir)
        items_path = run_dir / "items.jsonl"
        items_path.write_text(
            "".join(json.dumps(dataclasses.asdict(i)) + "\n" for i in items)
        )
        for label, path in paths.items():
            out_path = run_dir / "outputs" / f"{label}.jsonl"
            if out_path.exists():
                print(f"skip {label}: {out_path} exists")
                continue
            print(f"generating {label} ({len(items)} items)...")
            llm = llm_factory(path, config.generation)
            records = gen_mod.run_items(
                llm, items, label, manifest[label]["sha256"], config.generation
            )
            gen_mod.write_records(records, out_path)

    if stage in ("grade", "all"):
        items = [
            EvalItem(**{**d, "expected": tuple(d["expected"]) if d["expected"] else None})
            for d in map(json.loads, (run_dir / "items.jsonl").read_text().splitlines())
        ]
        items_by_id = {i.id: i for i in items}
        baseline = {
            r.item_id: r for r in gen_mod.read_records(run_dir / "outputs" / "F16.jsonl")
        }
        graded_dicts = []
        for out_path in sorted((run_dir / "outputs").glob("*.jsonl")):
            for record in gen_mod.read_records(out_path):
                g = grade_record(items_by_id, record)
                div = divergence_for_records(
                    baseline[record.item_id], record, tokenize=str.split
                )
                graded_dicts.append({**dataclasses.asdict(g), "divergence": div})
        with open(run_dir / "grades.jsonl", "w") as f:
            for d in graded_dicts:
                f.write(json.dumps(d) + "\n")
        print(f"graded {len(graded_dicts)} outputs")

    if stage in ("report", "all"):
        grades = read_grades(run_dir / "grades.jsonl")
        rows = add_retention(aggregate(grades))
        write_csv(rows, run_dir / "results.csv")
        write_json(rows, run_dir / "results.json")
        ladder_order = ["F16"] + [q.label for q in config.quants]
        plot_retention(rows, ladder_order, run_dir / "retention.png")
        print(f"report written under {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="bitcliff_pipeline")
    parser.add_argument("config")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage", default="all",
        choices=["download", "generate", "grade", "report", "all"],
    )
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()
    config = load_config(args.config)
    run_pipeline(
        config,
        run_id=args.run_id,
        models_dir=Path(args.models_dir),
        runs_dir=Path(args.runs_dir),
        stage=args.stage,
        base_dir=Path(args.config).resolve().parent.parent,
    )


if __name__ == "__main__":
    main()
```

Note for the implementer: `read_grades` returns `GradeResult` objects and ignores nothing — but `grades.jsonl` rows carry an extra `divergence` key, so `GradeResult(**json.loads(line))` in `grading.read_grades` will raise `TypeError` on unknown kwargs. Fix `read_grades` in this task to tolerate extra keys:

```python
def read_grades(path: Path) -> list[GradeResult]:
    field_names = {f.name for f in dataclasses.fields(GradeResult)}
    out = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        out.append(GradeResult(**{k: v for k, v in d.items() if k in field_names}))
    return out
```

(add `import dataclasses` to `grading.py`; the Task 7 roundtrip test still passes).

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest -v`
Expected: ALL tests pass (Tasks 1–10).

- [ ] **Step 5: Commit**

```bash
git add bitcliff/pipeline
git commit -m "feat(pipeline): CLI orchestrator with resumable stages"
```

---

### Task 11: Pilot runbook (the weekend procedure)

**Files:**
- Create: `bitcliff/pipeline/PILOT_RUNBOOK.md`

**Interfaces:**
- Consumes: the complete pipeline (Tasks 1–10).
- Produces: the documented, human-executed Phase 0A procedure and its go/no-go gate. No code.

- [ ] **Step 1: Write the runbook**

`bitcliff/pipeline/PILOT_RUNBOOK.md`:

```markdown
# Phase 0A Pilot Runbook — Qwen2.5-1.5B ladder, local hardware

**This run is EXPLORATORY (spec §10).** Its numbers are thrown away. They may inform
curation, the length budget, and parameter choices — and that fact gets declared in the
registration. Nothing here is confirmatory. Zero cloud spend.

## 1. One-time setup

```bash
cd bitcliff/pipeline
uv sync
```

Convert F16 locally (the reference; never a download recommendation):

```bash
# ~3 GB download + ~3 GB output; needs the llama.cpp conversion script
git clone --depth 1 https://github.com/ggml-org/llama.cpp ../llama.cpp
uv run --with -r ../llama.cpp/requirements/requirements-convert_hf_to_gguf.txt \
  hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/hf/Qwen2.5-1.5B-Instruct
uv run --with -r ../llama.cpp/requirements/requirements-convert_hf_to_gguf.txt \
  python ../llama.cpp/convert_hf_to_gguf.py models/hf/Qwen2.5-1.5B-Instruct \
  --outtype f16 --outfile models/f16/Qwen2.5-1.5B-Instruct-f16.gguf
```

(If the `--with -r` incantation fights back, make a scratch venv for the converter;
it only runs once. Any working conversion is fine — the F16 gets hashed either way.)

Sanity-check the ladder still matches the uploader's repo (filenames drift when
uploaders re-quantize; the manifest will catch silent content changes):

```bash
uv run python -c "
from huggingface_hub import list_repo_files
print('\n'.join(sorted(f for f in list_repo_files('bartowski/Qwen2.5-1.5B-Instruct-GGUF') if f.endswith('.gguf'))))"
```

## 2. The run

```bash
uv run python -m bitcliff_pipeline configs/qwen2.5-1.5b-pilot.yaml \
  --run-id pilot-0a --stage all
```

Notes:
- Ladder = 8 rungs × ~90 items (~40 retrieval, 40 arithmetic, 10 spectacle).
  Interrupt any time; `--stage generate` resumes, skipping finished rungs.
- Total quant downloads ≈ 8 GB. F16 is the slowest rung; start it before dinner.

## 3. Length-budget check (spec §6: the budget must be genuinely generous)

```bash
uv run python -c "
from bitcliff_pipeline.generate import read_records
recs = read_records('runs/pilot-0a/outputs/F16.jsonl')
trunc = [r.item_id for r in recs if r.finish_reason == 'length']
print(f'{len(trunc)}/{len(recs)} F16 outputs truncated'); print(trunc)"
```

If ANY F16 output is truncated, raise `max_tokens` in the config, delete
`runs/pilot-0a/outputs/`, and rerun. The frozen budget must never clip full precision.

## 4. Read the curves (the whole point)

Open `runs/pilot-0a/retention.png` and `results.csv`. Read raw outputs for the
low rungs: `runs/pilot-0a/outputs/IQ2_M.jsonl` — this is where curation candidates
for the launch-50 come from.

## 5. The gate (spec §10 "Bars")

- **Spectacle bar:** does the 1.5B show visible drama on curated prompts by IQ2_M —
  forgotten facts, loops, salad? If yes, the spectacle half has content.
- **Signal bar:** are the retrieval/arithmetic curves informative? Flat-then-cliff
  COUNTS as informative. Only "every curve is noise" kills the project.
- Record the verdict, the chosen length budget, and curation notes in
  `runs/pilot-0a/PILOT_NOTES.md` (committed; the outputs themselves stay gitignored).

**If the gate passes → next plan: the freeze** (registration text, twin problems,
margins, license audit, file lists+hashes for the 8B ladders). Nothing confirmatory
runs and no GPU money is spent before that commit (spec §10; aws-ops).

**If the bottom rungs aren't deranged enough:** quantize in-house bottom rungs
(spec §4 exception, 1.5B only, clearly labeled) with llama.cpp's `llama-quantize`
and add them to the config with `uploader: bitcliff-inhouse, imatrix: false`.
```

- [ ] **Step 2: Verify the runbook's commands against the implemented CLI**

Run: `uv run python -m bitcliff_pipeline --help`
Expected: usage shows `config`, `--run-id`, `--stage {download,generate,grade,report,all}`, `--models-dir`, `--runs-dir` — matching every command in the runbook.

- [ ] **Step 3: Commit**

```bash
git add bitcliff/pipeline/PILOT_RUNBOOK.md
git commit -m "docs(pipeline): Phase 0A pilot runbook with go/no-go gate"
```

---

## Out of scope — follow-on plans, gated on the pilot

Per the spec's own gating (§10, §18), these each get their own plan AFTER the pilot curves exist. Listed so nothing in the spec is silently dropped:

1. **The freeze plan** (spec §6, §10): registration document, statistical margins and the four cell states, equivalence-test design, GSM8K twin construction + round-trip checks + embargo mechanics, license audit, verified 8B file lists and hashes, third-party timestamping.
2. **Phase 0B confirmatory plan** (spec §4, §7, §10): both 8B ladders on GPU (g6e.xlarge per aws-ops), the uploader shootout (bartowski/unsloth/mradermacher at Q4+Q3), official-vs-community on Qwen, all divergence measurements (registered questions 1–3), S3 artifact sync + teardown per aws-ops.
3. **Phase 0B′ + 0C plan** (spec §10): 1.5B confirmatory rerun on the actual serving machine (c7i.xlarge), curated-50 regeneration there, blind guess-the-quant gate with two naive raters.
4. **The site plan** (spec §5): Vercel + Cloudflare (per aws-ops: no managed AWS for the site), playground with ladder/AB views, live 1.5B box with rate limits and precomputed fallback, the matrices with cliff lines and four cell states, durability page, methodology + rendered registration, permalinks, share cards with tracking tags, CSV/JSON downloads.
5. **Launch content plan** (spec §11): demo mode + 15-second clip, bartowski DM, X thread, r/LocalLLaMA post, Show HN, the follow-up drip calendar.
6. **Phase 2–4 plans** (spec §5, §9, §10): the Picker, snippet generator, instruction-following + long-context suites, gallery + voting, the 70B ladder and iso-RAM view (g6e.12xlarge, stop-and-ask per aws-ops).

## Self-review notes

- **Spec coverage:** every launch-pipeline concept the pilot needs (ladder-as-files, hashing, determinism, grade states, truncation separation, loop detection, divergence field, retention curves, exploratory framing, length-budget check, in-house bottom-rung fallback, go/no-go bars) maps to a task. Everything else is explicitly parked in "Out of scope" with its spec section.
- **Type consistency:** `EvalItem` / `OutputRecord` / `GradeResult` field names and orders are identical across Tasks 1–10; `GenSettings` construction sites all use the 5-arg shape; the `read_grades` extra-key fix in Task 10 is called out explicitly rather than silently assumed.
- **Known simplifications (pilot-only, by design):** word-level divergence instead of true token-level (real tokenizer wired in a later plan), 0.5-credit partials (scoring rule frozen at registration), the retrieval generator is multivalue2-*shaped* pending reconciliation with the paper's exact task at the freeze. Each is noted where it lives.
