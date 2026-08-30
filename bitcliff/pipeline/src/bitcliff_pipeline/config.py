from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExtraFile:
    """One additional file that belongs to a QuantFile's rung but is not
    itself the file llama-cpp loads -- e.g. the second (and later) shard of
    a sharded GGUF (RUN_0B.md §2 P3: "official q4_k_m (2 shards) ... pass
    the first shard path; record both files"). `filename` is downloaded
    alongside the primary quant file (from the same repo); `sha256`, when
    present, is verified the same way as QuantFile.sha256 (0B P3: config
    pins tie to the committed reference manifests)."""

    filename: str
    sha256: str | None = None


@dataclass(frozen=True)
class QuantFile:
    label: str
    filename: str
    uploader: str
    imatrix: bool
    spectacle_only: bool = False
    # 0B P3 additions (RUN_0B.md §2 P3 "config key additions"):
    sha256: str | None = None
    """Config-pinned sha256 for `filename`, from the committed
    reference-manifests/*.json. When set, `models.ensure_quants` verifies
    the downloaded (or already-present) file's actual hash against this pin
    before the run can proceed -- distinct from `hashing.verify_manifest`'s
    run-local check (which only detects drift between a run's own download
    and generate stages, not drift from the registered manifest). `None`
    (the default) skips this pin, e.g. for older/non-0B configs that
    predate it."""
    hf_repo: str | None = None
    """Per-quant repo override. Falls back to LadderConfig.hf_repo when
    None (every pre-0B config's implicit behavior, preserved). Needed for
    RUN_0B.md's uploader shootout (Arm 1): three files' worth of quants —
    unsloth, mradermacher static, mradermacher i1 — each from a distinct HF
    repo, inside one run config whose top-level hf_repo can only name one
    of them."""
    extra_files: tuple[ExtraFile, ...] = ()
    """Additional shard files for this rung (see ExtraFile); () for every
    unsharded quant (the overwhelming majority)."""


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


def _load_quant(raw_quant: dict) -> QuantFile:
    raw_quant = dict(raw_quant)
    raw_extra = raw_quant.pop("extra_files", None) or []
    extra_files = tuple(ExtraFile(**e) for e in raw_extra)
    return QuantFile(**raw_quant, extra_files=extra_files)


def load_config(path: str | Path) -> LadderConfig:
    raw = yaml.safe_load(Path(path).read_text())
    quants = tuple(_load_quant(q) for q in raw["quants"])
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
