from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class QuantFile:
    label: str
    filename: str
    uploader: str
    imatrix: bool
    spectacle_only: bool = False


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
