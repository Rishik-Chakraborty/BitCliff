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
