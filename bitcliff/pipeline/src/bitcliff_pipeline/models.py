from pathlib import Path

from huggingface_hub import hf_hub_download

from .config import LadderConfig
from .hashing import ManifestMismatch, sha256_file


def _verify_pin(path: Path, expected_sha256: str | None) -> None:
    """0B P3: verify a downloaded (or already-present) file against its
    config-pinned sha256 (from the committed reference-manifests/*.json),
    when one is set. A mismatch is a hard stop, per RUN_0B.md §4 ("a
    mismatch is a hard stop -- uploader re-quantized... STOP instance, log,
    ask") -- distinct from hashing.verify_manifest's run-local check, which
    only detects drift between a run's own download and generate stages,
    not drift from the registered manifest."""
    if expected_sha256 is None:
        return
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ManifestMismatch(
            f"{path.name}: config-pinned sha256 {expected_sha256[:12]}... "
            f"does not match the file on disk ({actual[:12]}...) -- the "
            f"reference-manifests/*.json pin and the actual file disagree "
            f"(re-quantized upload or a corrupted download); STOP and check "
            f"before using this file (RUN_0B.md §4)"
        )


def ensure_quants(
    config: LadderConfig, models_dir: Path, downloader=hf_hub_download
) -> dict[str, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for q in config.quants:
        repo = q.hf_repo or config.hf_repo
        dest = models_dir / q.filename
        if not dest.exists():
            downloader(repo_id=repo, filename=q.filename, local_dir=str(models_dir))
        _verify_pin(dest, q.sha256)
        for extra in q.extra_files:
            edest = models_dir / extra.filename
            if not edest.exists():
                downloader(repo_id=repo, filename=extra.filename, local_dir=str(models_dir))
            _verify_pin(edest, extra.sha256)
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
