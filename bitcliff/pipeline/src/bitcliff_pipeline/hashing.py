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
        if label.startswith("_"):
            # Reserved metadata (e.g. "_run_config"), not a rung entry.
            continue
        actual = sha256_file(files[label])
        if actual != entry["sha256"]:
            raise ManifestMismatch(
                f"{label}: expected {entry['sha256'][:12]}..., got {actual[:12]}..."
            )
