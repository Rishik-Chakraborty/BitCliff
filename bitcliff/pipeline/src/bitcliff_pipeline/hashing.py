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


def item_set_sha256(items) -> str:
    """Deterministic hash over an item set: sorted item ids + each item's
    token content (prompt_tokens if the item was built in token space, else
    its text prompt) + its gold answer(s). Order-independent (sorted by id
    first) so the same item set hashes identically regardless of
    build/iteration order.

    Moved here (pre-rerun hardening, OPEN_QUESTIONS §8) from
    `scripts/calibrate_f16.py`, which originated it for its own
    measurements.jsonl bookkeeping -- this is now the one implementation,
    reused by `scripts/calibrate_f16.py`, `src/bitcliff_pipeline/registered.py`'s
    derived-hash provenance, and the boot-time item-set gate in `__main__`.
    """
    import hashlib
    import json

    h = hashlib.sha256()
    for it in sorted(items, key=lambda i: i.id):
        h.update(it.id.encode())
        h.update(b"\x00")
        if it.prompt_tokens is not None:
            h.update(json.dumps(list(it.prompt_tokens)).encode())
        else:
            h.update(it.prompt.encode())
        h.update(b"\x00")
        h.update(json.dumps(sorted(it.expected or ())).encode())
        h.update(b"\x01")
    return h.hexdigest()
