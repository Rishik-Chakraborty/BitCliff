#!/usr/bin/env python3
"""Enumerate reference GGUF files on the Hugging Face Hub — NO downloads.

For the BitCliff freeze-execution plan (Task 5): builds verified file-list
manifests (path, size, sha256-via-lfs.oid) for the reference GGUF quant
builds the shootout/comparison suites are pinned against, without pulling
any multi-gigabyte weights onto disk.

Uses only the Python standard library (urllib), per the task brief.

Data sources (HF Hub REST API, unauthenticated, public repos only):
  - GET https://huggingface.co/api/models/{repo}
        -> repo metadata; "sha" is the current revision (commit) sha of the
           default branch (main).
  - GET https://huggingface.co/api/models/{repo}/tree/main?recursive=true
        -> flat file listing; LFS-tracked files (all .gguf files are LFS)
           carry an "lfs": {"oid": <sha256 hex>, "size": <bytes>} object.
  - GET https://huggingface.co/api/models?search=<query>&limit=<n>
        -> used only as a fallback when a hardcoded repo name 404s, to find
           the correct current name.

Output: reference-manifests/*.json (committed as freeze evidence).
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

API_ROOT = "https://huggingface.co/api/models"
OUT_DIR = Path(__file__).resolve().parent.parent / "reference-manifests"

USER_AGENT = "bitcliff-pipeline-enumerate-reference-files/1.0"


def _get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_status(url: str) -> int | None:
    """Return the HTTP status for a HEAD-ish check without raising, or None
    on a non-HTTP error (network failure etc.)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def repo_exists(repo: str) -> bool:
    status = _http_status(f"{API_ROOT}/{repo}")
    return status == 200


def search_models(query: str, limit: int = 20) -> list[dict]:
    url = f"{API_ROOT}?search={urllib.parse.quote(query)}&limit={limit}"
    return _get_json(url)  # type: ignore[return-value]


def get_repo_info(repo: str) -> dict:
    return _get_json(f"{API_ROOT}/{repo}")  # type: ignore[return-value]


def get_tree(repo: str, recursive: bool = True) -> list[dict]:
    suffix = "?recursive=true" if recursive else ""
    return _get_json(f"{API_ROOT}/{repo}/tree/main{suffix}")  # type: ignore[return-value]


@dataclass
class GgufEntry:
    path: str
    size: int
    sha256: str | None
    uploader: str | None = None
    imatrix: bool | None = None

    def to_json(self) -> dict:
        d = {"path": self.path, "size_bytes": self.size, "sha256": self.sha256}
        if self.uploader is not None:
            d["uploader"] = self.uploader
        if self.imatrix is not None:
            d["imatrix"] = self.imatrix
        return d


def gguf_files(repo: str) -> list[GgufEntry]:
    tree = get_tree(repo)
    out = []
    for entry in tree:
        path = entry.get("path", "")
        if not path.endswith(".gguf"):
            continue
        lfs = entry.get("lfs") or {}
        size = lfs.get("size", entry.get("size"))
        sha256 = lfs.get("oid")
        out.append(GgufEntry(path=path, size=size, sha256=sha256))
    return out


def resolve_repo(hardcoded_repo: str, search_query: str) -> tuple[str, str | None]:
    """Return (repo_id_to_use, substitution_note_or_None).

    If `hardcoded_repo` 404s (or otherwise doesn't return 200), search the HF
    API for `search_query` and, if a plausible replacement is found, use it
    and return a note describing the substitution.
    """
    if repo_exists(hardcoded_repo):
        return hardcoded_repo, None

    note = f"'{hardcoded_repo}' did not resolve (non-200 on GET {API_ROOT}/{hardcoded_repo})."
    results = search_models(search_query, limit=20)
    candidates = [r["id"] for r in results if "id" in r]
    print(f"WARNING: {note} Search candidates for {search_query!r}: {candidates}", file=sys.stderr)

    # Heuristic auto-pick: same org namespace as the hardcoded repo, shortest
    # id (fine-tune/merge/LoRA repos tend to have long, qualifier-laden
    # names), and confirmed to actually exist. Reviewed and recorded as a
    # substitution_note in the output manifest either way — never silent.
    org = hardcoded_repo.split("/", 1)[0]
    same_org = [c for c in candidates if c.startswith(f"{org}/")]
    same_org.sort(key=len)
    for cand in same_org:
        if repo_exists(cand):
            print(f"AUTO-SUBSTITUTED: '{hardcoded_repo}' -> '{cand}'", file=sys.stderr)
            return cand, f"{note} Auto-substituted with '{cand}' (same org, shortest matching name)."

    raise RuntimeError(
        f"{note} No automatic substitution rule matched; resolve manually. "
        f"Search candidates: {candidates}"
    )


def build_manifest_entry(repo: str, requested_repo: str | None = None) -> dict:
    info = get_repo_info(repo)
    revision_sha = info.get("sha")
    files = [e.to_json() for e in gguf_files(repo)]
    entry = {
        "repo": repo,
        "revision_sha": revision_sha,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "gguf_files": files,
        "gguf_file_count": len(files),
    }
    if requested_repo is not None and requested_repo != repo:
        entry["substitution_note"] = (
            f"requested repo '{requested_repo}' was not found/accessible; "
            f"substituted '{repo}' (found via HF search API)"
        )
    return entry


def write_manifest(filename: str, data: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {out_path} ({len(data.get('gguf_files', data.get('repos', [])))} top-level entries)")


# ---------------------------------------------------------------------------
# Manifest builders
# ---------------------------------------------------------------------------


def build_llama_3_1_8b_bartowski() -> None:
    repo = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    entry = build_manifest_entry(repo)
    write_manifest("llama-3.1-8b-bartowski.json", entry)


def build_qwen2_5_7b_bartowski() -> None:
    repo = "bartowski/Qwen2.5-7B-Instruct-GGUF"
    entry = build_manifest_entry(repo)
    write_manifest("qwen2.5-7b-bartowski.json", entry)


def build_qwen2_5_7b_official() -> None:
    repo = "Qwen/Qwen2.5-7B-Instruct-GGUF"
    entry = build_manifest_entry(repo)
    entry["note"] = (
        "Official Qwen GGUFs are sharded for several quant levels (multi-part "
        "*-NNNNN-of-MMMMM.gguf files); all parts are recorded above. Filenames "
        "are lowercase, unlike bartowski's repos."
    )
    write_manifest("qwen2.5-7b-official.json", entry)


SHOOTOUT_QUANTS = ("Q4_K_M", "Q3_K_M")


def _filter_shootout_quants(files: list[GgufEntry]) -> list[GgufEntry]:
    """Keep only files whose name contains exactly Q4_K_M or Q3_K_M as a
    quant-level token (not a substring of a longer token like IQ4_K_M or
    Q4_K_M_L). The token must be immediately preceded by '.', '-', or '_'
    (never by another letter/digit) and immediately followed by '.' (the
    ".gguf" extension) or '-' (a multi-part shard suffix like
    "-00001-of-00002.gguf").
    """
    import re

    out = []
    for f in files:
        for q in SHOOTOUT_QUANTS:
            pattern = rf"(?<![A-Za-z0-9]){re.escape(q)}(?=[.\-])"
            if re.search(pattern, f.path):
                out.append(f)
                break
    return out


def build_shootout_8b() -> None:
    repos_spec = [
        {
            "hardcoded": "unsloth/Meta-Llama-3.1-8B-Instruct-GGUF",
            "search_query": "unsloth Llama-3.1-8B-Instruct GGUF",
            "uploader": "unsloth",
            "imatrix": False,  # unsloth's "UD" dynamic quants use their own
            # calibration; the plain Q4_K_M/Q3_K_M files here are standard
            # (non-imatrix) llama.cpp quants.
        },
        {
            "hardcoded": "mradermacher/Meta-Llama-3.1-8B-Instruct-GGUF",
            "search_query": "mradermacher Meta-Llama-3.1-8B-Instruct GGUF",
            "uploader": "mradermacher",
            "imatrix": False,  # static (non-imatrix) quants
        },
        {
            "hardcoded": "mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF",
            "search_query": "mradermacher Meta-Llama-3.1-8B-Instruct i1 GGUF",
            "uploader": "mradermacher",
            "imatrix": True,  # "i1" = imatrix-calibrated quants
        },
    ]

    repo_entries = []
    for spec in repos_spec:
        hardcoded = spec["hardcoded"]
        if repo_exists(hardcoded):
            repo = hardcoded
            substitution_note = None
        else:
            repo, substitution_note = resolve_repo(hardcoded, spec["search_query"])

        info = get_repo_info(repo)
        revision_sha = info.get("sha")
        all_files = gguf_files(repo)
        filtered = _filter_shootout_quants(all_files)
        for f in filtered:
            f.uploader = spec["uploader"]
            f.imatrix = spec["imatrix"]

        repo_entry = {
            "requested_repo": hardcoded,
            "repo": repo,
            "revision_sha": revision_sha,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "uploader": spec["uploader"],
            "imatrix": spec["imatrix"],
            "quant_filter": list(SHOOTOUT_QUANTS),
            "gguf_files": [f.to_json() for f in filtered],
            "gguf_file_count": len(filtered),
            "total_gguf_files_in_repo": len(all_files),
        }
        if substitution_note is not None:
            repo_entry["substitution_note"] = substitution_note
        repo_entries.append(repo_entry)

    manifest = {
        "scope": "8B only, Q4_K_M + Q3_K_M only (ruling 5)",
        "repos": repo_entries,
    }
    write_manifest("shootout-8b.json", manifest)


def main() -> None:
    build_llama_3_1_8b_bartowski()
    build_qwen2_5_7b_bartowski()
    build_qwen2_5_7b_official()
    build_shootout_8b()


if __name__ == "__main__":
    main()
