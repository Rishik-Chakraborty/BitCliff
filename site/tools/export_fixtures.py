#!/usr/bin/env python3
"""Export BitCliff playground fixtures from the pilot-0a pipeline run.

Reads (read-only, never writes outside site/):
  bitcliff/pipeline/runs/pilot-0a/items.jsonl
  bitcliff/pipeline/runs/pilot-0a/grades.jsonl
  bitcliff/pipeline/runs/pilot-0a/outputs/*.jsonl
  bitcliff/pipeline/runs/pilot-0a/manifest.json
  bitcliff/pipeline/configs/qwen2.5-1.5b-pilot.yaml   (best-effort, stdlib-only parse)

Writes:
  site/fixtures/pilot-0a.json

Stdlib only. Deterministic output (sorted keys, fixed key order via a
custom encoder is unnecessary -- json.dump(sort_keys=True) is enough for
byte-for-byte reproducibility across re-runs on unchanged inputs).

Embargo / scope rules enforced here (see claude/IDEA.md, OPEN_QUESTIONS.md,
PILOT_NOTES.md, and the overnight work order):
  - Only pilot-0a is ever ingested. No other run directory is read.
  - Only items whose suite is in {arithmetic, spectacle} are exported.
    The pilot's "retrieval" suite is a retired placeholder (see
    PILOT_NOTES.md item 2) and must never reach the fixtures or the site.
  - Rungs marked spectacle_only in the manifest (the in-house
    IQ2_XXS/IQ1_M/IQ1_S re-quantizations) are exported with that flag
    intact so the UI can label them, never hide it.
  - multivalue2 / longctx prompts do not exist in pilot-0a at all; nothing
    here needs to filter for them, but the suite allowlist above is the
    structural guarantee that only arithmetic/spectacle ever ships.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "bitcliff" / "pipeline" / "runs" / "pilot-0a"
CONFIG_PATH = REPO_ROOT / "bitcliff" / "pipeline" / "configs" / "qwen2.5-1.5b-pilot.yaml"
OUT_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "pilot-0a.json"

RUN_ID = "pilot-0a"

# Canonical ladder order, full precision first, deranged zone last.
# This mirrors configs/qwen2.5-1.5b-pilot.yaml `quants:` order (F16 is the
# always-present reference, prepended ahead of the configured quant list).
LADDER_ORDER = [
    "F16",
    "Q8_0",
    "Q6_K",
    "Q5_K_M",
    "Q4_K_M",
    "Q3_K_M",
    "Q2_K",
    "IQ2_M",
    "IQ2_XXS",
    "IQ1_M",
    "IQ1_S",
]

ALLOWED_SUITES = {"arithmetic", "spectacle"}

IN_HOUSE_TAG = (
    "in-house re-quantization — spectacle only, not a download recommendation"
)

DISCLAIMER = (
    "Exploratory pilot data — for demonstration; not the registered measurements."
)


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def human_size(num_bytes: int) -> str:
    """Format bytes as a short human-readable size (binary units)."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def parse_config_quants(path: Path) -> list[dict]:
    """Best-effort stdlib-only parse of the flow-style `quants:` list in
    configs/qwen2.5-1.5b-pilot.yaml. Used only as a cross-check against
    manifest.json; the manifest remains the source of truth for rung
    metadata. Returns [] if the file is missing or the format is
    unrecognized -- this is diagnostic, not load-bearing.
    """
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    quants = []
    # Lines look like:
    #   - {label: Q8_0,   filename: Qwen2.5-1.5B-Instruct-Q8_0.gguf,   uploader: bartowski, imatrix: true}
    for m in re.finditer(r"\{([^}]*)\}", text):
        body = m.group(1)
        entry = {}
        for field in body.split(","):
            if ":" not in field:
                continue
            k, v = field.split(":", 1)
            k = k.strip()
            v = v.strip()
            if v in ("true", "false"):
                v = v == "true"
            entry[k] = v
        if "label" in entry:
            quants.append(entry)
    return quants


def build_rung_metadata(manifest: dict, config_quants: list[dict]) -> dict:
    config_by_label = {q["label"]: q for q in config_quants if "label" in q}
    rungs = {}
    for label, entry in manifest.items():
        sha256 = entry.get("sha256", "")
        rungs[label] = {
            "label": label,
            "uploader": entry.get("uploader"),
            "imatrix": bool(entry.get("imatrix", False)),
            "spectacle_only": bool(entry.get("spectacle_only", False)),
            "size_bytes": entry.get("size_bytes"),
            "size_human": human_size(entry.get("size_bytes", 0)),
            "sha256_short": sha256[:12],
            "filename": entry.get("filename"),
        }
        cfg = config_by_label.get(label)
        if cfg and cfg.get("uploader") and cfg["uploader"] != entry.get("uploader"):
            print(
                f"WARNING: uploader mismatch for {label}: "
                f"manifest={entry.get('uploader')!r} config={cfg['uploader']!r}",
                file=sys.stderr,
            )
    return rungs


def main() -> int:
    items_path = RUN_DIR / "items.jsonl"
    grades_path = RUN_DIR / "grades.jsonl"
    manifest_path = RUN_DIR / "manifest.json"
    outputs_dir = RUN_DIR / "outputs"

    for p in (items_path, grades_path, manifest_path, outputs_dir):
        if not p.exists():
            print(f"ERROR: required input missing: {p}", file=sys.stderr)
            return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest_labels = {k for k in manifest.keys() if not k.startswith("_")}  # Exclude reserved metadata keys (e.g., _run_config)
    ladder_labels = set(LADDER_ORDER)
    if manifest_labels != ladder_labels:
        print(
            "ERROR: manifest rung labels do not match the hardcoded LADDER_ORDER.\n"
            f"  manifest only: {sorted(manifest_labels - ladder_labels)}\n"
            f"  ladder only:   {sorted(ladder_labels - manifest_labels)}",
            file=sys.stderr,
        )
        return 1

    config_quants = parse_config_quants(CONFIG_PATH)
    rung_meta = build_rung_metadata(manifest, config_quants)

    items = read_jsonl(items_path)
    grades = read_jsonl(grades_path)

    # Filter to allowed suites only (excludes the retired "retrieval"
    # placeholder suite -- this is the embargo enforcement point).
    items_by_id = {}
    excluded_suites = set()
    for it in items:
        if it["suite"] not in ALLOWED_SUITES:
            excluded_suites.add(it["suite"])
            continue
        items_by_id[it["id"]] = it

    # Load per-rung outputs, keyed by (item_id) -> text/finish_reason,
    # restricted to allowed items only.
    outputs_by_rung: dict[str, dict[str, dict]] = {}
    for label in LADDER_ORDER:
        out_path = outputs_dir / f"{label}.jsonl"
        if not out_path.exists():
            print(f"ERROR: missing outputs file for rung {label}: {out_path}", file=sys.stderr)
            return 1
        recs = read_jsonl(out_path)
        by_id = {}
        for r in recs:
            if r["item_id"] not in items_by_id:
                continue
            by_id[r["item_id"]] = r
        outputs_by_rung[label] = by_id

    # Index grades by (item_id, quant_label), restricted to allowed items.
    grades_index: dict[tuple[str, str], dict] = {}
    for g in grades:
        if g["item_id"] not in items_by_id:
            continue
        grades_index[(g["item_id"], g["quant_label"])] = g

    fixture_items = []
    missing_rungs = []
    for item_id in sorted(items_by_id.keys()):
        it = items_by_id[item_id]
        rung_data = {}
        for label in LADDER_ORDER:
            out = outputs_by_rung[label].get(item_id)
            grd = grades_index.get((item_id, label))
            if out is None or grd is None:
                missing_rungs.append((item_id, label))
                continue
            rung_data[label] = {
                "text": out["text"],
                "finish_reason": out.get("finish_reason"),
                "state": grd["state"],
                "truncated": bool(grd["truncated"]),
                "loop": bool(grd["loop"]),
                "divergence": grd["divergence"],
            }
        fixture_items.append(
            {
                "id": item_id,
                "suite": it["suite"],
                "prompt": it["prompt"],
                "expected": it.get("expected"),
                "rungs": rung_data,
            }
        )

    if missing_rungs:
        print(
            f"ERROR: {len(missing_rungs)} (item, rung) pairs missing output/grade data. "
            f"First few: {missing_rungs[:5]}",
            file=sys.stderr,
        )
        return 1

    # Pull generation settings from any single output record (they are
    # identical across the deterministic pilot run) for display purposes.
    sample_out = next(iter(outputs_by_rung["F16"].values()))
    generation_settings = sample_out.get("gen_settings", {})
    machine = sample_out.get("machine")

    fixture = {
        "run_id": RUN_ID,
        "disclaimer": DISCLAIMER,
        "in_house_tag": IN_HOUSE_TAG,
        "suites": sorted(ALLOWED_SUITES),
        "excluded_suites": sorted(excluded_suites),
        "ladder": LADDER_ORDER,
        "rungs": rung_meta,
        "generation_settings": generation_settings,
        "machine": machine,
        "items": fixture_items,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(fixture, f, sort_keys=True, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {OUT_PATH} ({len(fixture_items)} items, {len(LADDER_ORDER)} rungs)")
    print(f"Excluded suites (embargoed/retired): {sorted(excluded_suites)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
