#!/usr/bin/env python3
"""package_dataset.py — PREREG §11 published per-item dataset packager.

Builds the registered public dataset from a run directory (`items.jsonl`,
`grades.jsonl`, `outputs/*.jsonl`, `manifest.json`) with embargo exclusions
ENFORCED IN CODE, not by convention:

- `longctx_retrieval` items: before packaging, a positive corpus-provenance
  check requires the manifest's `_run_config.suites.longctx_retrieval.corpus_sha256`
  to match the registered 2b corpus (PG-1184 stripped); 2a's corpus is
  refused outright (PREREG §3.1, pre-0B ticket). The raw prompt text and
  the token-id prompt array are never read by the record-builder for that
  suite (see `build_longctx_metadata`, which only ever looks at
  `item["id"]`, `item["expected"]`, an explicit allowlist of extra keys,
  and a `reconstructed` dict of already-safe fields — never
  `item["prompt"]` / `item["prompt_tokens"]`). The registered `key` /
  `depths` / `n_answer_tokens` metadata (PREREG §11) is not present on real
  run items, so it is reconstructed by rebuilding the item set through the
  same vendored-generator path `--verify-recipe` uses (tokenizer + a corpus
  file verified against the registered sha256) and merging those three
  fields in by id, hard-failing (`ReconstructionMismatch`) if
  reconstruction disagrees with the run. After every file is written, a
  post-write scanner (`scan_for_embargoed_content`) re-reads every byte of
  every produced file and hard-fails if any longctx item's prompt text or
  token array shows up anywhere; the gold match strings (`expected`) are
  published per user ruling 4a and are deliberately outside the scanner's
  forbidden list (a bare 4-digit gold is not prompt/token-array leakage
  even though it also appears inside the excluded prompt text). Published:
  item id, `expected`, a best-effort `config` parsed from the item id, any
  allowlisted extra keys the item happens to carry, the reconstructed
  `key`/`depths`/`n_answer_tokens`, model outputs, grades, and
  `RECONSTRUCTION.md` (PREREG §3.1/§11).
- `arithmetic_twins` (and anything staged under a `private/` path inside the
  run directory): the packager REFUSES outright — raises before writing
  anything. Twins are embargoed until the paper extension (PREREG §3.3).
- The pilot's retired `retrieval` suite (superseded by `longctx_retrieval`,
  no conclusions survive — PREREG §1, §12): excluded, noted in the package
  manifest's exclusion list.
- Everything else (`arithmetic`, `spectacle`, `factual_qa`): published in
  full, including prompts.

CLI:
    uv run python scripts/package_dataset.py <run_dir> --out dist/dataset-<run-id>/ \
        [--verify-recipe] [--tokenizer-path PATH] [--corpus-path PATH]

    --tokenizer-path / --corpus-path are REQUIRED if and only if the run
    contains longctx_retrieval items (used to reconstruct registered
    key/depths/n_answer_tokens metadata, and, with --verify-recipe, to
    verify the reconstruction digest). Never used, and not required, for a
    run with no longctx_retrieval items.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bitcliff_pipeline import registered  # noqa: E402

PREREG_COMMIT = "5e6882b7a10c5e4670052855380e8646911db5f3"

RETIRED_RETRIEVAL_SUITE = "retrieval"
LONGCTX_SUITE = "longctx_retrieval"
TWINS_SUITE = "arithmetic_twins"

RETIRED_RETRIEVAL_REASON = (
    "pilot's retired placeholder suite, superseded by longctx_retrieval "
    "(the multivalue2 task); deleted, not renamed, no conclusion from it "
    "survives (PREREG §1, §12)"
)

# Published rung metadata, per PREREG §11's "rung metadata from the
# manifest" list — deliberately just these five fields.
RUNG_MANIFEST_FIELDS = ("uploader", "imatrix", "spectacle_only", "sha256")

# Extra per-item keys a longctx_retrieval item dict is allowed to carry into
# the published record, IF present (PREREG §11: "item id, key, depths,
# n_answer_tokens, config"). This is an explicit allowlist, not a denylist:
# `build_longctx_metadata` below never reads "prompt" or "prompt_tokens" off
# the item dict under any circumstance — the record-builder has no code path
# that could copy them, allowlist or not.
LONGCTX_SAFE_EXTRA_KEYS = (
    "key",
    "depth",
    "depths",
    "config",
    "variant",
    "target_tokens",
    "seed",
    "index",
    "n_answer_tokens",
)

# Registered corpus hashes for the two longctx configurations (PREREG §3.1),
# imported from `bitcliff_pipeline.registered` (the single home of every
# PREREG-registered constant, pre-rerun hardening, OPEN_QUESTIONS §8) rather
# than restated here. Names kept as this module's original
# CORPUS_2A_SHA256 / CORPUS_2B_STRIPPED_SHA256 (tests reference `pkg.
# CORPUS_2A_SHA256` / `pkg.CORPUS_2B_STRIPPED_SHA256` by these names).
# 2a (PG-essays): prompts are "never displayed on the site and never
# published; outputs and statistics only" — any run recording this corpus is
# refused. 2b (PG-1184 stripped) is the only publishable longctx corpus.
CORPUS_2A_SHA256 = registered.CORPUS_2A_SHA256

# 2b corpus pointer, PREREG §3.1 / CORPUS_MANIFEST.md §1.
CORPUS_2B_URL = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"
CORPUS_2B_NAME = 'Project Gutenberg #1184, "The Count of Monte Cristo"'
CORPUS_2B_STRIPPED_SHA256 = registered.CORPUS_2B_SHA256

LONGCTX_ID_RE = re.compile(
    r"^longctx_retrieval-(?P<variant>.+)-t(?P<target_tokens>\d+)"
    r"-s(?P<seed>\d+)-(?P<index>\d+)$"
)


class EmbargoViolation(RuntimeError):
    """Raised when a run cannot be packaged (twins present) or the post-write
    scanner finds embargoed content in the produced tree."""


class ReconstructionMismatch(RuntimeError):
    """Raised when the item set rebuilt via the vendored generator does not
    line up with the run's own longctx_retrieval items (by id). This is a
    packaging error, not a warning: reconstruction disagreeing with the run
    means the registered key/depths/n_answer_tokens metadata cannot be
    trusted, and PREREG §11 requires it to be published."""


# ---------------------------------------------------------------------------
# Loading the run
# ---------------------------------------------------------------------------


@dataclass
class RunData:
    manifest: dict
    items: list[dict]
    grades: list[dict]
    outputs: dict[str, list[dict]] = field(default_factory=dict)
    run_config: dict | None = None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def load_run(run_dir: Path) -> RunData:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    run_config = manifest.pop("_run_config", None)
    items = _read_jsonl(run_dir / "items.jsonl")
    grades = _read_jsonl(run_dir / "grades.jsonl")
    outputs: dict[str, list[dict]] = {}
    outputs_dir = run_dir / "outputs"
    if outputs_dir.exists():
        for p in sorted(outputs_dir.glob("*.jsonl")):
            outputs[p.stem] = _read_jsonl(p)
    return RunData(manifest=manifest, items=items, grades=grades, outputs=outputs, run_config=run_config)


# ---------------------------------------------------------------------------
# Embargo refusal (twins, private/)
# ---------------------------------------------------------------------------


def refuse_embargoed_run(run: RunData, run_dir: Path) -> None:
    """Raise EmbargoViolation before anything is written if this run cannot
    be packaged at all: it contains the twins suite, or a `private/` path is
    staged inside the run directory."""
    suites = {it.get("suite") for it in run.items} | {
        g.get("suite") for g in run.grades
    }
    if TWINS_SUITE in suites:
        raise EmbargoViolation(
            f"refusing to package {run_dir}: run contains suite "
            f"{TWINS_SUITE!r} — twins are embargoed until the paper "
            f"extension ships (PREREG §3.3); this packager never publishes "
            f"them"
        )
    if run_dir.exists():
        for p in run_dir.rglob("*"):
            if "private" in p.relative_to(run_dir).parts:
                raise EmbargoViolation(
                    f"refusing to package {run_dir}: found a 'private' path "
                    f"({p}) inside the run directory — embargoed content "
                    f"must never enter the published package"
                )


# ---------------------------------------------------------------------------
# Per-item metadata builders
# ---------------------------------------------------------------------------


def parse_longctx_id(item_id: str) -> dict | None:
    m = LONGCTX_ID_RE.match(item_id)
    if not m:
        return None
    d = m.groupdict()
    return {
        "variant": d["variant"],
        "target_tokens": int(d["target_tokens"]),
        "seed": int(d["seed"]),
        "index": int(d["index"]),
    }


def build_longctx_metadata(item: dict, reconstructed: dict | None = None) -> dict:
    """PREREG §11 STRUCTURAL exclusion: this function's only inputs are
    `item["id"]`, `item["expected"]`, an explicit allowlist of extra keys
    (`LONGCTX_SAFE_EXTRA_KEYS`), and an already-safe `reconstructed` dict of
    `key` / `depths` / `n_answer_tokens`. It never reads `item["prompt"]` or
    `item["prompt_tokens"]` — there is no code path here that could copy
    them into the published record, regardless of what else `item` holds.

    `expected` (the gold match strings / passcodes) IS published per user
    ruling 4a (2026-08-27, resolving the packager's own OPEN_QUESTIONS §4a):
    it comes straight off the run's items.jsonl, no reconstruction needed.
    This is a deliberate, separate read from the prompt/prompt_tokens
    exclusion above — a bare 4-digit gold string is not the document text or
    the token-id array, so publishing it does not touch the embargo.

    Additive, not all-or-nothing: the id-parsed `config` is always included
    when the id matches the expected shape, then `expected` (if present) is
    added, then any allowlisted extras present on `item` are overlaid, then
    the reconstructed registered metadata (`key`, `depths`,
    `n_answer_tokens` — PREREG §11) is overlaid on top.
    """
    meta: dict = {"id": item["id"]}
    parsed = parse_longctx_id(item["id"])
    if parsed is not None:
        meta["config"] = parsed
    if item.get("expected") is not None:
        meta["expected"] = item["expected"]
    extras = {k: item[k] for k in LONGCTX_SAFE_EXTRA_KEYS if k in item}
    meta.update(extras)
    if reconstructed is not None:
        for f in ("key", "depths", "n_answer_tokens"):
            if reconstructed.get(f) is not None:
                meta[f] = reconstructed[f]
    return meta


def build_published_metadata(item: dict) -> dict:
    """Non-longctx suites (arithmetic, spectacle, factual_qa) publish the
    item in full, including the prompt (PREREG §11)."""
    rec = dict(item)
    rec.pop("prompt_tokens", None)  # always null/absent for these suites
    return rec


# ---------------------------------------------------------------------------
# Joining items + outputs + grades + rung manifest into published records
# ---------------------------------------------------------------------------


def build_rung(
    quant_label: str, manifest_entry: dict, output_rec: dict, grade_rec: dict
) -> dict:
    rung = {"label": quant_label}
    for f in RUNG_MANIFEST_FIELDS:
        rung[f] = manifest_entry.get(f)
    rung["text"] = output_rec.get("text")
    rung["state"] = grade_rec.get("state")
    rung["truncated"] = grade_rec.get("truncated")
    rung["loop"] = grade_rec.get("loop")
    rung["divergence"] = grade_rec.get("divergence")
    return rung


def build_suite_records(
    suite: str,
    items: list[dict],
    outputs: dict[str, list[dict]],
    grades: list[dict],
    manifest: dict,
    longctx_reconstruction: dict[str, dict] | None = None,
) -> list[dict]:
    outputs_by_label = {
        label: {r["item_id"]: r for r in recs} for label, recs in outputs.items()
    }
    grades_by_item_quant = {
        (g["item_id"], g["quant_label"]): g for g in grades if g.get("suite") == suite
    }

    records = []
    for item in sorted(items, key=lambda it: it["id"]):
        if suite == LONGCTX_SUITE:
            reconstructed = (longctx_reconstruction or {}).get(item["id"])
            rec = build_longctx_metadata(item, reconstructed)
        else:
            rec = build_published_metadata(item)
        rec["suite"] = suite

        rungs = {}
        for quant_label in sorted(manifest.keys()):
            out_rec = outputs_by_label.get(quant_label, {}).get(item["id"])
            grade_rec = grades_by_item_quant.get((item["id"], quant_label))
            if out_rec is None or grade_rec is None:
                continue
            rungs[quant_label] = build_rung(
                quant_label, manifest[quant_label], out_rec, grade_rec
            )
        rec["rungs"] = rungs
        records.append(rec)
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Longctx verification digest (ids + token ids + golds)
# ---------------------------------------------------------------------------


def longctx_verification_digest(entries) -> str:
    """sha256 over (item_id, prompt_tokens, expected) triples, sorted by
    item_id — "a digest over ids+token ids+golds". Deliberately independent
    of any field ordering in the source files."""
    h = hashlib.sha256()
    for item_id, prompt_tokens, expected in sorted(entries, key=lambda e: e[0]):
        h.update(item_id.encode("utf-8"))
        h.update(b"|")
        h.update(json.dumps(list(prompt_tokens or [])).encode("utf-8"))
        h.update(b"|")
        h.update(json.dumps(list(expected or [])).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _longctx_entries_from_raw_items(items: list[dict]) -> list[tuple]:
    return [
        (it["id"], it.get("prompt_tokens"), it.get("expected"))
        for it in items
        if it.get("suite") == LONGCTX_SUITE
    ]


# ---------------------------------------------------------------------------
# Longctx metadata reconstruction (key / depths / n_answer_tokens)
#
# PREREG §11 registers these three fields as published per-item metadata for
# longctx_retrieval, but they are not present on real run items (only id,
# suite, prompt, expected, prompt_tokens are). They are recovered by
# rebuilding the item set through the vendored generator against a verified
# corpus — the SAME reconstruction path --verify-recipe uses
# (`_rebuild_longctx_raw_items`) — and merged in by id. A run item with no
# matching rebuilt item is a hard failure: reconstruction disagreeing with
# the run is a packaging error, not something to silently skip.
# ---------------------------------------------------------------------------


def _longctx_wrapper_id(variant: str, target_tokens: int, seed: int, index: int) -> str:
    return f"longctx_retrieval-{variant}-t{target_tokens}-s{seed}-{index:04d}"


def _rebuild_longctx_raw_items(
    tokenizer_path,
    corpus_path,
    corpus_sha256: str,
    n_items: int,
    seed: int,
    variant: str,
    target_tokens: int,
) -> list[dict]:
    """Rebuild raw multivalue2 item dicts (the vendored generator's own
    dicts, with `needle_texts` / `needle_depths` / `n_answer_tokens` —
    richer than the `EvalItem`s `bitcliff_pipeline.suites.longctx_retrieval`
    exposes) from a tokenizer directory and a corpus text file, verifying
    the corpus against `corpus_sha256` first. This is the one function that
    ever touches the real corpus text; monkeypatch this in tests to avoid
    needing a real tokenizer/corpus on disk."""
    corpus_text = Path(corpus_path).read_text(encoding="utf-8")
    digest = hashlib.sha256(corpus_text.encode("utf-8")).hexdigest()
    if digest != corpus_sha256:
        raise ValueError(
            f"corpus sha256 {digest} != expected {corpus_sha256}: refusing "
            f"to reconstruct longctx metadata from an unverified corpus"
        )
    tokenizer = _load_tokenizer_offline(Path(tokenizer_path))

    from bitcliff_pipeline.vendor import generate_multivalue2 as mv2

    token_ids = tokenizer(corpus_text, add_special_tokens=False)["input_ids"]
    # Overwrite, not merely pre-populate — see
    # bitcliff_pipeline.suites.longctx_retrieval.build_items for why.
    mv2._STREAM_CACHE[tokenizer.name_or_path] = token_ids
    return mv2.build_items(tokenizer, variant, n_items, target_tokens, seed)


def _extract_key(needle_texts: list[str]):
    """Recover the NATO-alphabet key name(s) from raw needle texts (e.g.
    "Tariq: 8721\\n") WITHOUT ever reading the passcode value on the same
    line — only the substring before the colon is kept. Returns a single
    string if every needle shares one key (multivalue2's shape), a list if
    several distinct keys appear, or None if nothing parses."""
    keys: list[str] = []
    for t in needle_texts or []:
        head = t.strip().split(":", 1)
        if len(head) == 2 and head[0].strip():
            name = head[0].strip()
            if name not in keys:
                keys.append(name)
    if not keys:
        return None
    return keys[0] if len(keys) == 1 else keys


def _reconstruct_longctx_metadata(
    longctx_items: list[dict],
    longctx_combos: list[dict],
    tokenizer_path,
    corpus_path,
) -> dict[str, dict]:
    reconstruction: dict[str, dict] = {}
    for combo in longctx_combos:
        raw_items = _rebuild_longctx_raw_items(
            tokenizer_path,
            corpus_path,
            CORPUS_2B_STRIPPED_SHA256,
            n_items=combo["n_items"],
            seed=combo["seed"],
            variant=combo["variant"],
            target_tokens=combo["target_tokens"],
        )
        for raw in raw_items:
            rebuilt_id = _longctx_wrapper_id(
                combo["variant"], combo["target_tokens"], combo["seed"], raw["index"]
            )
            reconstruction[rebuilt_id] = {
                "key": _extract_key(raw.get("needle_texts", [])),
                "depths": list(raw.get("needle_depths", [])),
                "n_answer_tokens": raw.get("n_answer_tokens"),
            }

    missing = sorted(it["id"] for it in longctx_items if it["id"] not in reconstruction)
    if missing:
        raise ReconstructionMismatch(
            "longctx reconstruction disagrees with the run: no rebuilt item "
            f"matches run item id(s) {missing!r} — refusing to package "
            "(PREREG §11 requires key/depths/n_answer_tokens; reconstruction "
            "mismatch is a packaging error, not a warning)"
        )
    return reconstruction


# ---------------------------------------------------------------------------
# RECONSTRUCTION.md
# ---------------------------------------------------------------------------


def infer_tokenizer_id(manifest: dict) -> str | None:
    f16 = manifest.get("F16", {})
    filename = f16.get("filename")
    if not filename:
        return None
    stem = filename
    for suffix in (".gguf",):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    for marker in ("-f16", "-F16"):
        idx = stem.rfind(marker)
        if idx != -1:
            stem = stem[:idx]
    return stem


def write_reconstruction_md(
    out_dir: Path, longctx_combos: list[dict], tokenizer_id: str | None
) -> None:
    lines = [
        "# RECONSTRUCTION.md — longctx_retrieval item reconstruction recipe",
        "",
        "Per PREREG §11: multivalue2 prompts (question text and the",
        "token-id prompt array) are excluded from this published dataset.",
        "Bit-identical items are reconstructible from the pieces below.",
        "",
        "## Generator",
        "",
        "- Vendored generator (Apache-2.0): "
        "`bitcliff_pipeline.vendor.generate_multivalue2` "
        "(`src/bitcliff_pipeline/vendor/generate_multivalue2.py`).",
        "- Adapter used by this pipeline: "
        "`bitcliff_pipeline.suites.longctx_retrieval.build_items` "
        "(`src/bitcliff_pipeline/suites/longctx_retrieval.py`).",
        "",
        "## Tokenizer",
        "",
        f"- `{tokenizer_id or '(unknown — see run manifest.json)'}` "
        "(HF tokenizer; the GGUF tokenizer is asserted identical to it "
        "per PREREG §3.1 `assert_tokenizer_match`).",
        "",
        "## Corpus",
        "",
        f"- {CORPUS_2B_NAME}",
        f"- Source: `{CORPUS_2B_URL}`",
        "- Strip rule and raw-download hash: CORPUS_MANIFEST.md §1.",
        f"- Registered stripped-text sha256: `{CORPUS_2B_STRIPPED_SHA256}`",
        "  (the generator's corpus-hash gate accepts exactly this hash).",
        "",
        "## Per-run item-set parameters (variant / target_tokens / seed)",
        "",
    ]
    for combo in longctx_combos:
        lines.append(
            f"- variant=`{combo['variant']}`, target_tokens=`{combo['target_tokens']}`, "
            f"seed=`{combo['seed']}`, n_items=`{combo['n_items']}`"
        )
    lines += [
        "",
        "## Digest check",
        "",
        "Rebuild the item set with the generator above against the",
        "verified corpus text and the parameters listed, then recompute",
        "`longctx_verification_digest` (this script, over item id +",
        "`prompt_tokens` + `expected`, sorted by id) and compare against",
        "`longctx_recipe.verification_digest` in `dataset-manifest.json`.",
        "`scripts/package_dataset.py --verify-recipe` automates this",
        "(PREREG §11; digest mechanism demonstrated in CORPUS_MANIFEST.md §3).",
        "",
    ]
    (out_dir / "RECONSTRUCTION.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Post-write embargo scanner
# ---------------------------------------------------------------------------


def scan_for_embargoed_content(out_dir: Path, raw_longctx_items: list[dict]) -> None:
    """Re-read every produced byte and hard-fail if any longctx item's
    prompt text or token array appears anywhere in the output tree.

    The forbidden list is built ONLY from each item's `prompt` /
    `prompt_tokens` (the embargoed fields) — the FULL prompt string and the
    FULL token-id array (as JSON and as a comma-joined run), never
    sub-fragments of either. The reconstructed `key` / `depths` /
    `n_answer_tokens` metadata that does get published (PREREG §11 — a NATO
    word, a handful of floats, an int) is never added to this list, so it
    cannot trip the scanner: this function has no way to know those values
    even exist, let alone flag them.

    Ruling 4a (2026-08-27): `expected` (the gold match strings, e.g. a bare
    4-digit passcode) IS published on longctx records, and is deliberately
    excluded from `forbidden` too. A gold string is a short substring that
    also happens to appear *inside* the embargoed prompt text (the needle
    sentence literally contains it) — but the scanner only ever forbids the
    prompt's FULL text or the token array's FULL serialization, never a bare
    4-digit fragment of either, so publishing golds cannot make this scanner
    false-positive against itself. If a real prompt sentence (or the full
    token array) is ever accidentally written to the output tree, THAT full
    string is still what trips this scanner — a short gold digit run inside
    it is not what's being matched on."""
    forbidden: list[tuple[str, str, str]] = []
    for it in raw_longctx_items:
        prompt = it.get("prompt")
        if prompt:
            forbidden.append(("prompt text", it["id"], prompt))
        toks = it.get("prompt_tokens")
        if toks:
            forbidden.append(
                ("prompt_tokens (JSON array)", it["id"], json.dumps(list(toks)))
            )
            forbidden.append(
                (
                    "prompt_tokens (comma-joined)",
                    it["id"],
                    ",".join(str(t) for t in toks),
                )
            )
    if not forbidden:
        return

    violations = []
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, ValueError):
            content = path.read_bytes().decode("utf-8", errors="ignore")
        for kind, item_id, needle in forbidden:
            if needle and needle in content:
                violations.append(f"{path}: contains {kind} for item {item_id!r}")

    if violations:
        raise EmbargoViolation(
            "EMBARGO SCANNER FAILED — longctx_retrieval prompt/token content "
            "found in the published dataset tree (PREREG §11):\n"
            + "\n".join(violations)
        )


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------


def _check_longctx_corpus_provenance(run: RunData, run_dir: Path) -> None:
    """Positive provenance check (pre-0B ticket, freeze-plan §10; replaces
    the retired seed-signature heuristic of OPEN_QUESTIONS §4b — that
    heuristic became wrong the moment PREREG Amendment 1 §A registered 2b at
    the same n=96/seed=2024 as 2a). The run's manifest must positively
    record which corpus built its longctx items; only the registered 2b
    corpus (PG-1184) is publishable."""
    recorded = (
        ((run.run_config or {}).get("suites", {}) or {})
        .get(LONGCTX_SUITE, {})
        .get("corpus_sha256")
    )
    if not recorded:
        raise EmbargoViolation(
            f"refusing to package {run_dir}: run contains longctx_retrieval "
            f"items but its manifest records no corpus provenance "
            f"(manifest.json _run_config.suites.longctx_retrieval."
            f"corpus_sha256 is missing). Only runs positively recorded "
            f"against the registered 2b corpus "
            f"({CORPUS_2B_STRIPPED_SHA256}) are publishable (PREREG §3.1, "
            f"§11); regenerate the run with a pipeline that records "
            f"_run_config, or amend the manifest from the run's own "
            f"provenance records."
        )
    if recorded == CORPUS_2A_SHA256:
        raise EmbargoViolation(
            f"refusing to package {run_dir}: manifest records PREREG §3.1 "
            f"config 2a's corpus ({CORPUS_2A_SHA256}) — 2a's prompts 'are "
            f"never displayed on the site and never published; outputs and "
            f"statistics only' (PREREG §3.1)."
        )
    if recorded != CORPUS_2B_STRIPPED_SHA256:
        raise EmbargoViolation(
            f"refusing to package {run_dir}: manifest records an "
            f"unrecognized corpus sha256 ({recorded}); the only publishable "
            f"longctx corpus is the registered 2b corpus "
            f"({CORPUS_2B_STRIPPED_SHA256}) (PREREG §3.1, §11)."
        )


def package(
    run_dir: Path,
    out_dir: Path,
    tokenizer_path=None,
    corpus_path=None,
) -> dict:
    """`tokenizer_path` / `corpus_path` are required if and only if the run
    contains `longctx_retrieval` items (used to reconstruct the registered
    `key`/`depths`/`n_answer_tokens` metadata — PREREG §11); a run without
    that suite ignores both."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)

    run = load_run(run_dir)
    refuse_embargoed_run(run, run_dir)

    suites_present = sorted({it["suite"] for it in run.items})

    longctx_items = [it for it in run.items if it.get("suite") == LONGCTX_SUITE]
    longctx_combos: list[dict] = []
    if longctx_items:
        by_combo: dict[tuple, list[dict]] = {}
        for it in longctx_items:
            parsed = parse_longctx_id(it["id"])
            key = (
                (it.get("variant"), it.get("target_tokens"), it.get("seed"))
                if "variant" in it and "target_tokens" in it and "seed" in it
                else (
                    (parsed["variant"], parsed["target_tokens"], parsed["seed"])
                    if parsed
                    else (None, None, None)
                )
            )
            by_combo.setdefault(key, []).append(it)
        for (variant, target_tokens, seed), its in sorted(
            by_combo.items(), key=lambda kv: str(kv[0])
        ):
            longctx_combos.append(
                {
                    "variant": variant,
                    "target_tokens": target_tokens,
                    "seed": seed,
                    "n_items": len(its),
                }
            )
        # Checked BEFORE anything is written: refusing a 2a-shaped run must
        # never leave a partial/stale package behind.
        _check_longctx_corpus_provenance(run, run_dir)

        if tokenizer_path is None or corpus_path is None:
            raise ValueError(
                f"refusing to package {run_dir}: it contains "
                f"longctx_retrieval items, so --tokenizer-path and "
                f"--corpus-path are required to reconstruct the registered "
                f"key/depths/n_answer_tokens metadata (PREREG §11)"
            )
        # Also checked before out_dir is touched — a reconstruction mismatch
        # must never leave a partial/stale package behind either.
        longctx_reconstruction = _reconstruct_longctx_metadata(
            longctx_items, longctx_combos, tokenizer_path, corpus_path
        )
    else:
        longctx_reconstruction = None

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    suite_counts: dict[str, int] = {}
    exclusions: list[dict] = []

    for suite in suites_present:
        items_for_suite = [it for it in run.items if it["suite"] == suite]
        if suite == RETIRED_RETRIEVAL_SUITE:
            exclusions.append(
                {
                    "suite": suite,
                    "item_count": len(items_for_suite),
                    "reason": RETIRED_RETRIEVAL_REASON,
                }
            )
            continue
        records = build_suite_records(
            suite,
            items_for_suite,
            run.outputs,
            run.grades,
            run.manifest,
            longctx_reconstruction=(
                longctx_reconstruction if suite == LONGCTX_SUITE else None
            ),
        )
        write_jsonl(records, out_dir / f"{suite}.jsonl")
        suite_counts[suite] = len(records)

    longctx_recipe = None
    if longctx_items:
        digest = longctx_verification_digest(
            _longctx_entries_from_raw_items(run.items)
        )
        longctx_recipe = {
            "verification_digest": digest,
            "n_items": len(longctx_items),
            "combos": longctx_combos,
            "corpus_name": CORPUS_2B_NAME,
            "corpus_sha256": CORPUS_2B_STRIPPED_SHA256,
        }
        write_reconstruction_md(
            out_dir, longctx_combos, infer_tokenizer_id(run.manifest)
        )
        exclusions.append(
            {
                "field": "longctx_retrieval.prompt / longctx_retrieval.prompt_tokens",
                "item_count": len(longctx_items),
                "reason": (
                    "multivalue2 prompts (question text and the token-id "
                    "prompt array) are structurally excluded from "
                    "publication in every configuration (PREREG §11); see "
                    "RECONSTRUCTION.md"
                ),
            }
        )

    source_files = {}
    for name in ("manifest.json", "items.jsonl", "grades.jsonl"):
        p = run_dir / name
        if p.exists():
            source_files[name] = hashlib.sha256(p.read_bytes()).hexdigest()

    dataset_manifest = {
        "run_id": run_dir.name,
        "prereg_commit": PREREG_COMMIT,
        "source_files": source_files,
        "suite_record_counts": suite_counts,
        "exclusions": exclusions,
        "twins_check": (
            f"no {TWINS_SUITE!r} suite present in this run — verified before "
            "any file was written"
        ),
    }
    if longctx_recipe is not None:
        dataset_manifest["longctx_recipe"] = longctx_recipe

    (out_dir / "dataset-manifest.json").write_text(
        json.dumps(dataset_manifest, sort_keys=True, indent=2) + "\n"
    )

    # Generation date lives outside the deterministic content, per PREREG
    # §11 packaging instructions: re-packaging an unchanged run must diff
    # cleanly except for this one file.
    (out_dir / "PACKAGED_AT").write_text(
        datetime.datetime.now(datetime.timezone.utc).date().isoformat() + "\n"
    )

    scan_for_embargoed_content(out_dir, longctx_items)

    return {
        "run_id": run_dir.name,
        "suite_record_counts": suite_counts,
        "exclusions": exclusions,
        "out_dir": str(out_dir),
    }


# ---------------------------------------------------------------------------
# --verify-recipe
# ---------------------------------------------------------------------------

# CORPUS_MANIFEST.md §3's registered reference: real Qwen2.5-1.5B-Instruct
# tokenizer, real (verified) 2a corpus, mv2.build_items(tokenizer,
# "multivalue2", 20, 4096, 2024), mv2.items_digest(items).
CONFIG_2A_REFERENCE_DIGEST = (
    "9220589bd8607bd0ff3be5bdcfecd23df07cac82d354d992468b15b60f398972"
)


def _default_pipeline_root(run_dir: Path) -> Path:
    # runs/<run-id> -> pipeline root is two parents up.
    return run_dir.resolve().parent.parent


def _load_tokenizer_offline(tokenizer_dir: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)


def _verify_longctx_recipe(
    run_dir: Path,
    dataset_manifest: dict,
    pipeline_root: Path,
    tokenizer_path=None,
    corpus_path=None,
) -> tuple[str, str]:
    recipe = dataset_manifest.get("longctx_recipe")
    if not recipe:
        return "SKIPPED", "no longctx_recipe recorded in dataset-manifest.json"

    # Reuse whatever was given to `package()` for reconstruction if present
    # (the same reconstruction path); otherwise fall back to the same
    # locally-cached-only auto-detection the 2a-machinery check uses.
    if tokenizer_path is None:
        tokenizer_id = infer_tokenizer_id(load_run(run_dir).manifest) or ""
        tokenizer_path = pipeline_root / "models" / "hf" / tokenizer_id
    if corpus_path is None:
        corpus_path = pipeline_root / "corpora" / "pg1184-monte-cristo.txt"
    tokenizer_path = Path(tokenizer_path)
    corpus_path = Path(corpus_path)

    if not tokenizer_path.exists():
        return (
            "SKIPPED",
            f"tokenizer not cached locally at {tokenizer_path}; not downloading",
        )
    if not corpus_path.exists():
        return (
            "SKIPPED",
            f"2b corpus not cached locally at {corpus_path}; not downloading",
        )

    entries = []
    try:
        for combo in recipe["combos"]:
            # Same reconstruction path package() uses for key/depths/
            # n_answer_tokens: vendored generator + tokenizer + a corpus
            # verified against the registered sha256.
            raw_items = _rebuild_longctx_raw_items(
                tokenizer_path,
                corpus_path,
                recipe["corpus_sha256"],
                n_items=combo["n_items"],
                seed=combo["seed"],
                variant=combo["variant"],
                target_tokens=combo["target_tokens"],
            )
            for raw in raw_items:
                item_id = _longctx_wrapper_id(
                    combo["variant"], combo["target_tokens"], combo["seed"],
                    raw["index"],
                )
                entries.append((item_id, raw["gen_prompt_ids"], raw["match_strings"]))
    except ValueError as exc:
        return "FAIL", str(exc)
    except Exception as exc:  # pragma: no cover - environment dependent
        return "SKIPPED", f"could not load tokenizer offline: {exc}"

    digest = longctx_verification_digest(entries)
    if digest == recipe["verification_digest"]:
        return "PASS", "rebuilt item-set digest matches dataset-manifest.json"
    return (
        "FAIL",
        f"rebuilt digest {digest} != recorded {recipe['verification_digest']}",
    )


def _verify_2a_machinery(pipeline_root: Path) -> tuple[str, str]:
    import os

    tokenizer_path = pipeline_root / "models" / "hf" / "Qwen2.5-1.5B-Instruct"
    if not tokenizer_path.exists():
        return (
            "SKIPPED",
            f"tokenizer not cached locally at {tokenizer_path}; not downloading",
        )

    old_env = {
        k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    }
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        try:
            from datasets import load_dataset

            from bitcliff_pipeline.vendor import generate_multivalue2 as mv2

            ds = load_dataset(mv2.CORPUS_DATASET, split="train")
            corpus_text = "\n\n".join(ds["text"])
        except Exception as exc:
            return (
                "SKIPPED",
                f"2a corpus (sgoel9/paul_graham_essays) not cached locally; "
                f"not downloading ({exc})",
            )
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    actual_sha = hashlib.sha256(corpus_text.encode("utf-8")).hexdigest()
    if actual_sha != CORPUS_2A_SHA256:
        return (
            "FAIL",
            f"local 2a corpus sha256 {actual_sha} != registered "
            f"{CORPUS_2A_SHA256}",
        )

    try:
        tokenizer = _load_tokenizer_offline(tokenizer_path)
    except Exception as exc:  # pragma: no cover - environment dependent
        return "SKIPPED", f"could not load tokenizer offline: {exc}"

    from bitcliff_pipeline.vendor import generate_multivalue2 as mv2

    token_ids = tokenizer(corpus_text, add_special_tokens=False)["input_ids"]
    mv2._STREAM_CACHE[tokenizer.name_or_path] = token_ids
    items = mv2.build_items(tokenizer, "multivalue2", 20, 4096, 2024)
    digest = mv2.items_digest(items)
    if digest == CONFIG_2A_REFERENCE_DIGEST:
        return "PASS", "recipe machinery reproduces the registered 2a digest"
    return (
        "FAIL",
        f"recipe-machinery digest {digest} != registered "
        f"{CONFIG_2A_REFERENCE_DIGEST}",
    )


def verify_recipe(
    run_dir: Path, out_dir: Path, tokenizer_path=None, corpus_path=None
) -> tuple[str, str]:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    dataset_manifest = json.loads(
        (out_dir / "dataset-manifest.json").read_text()
    )
    pipeline_root = _default_pipeline_root(run_dir)

    if dataset_manifest.get("longctx_recipe"):
        return _verify_longctx_recipe(
            run_dir, dataset_manifest, pipeline_root, tokenizer_path, corpus_path
        )
    return _verify_2a_machinery(pipeline_root)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-recipe", action="store_true")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help=(
            "HF tokenizer directory (e.g. models/hf/Qwen2.5-1.5B-Instruct). "
            "REQUIRED if the run contains longctx_retrieval items: used to "
            "reconstruct the registered key/depths/n_answer_tokens metadata "
            "(PREREG §11) via the vendored generator, and (with "
            "--verify-recipe) to verify the reconstruction digest. Ignored, "
            "and not required, for a run with no longctx_retrieval items."
        ),
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=None,
        help=(
            "Local 2b corpus text file (Gutenberg PG #1184, stripped), "
            "verified against the registered sha256 (PREREG §3.1 / "
            "CORPUS_MANIFEST.md §1) before use. REQUIRED if the run "
            "contains longctx_retrieval items. Never downloaded — the run "
            "fails loudly instead of fetching anything."
        ),
    )
    args = parser.parse_args(argv)

    run_peek = load_run(args.run_dir)
    has_longctx = any(it.get("suite") == LONGCTX_SUITE for it in run_peek.items)
    if has_longctx and (args.tokenizer_path is None or args.corpus_path is None):
        parser.error(
            "run contains longctx_retrieval items: --tokenizer-path and "
            "--corpus-path are required to reconstruct the registered "
            "key/depths/n_answer_tokens metadata (PREREG §11)"
        )

    stats = package(
        args.run_dir,
        args.out,
        tokenizer_path=args.tokenizer_path,
        corpus_path=args.corpus_path,
    )
    print(f"packaged {stats['run_id']} -> {stats['out_dir']}")
    for suite, count in sorted(stats["suite_record_counts"].items()):
        print(f"  {suite}: {count} items")
    for excl in stats["exclusions"]:
        target = excl.get("suite") or excl.get("field")
        print(f"  excluded: {target} ({excl['reason']})")

    if args.verify_recipe:
        status, message = verify_recipe(
            args.run_dir,
            args.out,
            tokenizer_path=args.tokenizer_path,
            corpus_path=args.corpus_path,
        )
        print(f"verify-recipe: {status} — {message}")
        if status == "FAIL":
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
