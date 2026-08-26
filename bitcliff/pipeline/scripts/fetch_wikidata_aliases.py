#!/usr/bin/env python3
"""Mechanical Wikidata alias augmentation for the closed-book factual QA
suite (PREREG §3.4 branch (b), user ruling 2026-08-26).

Purpose: for the SAME 500 PopQA records the `factual_qa` suite samples
(seed 7411, `factual_qa.picked_records`), fetch each sampled record's
object-entity English Wikidata label + aliases, and write a committed
mapping file (`data/popqa_wikidata_aliases_seed7411.json`) that
`factual_qa.load_popqa_items(..., alias_augmentation_path=...)` consumes.

This is deliberately MECHANICAL and OUTPUT-BLIND: it is a single rule
("English label + English aliases of the object entity, fetched for every
sampled record") applied uniformly, run and committed WITHOUT reading any
model output. No per-item judgment calls are made here.

Record set reproduction: rather than re-implementing PopQA's stratified,
seeded sampling (risking drift from the registered rule), this script
imports and calls `bitcliff_pipeline.suites.factual_qa.picked_records`
directly — the exact function `items_from_records` itself uses — so the
500-record set is identical by construction to the one the suite draws for
run `factualqa-explore` / the confirmatory run.

Object entity QID: derived via `factual_qa.object_qid`, which reads the
record's `o_uri` field. NOTE: PopQA's `obj_id` column is an internal
numeric identifier, NOT the Wikidata QID — verified against
datasets-server (a record with obj_id=2834605 has o_uri encoding Q82955;
the numbers do not correspond) — see that function's docstring.

Network: stdlib `urllib` only, polite `User-Agent`, <=50 ids per
`wbgetentities` call (the Wikidata API's own per-request id cap), a brief
sleep between calls.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "bitcliff-pipeline-fetch-wikidata-aliases/1.0 (research use; see repo LICENSE_AUDIT.md)"
BATCH_SIZE = 50
SLEEP_SECONDS = 0.3

AUGMENTATION_RULE = (
    "English label + English aliases of the object entity, fetched for "
    "every sampled record, output-blind."
)

DEFAULT_N_ITEMS = 500
DEFAULT_SEED = 7411
DEFAULT_OUT = REPO_ROOT / "data" / "popqa_wikidata_aliases_seed7411.json"


def chunk(seq, size):
    """Split `seq` into consecutive lists of at most `size` elements."""
    seq = list(seq)
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def entities_url(qids: list[str]) -> str:
    params = {
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "props": "aliases|labels",
        "languages": "en",
        "format": "json",
    }
    return f"{WIKIDATA_API}?{urllib.parse.urlencode(params, safe='|')}"


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_entities_response(data: dict) -> dict[str, list[str]]:
    """Map each entity in a `wbgetentities` response to its sorted, unique
    (English label + English aliases) value list. Entities the API marks
    `missing` are skipped (never fabricated)."""
    out: dict[str, list[str]] = {}
    for qid, entity in data.get("entities", {}).items():
        if "missing" in entity:
            continue
        values: set[str] = set()
        label = entity.get("labels", {}).get("en", {}).get("value")
        if label:
            values.add(label)
        for alias in entity.get("aliases", {}).get("en", []):
            v = alias.get("value")
            if v:
                values.add(v)
        out[qid] = sorted(values)
    return out


def fetch_aliases(qids: list[str], sleep_s: float = SLEEP_SECONDS) -> tuple[dict[str, list[str]], int]:
    """Batch-fetch aliases for `qids`. Returns (mapping, n_missing)."""
    aliases: dict[str, list[str]] = {}
    n_missing = 0
    batches = chunk(qids, BATCH_SIZE)
    for i, batch in enumerate(batches):
        url = entities_url(batch)
        data = _get_json(url)
        batch_aliases = parse_entities_response(data)
        aliases.update(batch_aliases)
        n_missing += len(batch) - len(batch_aliases)
        print(f"  batch {i + 1}/{len(batches)}: {len(batch)} ids -> {len(batch_aliases)} resolved", file=sys.stderr)
        if i < len(batches) - 1:
            time.sleep(sleep_s)
    return aliases, n_missing


def build_mapping(n_items: int, seed: int) -> tuple[dict, dict]:
    """Reproduce the sampled record set and fetch its object-entity
    aliases. Returns (header, aliases)."""
    from datasets import load_dataset

    from bitcliff_pipeline.suites import factual_qa

    print(f"loading akariasai/PopQA test split...", file=sys.stderr)
    ds = load_dataset("akariasai/PopQA", split="test")

    print(f"re-deriving the n={n_items}, seed={seed} sampled record set via "
          f"factual_qa.picked_records...", file=sys.stderr)
    picked = factual_qa.picked_records(ds, n_items, seed)
    assert len(picked) == n_items, f"expected {n_items} picked records, got {len(picked)}"

    qids = sorted({factual_qa.object_qid(r) for r in picked})
    print(f"{len(picked)} sampled records -> {len(qids)} unique object QIDs", file=sys.stderr)

    aliases, n_missing = fetch_aliases(qids)
    n_with_aliases = len(aliases)
    n_aliases_total = sum(len(v) for v in aliases.values())
    n_aliases_mean = round(n_aliases_total / n_with_aliases, 3) if n_with_aliases else 0.0

    header = {
        "retrieval_date": datetime.date.today().isoformat(),
        "rule": AUGMENTATION_RULE,
        "endpoint": f"{WIKIDATA_API}?action=wbgetentities&ids=...&props=aliases|labels&languages=en&format=json",
        "batch_size": BATCH_SIZE,
        "n_items_sampled": n_items,
        "seed": seed,
        "n_qids": len(qids),
        "n_qids_resolved": n_with_aliases,
        "n_qids_missing": n_missing,
        "n_aliases_total": n_aliases_total,
        "n_aliases_mean_per_qid": n_aliases_mean,
    }
    return header, aliases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-items", type=int, default=DEFAULT_N_ITEMS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    header, aliases = build_mapping(args.n_items, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"_meta": header, "aliases": aliases}, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} ({header['n_qids_resolved']}/{header['n_qids']} QIDs resolved, "
          f"{header['n_aliases_total']} aliases total)", file=sys.stderr)


if __name__ == "__main__":
    main()
