#!/usr/bin/env python3
"""Smoke-check site/fixtures/pilot-0a.json for the invariants the playground
UI (app.js) depends on, without needing a browser.

Checks:
  1. The fixtures JSON parses and has the expected top-level shape.
  2. The ladder is the full 11-rung canonical order, F16 first.
  3. Every item carries an output for every ladder rung (no holes).
  4. Excluded suites (the retired "retrieval" placeholder, and anything
     outside {arithmetic, spectacle}) are absent from items and rungs.
  5. Rungs flagged spectacle_only in the fixture are exactly the
     bitcliff-inhouse rungs, and carry that uploader consistently
     (the in-house / spectacle-only tagging the UI relies on).
  6. F16's divergence is always null (it's the reference; nothing
     diverges from itself), matching the "F16 card shows no highlight"
     rule.
  7. Every rung's sha256_short is exactly 12 hex characters.

Exits non-zero on any failure. Stdlib only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "pilot-0a.json"

EXPECTED_LADDER = [
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
REQUIRED_RUNG_FIELDS = {"text", "finish_reason", "state", "truncated", "loop", "divergence"}
ALLOWED_STATES = {"correct", "partial", "wrong", "unscored"}


def fail(msg: str, failures: list[str]) -> None:
    failures.append(msg)


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    if not FIXTURE_PATH.exists():
        print(f"FAIL: fixtures file not found: {FIXTURE_PATH}", file=sys.stderr)
        return 1

    try:
        with FIXTURE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"FAIL: fixtures JSON does not parse: {e}", file=sys.stderr)
        return 1
    print(f"OK: {FIXTURE_PATH} parses as JSON")

    for key in ("run_id", "disclaimer", "in_house_tag", "suites", "excluded_suites",
                "ladder", "rungs", "items", "generation_settings"):
        if key not in data:
            fail(f"missing top-level key: {key!r}", failures)
    if failures:
        print("FAIL:\n  - " + "\n  - ".join(failures), file=sys.stderr)
        return 1
    print("OK: top-level shape present")

    # 2. ladder order
    if data["ladder"] != EXPECTED_LADDER:
        fail(f"ladder order mismatch: got {data['ladder']!r}, expected {EXPECTED_LADDER!r}", failures)
    else:
        print(f"OK: ladder order is canonical (F16 -> IQ1_S, {len(EXPECTED_LADDER)} rungs)")

    # 4a. suites allowlist on items + excluded_suites bookkeeping
    if "retrieval" not in data["excluded_suites"]:
        fail("expected 'retrieval' in excluded_suites (retired placeholder suite must be recorded as excluded)", failures)
    bad_suite_items = [it["id"] for it in data["items"] if it["suite"] not in ALLOWED_SUITES]
    if bad_suite_items:
        fail(f"{len(bad_suite_items)} items have a suite outside {ALLOWED_SUITES}: {bad_suite_items[:5]}", failures)
    else:
        print(f"OK: all {len(data['items'])} items are in allowed suites {sorted(ALLOWED_SUITES)}; "
              f"'retrieval' excluded and recorded")

    # embargo: no multivalue2/longctx items should ever appear (pilot-0a has none by
    # construction, but check defensively against id/suite substrings anyway)
    embargoed_hits = [
        it["id"] for it in data["items"]
        if "multivalue2" in it["id"].lower() or "longctx" in it["id"].lower()
        or "multivalue2" in it["suite"].lower() or "longctx" in it["suite"].lower()
    ]
    if embargoed_hits:
        fail(f"embargoed multivalue2/longctx items leaked into fixtures: {embargoed_hits}", failures)
    else:
        print("OK: no multivalue2/longctx items present (embargo holds)")

    # 3. every item has every ladder rung, with required fields
    missing_rung_report = []
    bad_state_report = []
    for it in data["items"]:
        rung_keys = set(it.get("rungs", {}).keys())
        if rung_keys != set(data["ladder"]):
            missing_rung_report.append((it["id"], sorted(set(data["ladder"]) - rung_keys)))
            continue
        for label in data["ladder"]:
            rd = it["rungs"][label]
            missing_fields = REQUIRED_RUNG_FIELDS - set(rd.keys())
            if missing_fields:
                missing_rung_report.append((it["id"], f"{label} missing fields {missing_fields}"))
            if rd.get("state") not in ALLOWED_STATES:
                bad_state_report.append((it["id"], label, rd.get("state")))
        # F16 divergence must always be null
        f16 = it["rungs"].get("F16", {})
        if f16.get("divergence") is not None:
            fail(f"item {it['id']}: F16 divergence is not null ({f16.get('divergence')!r}) "
                 f"-- F16 must never show a divergence highlight", failures)

    if missing_rung_report:
        fail(f"{len(missing_rung_report)} items have missing/malformed rungs, e.g. {missing_rung_report[:5]}", failures)
    else:
        print(f"OK: every item has all {len(data['ladder'])} rungs with required fields")

    if bad_state_report:
        fail(f"{len(bad_state_report)} (item, rung) pairs have an unrecognized grade state: {bad_state_report[:5]}", failures)
    else:
        print(f"OK: every rung grade state is one of {sorted(ALLOWED_STATES)}")

    # 5. spectacle_only <-> bitcliff-inhouse consistency, and the tag string is present
    if not data["in_house_tag"]:
        fail("in_house_tag is empty", failures)
    inhouse_rungs = sorted(l for l, m in data["rungs"].items() if m.get("spectacle_only"))
    mismatched = [
        label for label, meta in data["rungs"].items()
        if bool(meta.get("spectacle_only")) != (meta.get("uploader") == "bitcliff-inhouse")
    ]
    if mismatched:
        fail(f"spectacle_only flag and uploader=='bitcliff-inhouse' disagree for: {mismatched}", failures)
    else:
        print(f"OK: spectacle_only rungs ({inhouse_rungs}) are exactly the bitcliff-inhouse rungs "
              f"-- in-house tag applies consistently")

    # 6. sha256_short length
    bad_hash = [label for label, meta in data["rungs"].items() if len(meta.get("sha256_short", "")) != 12]
    if bad_hash:
        fail(f"rungs with sha256_short not exactly 12 chars: {bad_hash}", failures)
    else:
        print("OK: every rung's sha256_short is 12 characters")

    n_arith = sum(1 for it in data["items"] if it["suite"] == "arithmetic")
    n_spec = sum(1 for it in data["items"] if it["suite"] == "spectacle")
    print(f"\nSummary: {len(data['items'])} items ({n_arith} arithmetic, {n_spec} spectacle) "
          f"x {len(data['ladder'])} rungs = {len(data['items']) * len(data['ladder'])} rung-outputs")
    print(f"Rungs: {', '.join(data['ladder'])}")
    print(f"In-house/spectacle-only rungs: {', '.join(inhouse_rungs)}")

    if failures:
        print("\nFAIL:\n  - " + "\n  - ".join(failures), file=sys.stderr)
        return 1

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
