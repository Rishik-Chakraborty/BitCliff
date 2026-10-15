#!/usr/bin/env python3
# Ran on the 0B-prime box (i-04147e175eba11337) on 2026-10-14; committed for the
# record (OPEN_QUESTIONS §18). Body unchanged from the copy shipped to the box,
# sha256 6a87f2ddb1fdf6b31b49bfce68a562215730e4db69f1fa71849f5019f79dfa3b;
# byte-identical to s3://bitcliff-artifacts-048568674517/0b-prime/attempt-20261004/logs/bench_throughput.py.
"""RUN_0B_PRIME.md §6 throughput benchmark (product data, not science).

Enters no registered statistic. For one 0B ladder config: F16 + every quant,
on a fixed prompt set = the first 25 longctx_retrieval items (the §4 gate
items) + the first 25 arithmetic items, both read from the committed 0B run's
items.jsonl (identical ids across files). Greedy settings and n_ctx from the
config's generation block; per-suite budgets as the 0B run used them
(longctx 32, arithmetic = global). Generation is generate.make_llm +
generate.run_items (the pipeline's own code); timing comes from run_items'
per-item gen_wall_seconds / gen_tokens fields.

Writes one CSV row per file. speed_tok_s = arithmetic gen tokens / wall
(decode-dominated; the longctx rows are prefill-dominated at ~8k prompt
tokens and are reported separately as longctx_tok_s and longctx_wall_s).

usage: bench_throughput.py CONFIG RUN_DIR --models-dir models --out X.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from bitcliff_pipeline import generate as gen_mod
from bitcliff_pipeline.config import load_config
from bitcliff_pipeline.hashing import sha256_file
from bitcliff_pipeline.items import EvalItem


def first_items(run_dir: Path, suite: str, n: int) -> list[EvalItem]:
    out = []
    for line in (run_dir / "items.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d["suite"] != suite:
            continue
        out.append(EvalItem(
            id=d["id"], suite=d["suite"], prompt=d["prompt"],
            expected=tuple(d["expected"]) if d.get("expected") else None,
            prompt_tokens=tuple(d["prompt_tokens"]) if d.get("prompt_tokens") else None,
        ))
    out.sort(key=lambda it: it.id)
    return out[:n]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    cfg = load_config(a.config)
    gen = cfg.generation
    budgets = {"longctx_retrieval": 32}
    items = (first_items(a.run_dir, "longctx_retrieval", 25)
             + first_items(a.run_dir, "arithmetic", 25))
    assert len(items) == 50, len(items)

    files = [("F16", Path(cfg.f16_path))] + [
        (q.label, a.models_dir / q.filename) for q in cfg.quants]
    rows = []
    for label, path in files:
        sha = sha256_file(path)
        t0 = time.perf_counter()
        llm = gen_mod.make_llm(path, gen)
        load_s = time.perf_counter() - t0
        recs = gen_mod.run_items(llm, items, label, sha, gen,
                                 max_tokens_by_suite=budgets)
        del llm
        by = {}
        for r in recs:
            s = by.setdefault(r.suite, [0, 0.0])
            s[0] += r.gen_tokens or 0
            s[1] += r.gen_wall_seconds or 0.0
        ar, lc = by["arithmetic"], by["longctx_retrieval"]
        row = {
            "model_id": cfg.model_id, "label": label, "filename": path.name,
            "sha256": sha, "size_bytes": path.stat().st_size,
            "load_s": round(load_s, 3),
            "speed_tok_s": round(ar[0] / ar[1], 2) if ar[1] else "",
            "arith_gen_tokens": ar[0], "arith_wall_s": round(ar[1], 3),
            "longctx_tok_s": round(lc[0] / lc[1], 2) if lc[1] else "",
            "longctx_gen_tokens": lc[0], "longctx_wall_s": round(lc[1], 3),
            "n_ctx": gen.n_ctx, "machine": recs[0].machine,
        }
        rows.append(row)
        print("BENCH", json.dumps(row), flush=True)
        with a.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
