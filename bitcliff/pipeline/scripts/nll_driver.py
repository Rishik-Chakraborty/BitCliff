#!/usr/bin/env python3
"""Teacher-forced answer-span NLL driver (PREREG §3.1 Q2; replaces the 0B
box driver that was never committed -- OPEN_QUESTIONS §16).

For one run id, scores every load (F16 via `config.f16_path`, then every
quant in config order, or a `--labels` subset) with
`nll_scorer.score_records` over the items of the named suites, writing
`<runs-dir>/<run-id>/nll/<label>.jsonl` (`nll_scorer.write_records`; the
schema of the committed `runs-cloud/pipeline/runs/0b-*/nll/*.jsonl`
records, exactly `nll_scorer.NLLRecord`) and a per-run
`<runs-dir>/<run-id>/nll/manifest.json` recording the driver settings.

A script, not a `bitcliff_pipeline` CLI stage: the generate CLI's shape is
generation-specific (ruling in the ledger). Nothing is reimplemented:

- items come from `__main__.build_items(..., longctx_answer_specs=...)`
  (the builder generation used), on a config copy trimmed to the named
  suites only, and pass the registered item-set gate
  (`__main__.assert_item_sets_match_registered`) unless the config is
  `exploratory: true`;
- each longctx `EvalItem` is paired index-for-index with its `AnswerSpec`
  sidecar (ids asserted equal) into an `nll_scorer.NLLItem` -- only the
  longctx suites (`longctx_retrieval` = Configuration 2b,
  `longctx_retrieval_2a` = 2a) carry an answer-span spec, so naming any
  other suite is refused;
- `digit_token_ids` is `nll_scorer.digit_token_ids_from_decode` over the
  HF tokenizer at the suites' shared `tokenizer_path` (the one the items
  are built with), `vocab_size = len(tokenizer)`; an empty set raises
  (never None -- the digits-only sensitivity must be populated).

Settings: RECOVERED vs UNRECOVERABLE (evidence: runs-cloud/0b-nll.log and
runs-cloud/0b-full.log; the NLLCFG_* lines in 0b-full.log carry run id,
timestamps and exit codes only, no settings):

- RECOVERED `n_ctx = ceil(max_seq / 256) * 256`, max_seq = the longest
  `len(gen_prompt_ids) + len(answer_ids)` over the items scored. Evidence:
  "n_ctx sized to 8448 (max seq 8263)" (0b-nll.log, the Qwen-7B passes
  after the OOM fix, commit 7364672 at 2026-09-03 19:21 UTC);
  8448 = ceil(8263/256)*256. Logged in the same words.
- RECOVERED `logits_all=True` -- required by `make_llama_logits_provider`
  (post-fix path: reads `llm.scores`, commit 7364672).
- RECOVERED (by construction) `seed = config.generation.seed` and
  `n_gpu_layers=-1`, exactly as `generate.make_llm`.
- RECOVERED log shape: "nll items: <n>", "digit token ids: <count>"
  (10 on every 0B pass), "NLL_START <label> <filename>",
  "NLL_DONE <label> n=<n>", "NLL_PASS_COMPLETE". (The original printed the
  digit count after the F16 NLL_START line; this driver derives it before
  any load, so it prints first -- log order only, no effect on records.)
- UNRECOVERABLE: the n_ctx of the pre-fix passes (0b-llama-8b-ladder,
  0b-shootout-arm1 -- no n_ctx line in the log), and every llama-cpp
  constructor argument the log does not show (n_batch / n_ubatch,
  flash_attn, n_threads, type_k/type_v, rope overrides, ...). This driver
  passes none of them, i.e. uses llama-cpp-python 0.3.35's defaults; the
  box reproduction gate (byte-compare against the committed records) is
  what establishes equivalence -- nothing here asserts it.

Determinism: identical inputs give byte-identical `<label>.jsonl` and
`manifest.json` (no timestamps anywhere; keys sorted). Resume: a label
whose `<label>.jsonl` already exists is skipped (generate-stage pattern);
the manifest is merged, and a resumed invocation whose settings differ
from the existing manifest's is refused.

Usage (from bitcliff/pipeline; the box reproduction gate for Qwen-7B
Q4_K_M, Configuration 2b)::

    uv run python scripts/nll_driver.py configs/0b/0b-qwen-7b-ladder.yaml \\
        --run-id 0b-qwen-7b-ladder-nllgate --runs-dir runs --models-dir models \\
        --suites longctx_retrieval --labels Q4_K_M
    cmp runs/0b-qwen-7b-ladder-nllgate/nll/Q4_K_M.jsonl \\
        runs-cloud/pipeline/runs/0b-qwen-7b-ladder/nll/Q4_K_M.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import bitcliff_pipeline.__main__ as main_mod
from bitcliff_pipeline import nll_scorer
from bitcliff_pipeline.config import LadderConfig, load_config
from bitcliff_pipeline.hashing import item_set_sha256, load_manifest, sha256_file
from bitcliff_pipeline.suites import longctx_retrieval

N_CTX_ROUND = 256
N_CTX_RULE = f"ceil(max_seq/{N_CTX_ROUND})*{N_CTX_ROUND}"
N_GPU_LAYERS = -1
LOGITS_ALL = True
UNRECOVERED_LLAMA_ARGS = "llama-cpp-python defaults (n_batch, n_ubatch, flash_attn, n_threads, type_k/type_v, ...): unrecoverable from the 0B log"


def make_nll_llm(model_path: Path, n_ctx: int, seed: int):
    """Default factory: `llama_cpp.Llama` with the recovered settings (see
    module docstring). Every other constructor argument is left at the
    llama-cpp-python 0.3.35 default (unrecoverable)."""
    from llama_cpp import Llama

    return Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        seed=seed,
        n_gpu_layers=N_GPU_LAYERS,
        logits_all=LOGITS_ALL,
        verbose=False,
    )


def n_ctx_for(items: list[nll_scorer.NLLItem]) -> tuple[int, int]:
    """(n_ctx, max_seq): the recovered rule, over the items scored."""
    max_seq = max(len(i.gen_prompt_ids) + len(i.answer_ids) for i in items)
    return math.ceil(max_seq / N_CTX_ROUND) * N_CTX_ROUND, max_seq


def resolve_suites(config: LadderConfig, suites: list[str] | None) -> list[str]:
    if suites is None:
        if config.nll is None:
            raise ValueError(
                "config has no `nll:` block and no --suites given -- name the "
                "suites to score (the 0B configs have no nll block; the box "
                "reproduction gate passes --suites longctx_retrieval)"
            )
        suites = list(config.nll.suites)
    if not suites or len(set(suites)) != len(suites):
        raise ValueError(f"suites must be a non-empty list without duplicates: {suites}")
    missing = [s for s in suites if s not in config.suites]
    if missing:
        raise ValueError(f"suite(s) {missing} not in the config's `suites` block")
    no_spec = [s for s in suites if s not in longctx_retrieval.SUITE_KEYS]
    if no_spec:
        raise ValueError(
            f"suite(s) {no_spec} have no answer-span spec (AnswerSpec exists "
            f"only for {list(longctx_retrieval.SUITE_KEYS)}) -- cannot teacher-force"
        )
    return suites


def resolve_labels(config: LadderConfig, models_dir: Path, labels: list[str] | None) -> dict[str, Path]:
    """label -> model path, F16 first then config order (as
    `models.resolve_all`), restricted to `labels`; only the selected files
    must exist."""
    paths = {"F16": config.f16_path, **{q.label: models_dir / q.filename for q in config.quants}}
    if labels is not None:
        unknown = [lab for lab in labels if lab not in paths]
        if unknown:
            raise ValueError(f"unknown label(s) {unknown}; config has {list(paths)}")
        paths = {lab: p for lab, p in paths.items() if lab in labels}
    missing = [lab for lab, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing model files for: {', '.join(missing)}")
    return paths


def build_nll_items(config: LadderConfig, suites: list[str], base_dir: Path):
    """(eval_items, nll_items, tokenizer) for the named suites only."""
    trimmed = dataclasses.replace(config, suites={k: config.suites[k] for k in suites})
    tok_path = main_mod._longctx_tokenizer_path(trimmed)
    tokenizer = main_mod._load_hf_tokenizer(base_dir / tok_path)
    specs: list = []
    items = main_mod.build_items(
        trimmed, base_dir, longctx_tokenizer=tokenizer, longctx_answer_specs=specs
    )
    longctx_items = [i for i in items if i.suite in longctx_retrieval.SUITE_KEYS]
    if len(longctx_items) != len(specs) or len(items) != len(longctx_items):
        raise RuntimeError(
            f"answer-span sidecar misaligned: {len(items)} items, "
            f"{len(longctx_items)} longctx items, {len(specs)} specs"
        )
    nll_items = []
    for item, spec in zip(longctx_items, specs):
        if item.id != spec.item_id:
            raise RuntimeError(f"answer-span sidecar misaligned: {item.id} != {spec.item_id}")
        nll_items.append(
            nll_scorer.NLLItem(
                id=item.id, suite=item.suite,
                gen_prompt_ids=tuple(item.prompt_tokens), answer_ids=tuple(spec.answer_ids),
            )
        )
    return items, nll_items, tokenizer


def _model_sha256(label: str, path: Path, run_manifest: dict | None) -> tuple[str, str]:
    actual = sha256_file(path)
    if run_manifest is not None and label in run_manifest:
        recorded = run_manifest[label]["sha256"]
        if recorded != actual:
            raise RuntimeError(
                f"{label}: run manifest.json records sha256 {recorded}, but "
                f"{path} hashes to {actual} -- refusing to score a different file"
            )
        return actual, "run manifest.json (verified)"
    return actual, "hashed"


def run_nll(
    config: LadderConfig,
    *,
    run_id: str,
    runs_dir: Path,
    models_dir: Path,
    base_dir: Path,
    suites: list[str] | None = None,
    labels: list[str] | None = None,
    llm_factory=make_nll_llm,
    config_sha256: str | None = None,
) -> None:
    suites = resolve_suites(config, suites)
    paths = resolve_labels(config, models_dir, labels)
    run_dir = runs_dir / run_id
    out_dir = run_dir / "nll"

    items, nll_items, tokenizer = build_nll_items(config, suites, base_dir)
    if config.exploratory:
        gate = "skipped (exploratory config)"
        print(
            f"WARNING: config.exploratory=True (model_id={config.model_id!r}) "
            f"-- SKIPPING the registered item-set hash gate; never treat these "
            f"NLL records as confirmatory (OPEN_QUESTIONS §8)."
        )
    else:
        main_mod.assert_item_sets_match_registered(items, config.model_id)
        gate = "passed"
    print(f"nll items: {len(nll_items)}")

    n_ctx, max_seq = n_ctx_for(nll_items)
    print(f"n_ctx sized to {n_ctx} (max seq {max_seq})")

    digit_ids = nll_scorer.digit_token_ids_from_decode(tokenizer.decode, len(tokenizer))
    if not digit_ids:
        raise RuntimeError(
            "digit token ids: derived set is EMPTY for this tokenizer -- the "
            "registered digits-only sensitivity would be all-null; refusing"
        )
    print(f"digit token ids: {len(digit_ids)}")

    seed = config.generation.seed
    settings = {
        "model_id": config.model_id,
        "config_sha256": config_sha256,
        "suites": list(suites),
        "n_items": {s: sum(1 for i in nll_items if i.suite == s) for s in suites},
        "item_set_sha256": {
            s: item_set_sha256([i for i in items if i.suite == s]) for s in suites
        },
        "item_set_gate": gate,
        "n_ctx": {"rule": N_CTX_RULE, "max_seq": max_seq, "value": n_ctx},
        "logits_all": LOGITS_ALL,
        "seed": seed,
        "n_gpu_layers": N_GPU_LAYERS,
        "other_llama_args": UNRECOVERED_LLAMA_ARGS,
        "digit_token_ids": {"count": len(digit_ids), "ids": sorted(digit_ids)},
        "machine": nll_scorer._machine_fingerprint(),
        "driver_sha256": sha256_file(Path(__file__)),
    }

    manifest_path = out_dir / "manifest.json"
    label_entries: dict = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        prior = {k: v for k, v in existing.items() if k != "labels"}
        if prior != settings:
            drift = sorted(k for k in set(prior) | set(settings) if prior.get(k) != settings.get(k))
            raise RuntimeError(
                f"{manifest_path}: existing NLL manifest settings differ from "
                f"this invocation's in {drift} -- refusing to resume into a "
                f"run scored under different settings"
            )
        label_entries = dict(existing.get("labels", {}))

    run_manifest_path = run_dir / "manifest.json"
    run_manifest = load_manifest(run_manifest_path) if run_manifest_path.exists() else None

    def write_manifest() -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({**settings, "labels": label_entries}, indent=2, sort_keys=True) + "\n"
        )

    for label, path in paths.items():
        out_path = out_dir / f"{label}.jsonl"
        if out_path.exists():
            print(f"skip {label}: {out_path} exists")
            continue
        model_sha, source = _model_sha256(label, path, run_manifest)
        print(f"NLL_START {label} {path.name}")
        llm = llm_factory(path, n_ctx, seed)
        try:
            records = nll_scorer.score_records(llm, nll_items, label, model_sha, digit_ids)
        finally:
            close = getattr(llm, "close", None)
            if callable(close):
                close()
            del llm
        nll_scorer.write_records(records, out_path)
        label_entries[label] = {
            "filename": path.name, "model_sha256": model_sha, "sha256_source": source,
        }
        write_manifest()
        print(f"NLL_DONE {label} n={len(records)}")
    write_manifest()
    print("NLL_PASS_COMPLETE")


def _csv(s: str) -> list[str]:
    return [x for x in (p.strip() for p in s.split(",")) if x]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Teacher-forced answer-span NLL driver (PREREG §3.1 Q2)."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--suites", type=_csv, default=None,
        help="comma-separated suite keys (default: the config's `nll.suites`)",
    )
    parser.add_argument(
        "--labels", type=_csv, default=None,
        help="comma-separated load labels to score (default: F16 + every quant)",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    run_nll(
        config,
        run_id=args.run_id,
        runs_dir=args.runs_dir,
        models_dir=args.models_dir,
        base_dir=main_mod._resolve_base_dir(args.config),
        suites=args.suites,
        labels=args.labels,
        config_sha256=sha256_file(args.config),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
