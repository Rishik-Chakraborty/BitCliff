import argparse
import dataclasses
import json
from collections import defaultdict
from pathlib import Path

from . import generate as gen_mod
from . import registered
from .config import LadderConfig, load_config
from .divergence import divergence_for_records
from .grading import grade_record, read_grades
from .hashing import build_manifest, item_set_sha256, load_manifest, verify_manifest, write_manifest
from .items import EvalItem
from .models import ensure_quants, resolve_all
from .report import add_retention, aggregate, plot_retention, write_csv, write_json

# Pre-rerun hardening (OPEN_QUESTIONS §8): the registered n for every suite
# the boot-time item-set hash gate below covers. A suite with no entry here
# (e.g. "spectacle", which is unscored and has no registered item set) is
# never gated. `arithmetic_twins` has no config-supplied n_items (PREREG
# §3.3 fixes it at all 47 template pairs = 94 items) -- its registered n is
# still checked, just against the item count itself rather than a config
# key.
REGISTERED_SUITE_N: dict[str, int] = {
    "arithmetic": registered.ARITHMETIC_N,
    "arithmetic_twins": registered.TWINS_N,
    "factual_qa": registered.FACTUAL_QA_N,
    "longctx_retrieval": registered.LONGCTX_N,
}


def assert_item_sets_match_registered(items: list[EvalItem], model_id: str) -> None:
    """Boot-time item-set hash gate (pre-rerun hardening, OPEN_QUESTIONS
    §8): run once per invocation, immediately after items are built and
    BEFORE any generation -- the same abort-before-generation pattern as
    the tokenizer-match gate below, but for item-set identity rather than
    tokenizer agreement.

    Groups `items` by suite. A suite whose item count does not equal that
    suite's REGISTERED n (`REGISTERED_SUITE_N`) is left ungated -- a
    non-registered n (a smoke config's n=20, a unit test's n=4, ...) is by
    construction not a confirmatory draw, so there is nothing registered to
    check it against. For every suite AT its registered n, this recomputes
    the item-set hash (`hashing.item_set_sha256`) and compares it against
    `registered.expected_item_set_sha256(suite, model_id)`:

    - No pinned hash at all (neither registered nor derived) for that
      (suite, model_id): refuses outright -- unknown item sets are never
      silently accepted.
    - A pinned hash that does not match: refuses, naming suite/expected/
      actual -- this is the exact class of bug OPEN_QUESTIONS §8 documents
      (a hand-copied M3 weight vector sampling a non-registered factual_qa
      item set) made structurally impossible.
    - A pinned hash that matches: no-op.

    A config with `exploratory: true` skips calling this function entirely
    (see `run_pipeline` below), printing a loud warning instead.
    """
    by_suite: dict[str, list[EvalItem]] = defaultdict(list)
    for it in items:
        by_suite[it.suite].append(it)

    for suite, suite_items in by_suite.items():
        expected_n = REGISTERED_SUITE_N.get(suite)
        if expected_n is None or len(suite_items) != expected_n:
            continue  # not a registered-n draw for this suite -- ungated
        expected = registered.expected_item_set_sha256(suite, model_id)
        if expected is None:
            raise RuntimeError(
                f"item-set hash gate: no registered or derived item-set "
                f"hash pinned for suite={suite!r} model_id={model_id!r} at "
                f"n={expected_n} -- refusing to generate on an unknown item "
                f"set (unknown = refuse). Pin one in "
                f"bitcliff_pipeline.registered, or mark this config "
                f"`exploratory: true` if it is deliberately non-registered."
            )
        actual = item_set_sha256(suite_items)
        if actual != expected:
            raise RuntimeError(
                f"item-set hash gate: suite={suite!r} model_id={model_id!r} "
                f"-- expected item-set sha256 {expected}, got {actual}. "
                f"This config would generate on a NON-registered item set; "
                f"refusing before any generation (OPEN_QUESTIONS §8)."
            )


def _load_hf_tokenizer(path: Path):
    """Deferred `transformers` import (matching the pattern already used
    for other heavy/networked deps elsewhere in this pipeline, e.g.
    `calibrate_f16.py`'s `datasets` imports) -- importing `transformers` at
    module scope would drag it into every CLI invocation, including ones
    that never touch longctx_retrieval."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path))


def build_items(
    config: LadderConfig,
    base_dir: Path,
    longctx_tokenizer=None,
    longctx_answer_specs: list | None = None,
) -> list[EvalItem]:
    """`longctx_answer_specs`, when passed a list, is populated in place
    with the longctx_retrieval suite's `longctx_retrieval.AnswerSpec`
    sidecar (0B P3 carried item (a)), aligned 1:1 with this call's
    longctx_retrieval items in the same order they're appended to the
    returned list. `None` (the default, every pre-existing caller) skips
    this entirely and behaves exactly as before -- `build_items`'s return
    type and every other suite's behavior are unchanged."""
    from .suites import arithmetic, arithmetic_twins, factual_qa, longctx_retrieval, spectacle

    items: list[EvalItem] = []
    suites = config.suites
    if "arithmetic" in suites:
        s = suites["arithmetic"]
        items += arithmetic.load_gsm8k_items(s["n_items"], s["seed"])
    if "arithmetic_twins" in suites:
        # config block: `arithmetic_twins: {seed: 1301}` — n is fixed at all
        # 47 template pairs (94 items) by PREREG §3.3, so only the seed is
        # configurable.
        items += arithmetic_twins.load_pair_items(suites["arithmetic_twins"]["seed"])
    if "factual_qa" in suites:
        s = suites["factual_qa"]
        alias_augmentation_path = (
            base_dir / s["alias_augmentation_path"]
            if s.get("alias_augmentation_path")
            else None
        )
        # PREREG §7's popularity-mix knob (M1/M2/M3): an optional `weights`
        # key in the config block, forwarded straight through (0B P3 wiring
        # gap -- items_from_records already had the parameter; __main__
        # never passed it). YAML lists become a plain list; items_from_
        # records/_apportion_counts only ever index it, so a tuple isn't
        # required, but casting keeps the type stable regardless of the
        # YAML loader's list-vs-tuple behavior.
        weights = tuple(s["weights"]) if s.get("weights") else None
        items += factual_qa.load_popqa_items(
            s["n_items"], s["seed"],
            alias_augmentation_path=alias_augmentation_path, weights=weights,
        )
    if "longctx_retrieval" in suites:
        # PREREG §3.1 / RUN_0B.md §2 P1: needs a real tokenizer and a
        # verified corpus. Every registered value (variant, target_tokens,
        # seed, n_items, corpus_path, corpus_sha256, tokenizer_path) comes
        # from the config block, never hardcoded here. The corpus-hash gate
        # lives in `longctx_retrieval.build_items` itself (reused, not
        # duplicated): a corpus_path whose bytes don't hash to
        # corpus_sha256 raises AssertionError before any items are built.
        s = suites["longctx_retrieval"]
        tokenizer = (
            longctx_tokenizer
            if longctx_tokenizer is not None
            else _load_hf_tokenizer(base_dir / s["tokenizer_path"])
        )
        corpus_text = (base_dir / s["corpus_path"]).read_bytes().decode("utf-8")
        if longctx_answer_specs is not None:
            longctx_items, specs = longctx_retrieval.build_items_with_answer_spec(
                tokenizer,
                corpus_text,
                s["corpus_sha256"],
                n_items=s["n_items"],
                seed=s["seed"],
                variant=s["variant"],
                target_tokens=s["target_tokens"],
            )
            items += longctx_items
            longctx_answer_specs.extend(specs)
        else:
            items += longctx_retrieval.build_items(
                tokenizer,
                corpus_text,
                s["corpus_sha256"],
                n_items=s["n_items"],
                seed=s["seed"],
                variant=s["variant"],
                target_tokens=s["target_tokens"],
            )
    if "spectacle" in suites:
        items += spectacle.load_items(base_dir / suites["spectacle"]["path"])
    return items


def run_pipeline(
    config: LadderConfig,
    run_id: str,
    models_dir: Path,
    runs_dir: Path,
    stage: str,
    llm_factory=gen_mod.make_llm,
    base_dir: Path = Path("."),
) -> None:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"

    if stage in ("download", "all"):
        ensure_quants(config, models_dir)
        paths = resolve_all(config, models_dir)
        manifest = build_manifest(paths)
        meta = {
            q.label: {"uploader": q.uploader, "imatrix": q.imatrix, "spectacle_only": q.spectacle_only}
            for q in config.quants
        }
        meta["F16"] = {"uploader": "bitcliff-local-f16-conversion", "imatrix": False, "spectacle_only": False}
        for label, entry in manifest.items():
            entry.update(meta[label])
        # Pre-0B ticket (freeze-plan §10): record the run's config in the
        # manifest so dataset packaging can positively identify corpus
        # provenance (2a vs 2b) instead of the retired seed heuristic.
        manifest["_run_config"] = {
            "model_id": config.model_id,
            "suites": config.suites,
        }
        write_manifest(manifest, manifest_path)
        print(f"manifest written: {manifest_path}")

    if stage in ("generate", "all"):
        paths = resolve_all(config, models_dir)
        manifest = load_manifest(manifest_path)
        verify_manifest(manifest, paths)

        longctx_cfg = config.suites.get("longctx_retrieval")
        longctx_hf_tokenizer = (
            _load_hf_tokenizer(base_dir / longctx_cfg["tokenizer_path"])
            if longctx_cfg is not None
            else None
        )
        # 0B P3 carried item (a): capture the longctx_retrieval answer-span
        # sidecar (RUN_0B.md §2 P3(a) / P2 report's documented gap) as this
        # run's items are built, and persist it alongside items.jsonl so a
        # later P2 (nll_scorer) divergence pass can consume this run's
        # items directly -- pairing items.jsonl's prompt_tokens
        # (gen_prompt_ids) with answer_spans.jsonl's answer_ids by item id
        # -- without re-deriving anything from the tokenizer/corpus/vendored
        # generator a second time.
        longctx_answer_specs: list = []
        items = build_items(
            config, base_dir, longctx_tokenizer=longctx_hf_tokenizer,
            longctx_answer_specs=longctx_answer_specs,
        )

        # Pre-rerun hardening (OPEN_QUESTIONS §8): boot-time item-set hash
        # gate, BEFORE any generation for any file in this run.
        if config.exploratory:
            print(
                f"WARNING: config.exploratory=True (model_id={config.model_id!r}) "
                f"-- SKIPPING the registered item-set hash gate. This run's "
                f"item sets are NOT verified against any registered/derived "
                f"value; never treat this run's numbers as confirmatory "
                f"(OPEN_QUESTIONS §8)."
            )
        else:
            assert_item_sets_match_registered(items, config.model_id)

        items_path = run_dir / "items.jsonl"
        items_path.write_text(
            "".join(json.dumps(dataclasses.asdict(i)) + "\n" for i in items)
        )
        if longctx_answer_specs:
            answer_spans_path = run_dir / "answer_spans.jsonl"
            answer_spans_path.write_text(
                "".join(
                    json.dumps(dataclasses.asdict(spec)) + "\n"
                    for spec in longctx_answer_specs
                )
            )
        max_tokens_by_suite = {
            suite: scfg["max_tokens"]
            for suite, scfg in config.suites.items()
            if isinstance(scfg, dict) and "max_tokens" in scfg
        }

        # PREREG §3.1: the tokenizer-match gate must run before each
        # model's FIRST generation, over the registered 20-string sample
        # (the question strings of the run's first 20 longctx_retrieval
        # items, in id order). Reuses `longctx_retrieval.assert_tokenizer_
        # match` / `llama_tokenize_callable` -- the same mechanism
        # `calibrate_f16.py` already implements -- never a reimplemented
        # copy.
        #
        # PREREG §4: "a quant level is a file, not a label" -- a run config
        # can point several labels at distinct files from distinct
        # uploaders (the shootout matrix, P3), and there is no guarantee
        # every GGUF's bundled tokenizer agrees with the HF one just
        # because one sibling file already passed. So this gates EVERY
        # newly-loaded llama-cpp handle, keyed by that file's manifest
        # sha256 (file identity, not label identity -- two labels pointing
        # at the same bytes are checked once, correctly). Because of the
        # skip-if-output-exists short-circuit below, which file gets a
        # fresh llama-cpp handle "first" in a given invocation is
        # resume-dependent; per-file (not "first-overall") gating makes
        # that moot -- every file that actually gets loaded and generates
        # in THIS invocation is checked before its own first generation,
        # regardless of invocation order or prior partial runs.
        checked_tokenizer_shas: set[str] = set()
        if longctx_cfg is not None:
            from .suites import longctx_retrieval

            longctx_sample = [
                it.prompt
                for it in sorted(
                    (i for i in items if i.suite == "longctx_retrieval"),
                    key=lambda i: i.id,
                )[:20]
            ]

        for label, path in paths.items():
            out_path = run_dir / "outputs" / f"{label}.jsonl"
            if out_path.exists():
                print(f"skip {label}: {out_path} exists")
                continue
            print(f"generating {label} ({len(items)} items)...")
            llm = llm_factory(path, config.generation)
            gen_mod.assert_truncation_finish_reason(llm)
            if longctx_cfg is not None:
                file_sha256 = manifest[label]["sha256"]
                if file_sha256 not in checked_tokenizer_shas:
                    longctx_retrieval.assert_tokenizer_match(
                        longctx_hf_tokenizer,
                        longctx_retrieval.llama_tokenize_callable(llm),
                        longctx_sample,
                    )
                    checked_tokenizer_shas.add(file_sha256)
            records = gen_mod.run_items(
                llm, items, label, manifest[label]["sha256"], config.generation,
                max_tokens_by_suite,
            )
            gen_mod.write_records(records, out_path)

    if stage in ("grade", "all"):
        items = [
            EvalItem(
                **{
                    **d,
                    "expected": tuple(d["expected"]) if d["expected"] else None,
                    "prompt_tokens": (
                        tuple(d["prompt_tokens"]) if d.get("prompt_tokens") else None
                    ),
                }
            )
            for d in map(json.loads, (run_dir / "items.jsonl").read_text().splitlines())
        ]
        items_by_id = {i.id: i for i in items}
        baseline = {
            r.item_id: r for r in gen_mod.read_records(run_dir / "outputs" / "F16.jsonl")
        }
        graded_dicts = []
        for out_path in sorted((run_dir / "outputs").glob("*.jsonl")):
            for record in gen_mod.read_records(out_path):
                expected_prompt = items_by_id[record.item_id].prompt
                if record.prompt != expected_prompt:
                    raise RuntimeError(
                        f"{out_path.name}: prompt mismatch for {record.item_id} — "
                        "outputs are stale relative to items.jsonl; delete "
                        "runs/<run-id>/outputs/ and regenerate"
                    )
                g = grade_record(items_by_id, record)
                div = divergence_for_records(
                    baseline[record.item_id], record, tokenize=str.split
                )
                graded_dicts.append({**dataclasses.asdict(g), "divergence": div})
        with open(run_dir / "grades.jsonl", "w") as f:
            for d in graded_dicts:
                f.write(json.dumps(d) + "\n")
        print(f"graded {len(graded_dicts)} outputs")

    if stage in ("report", "all"):
        grades = read_grades(run_dir / "grades.jsonl")
        spectacle_labels = frozenset(q.label for q in config.quants if q.spectacle_only)
        rows = add_retention(aggregate(grades, spectacle_only_labels=spectacle_labels))
        write_csv(rows, run_dir / "results.csv")
        write_json(rows, run_dir / "results.json")
        ladder_order = ["F16"] + [q.label for q in config.quants]
        plot_retention(rows, ladder_order, run_dir / "retention.png")
        print(f"report written under {run_dir}")


def _resolve_base_dir(config_path: Path) -> Path:
    """The pipeline root: the directory containing the `configs/` ancestor
    of the config file, however deeply the config is nested (configs/x.yaml,
    configs/0b/x.yaml, ...). A config outside any `configs/` directory
    falls back to its own parent."""
    p = config_path.resolve()
    for ancestor in p.parents:
        if ancestor.name == "configs":
            return ancestor.parent
    return p.parent


def main() -> None:
    parser = argparse.ArgumentParser(prog="bitcliff_pipeline")
    parser.add_argument("config")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage", default="all",
        choices=["download", "generate", "grade", "report", "all"],
    )
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()
    config = load_config(args.config)
    run_pipeline(
        config,
        run_id=args.run_id,
        models_dir=Path(args.models_dir),
        runs_dir=Path(args.runs_dir),
        stage=args.stage,
        base_dir=_resolve_base_dir(Path(args.config)),
    )


if __name__ == "__main__":
    main()
