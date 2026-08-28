#!/usr/bin/env python3
"""F16 difficulty-calibration harness, PREREG §7 (the registered rule).

Per model, finds the hardest scored-suite configuration whose F16 accuracy
lies in [0.6, 0.85] -- measured F16-only, before any confirmatory quant
runs -- using ONLY the registered knobs and total orders:

- longctx_retrieval (§3.1/§7): knobs are `variant` (a fixed total order,
  easiest -> hardest, reused verbatim from
  `bitcliff_pipeline.vendor.generate_multivalue2.DIFFICULTY_ORDER`) and
  `target_tokens` (4096 tried first, then 8192; ties broken toward the
  larger value). Corpus is "The Count of Monte Cristo" (PG #1184), verified
  against the registered sha256. depths cycle (0.1, 0.5, 0.9) is fixed by
  the vendored generator's default.

  **PROVISIONAL n/seed**: PREREG registers no item count / sampling seed for
  the 2b (site/dataset) longctx runs -- only for the 2a paper-comparability
  config. This harness measures at n=96, seed=2024 (mirroring 2a) as a
  provisional stand-in, per the overnight work order and
  `OPEN_QUESTIONS.md` §2. Every longctx artifact this script writes carries
  the literal string in `PROVISIONAL_LONGCTX_NOTE` below.

- factual_qa (§3.4/§7): knob is the popularity-decile mix, chosen from the
  registered candidate set M1 (uniform) / M2 (linear-tail-heavy) / M3
  (step-tail-heavy), most tail-heavy first. n=500 and seed=2718 are
  REGISTERED and fixed (not a calibration knob) -- this harness never
  varies them.

  **Weight-vector direction note (read before touching M2_WEIGHTS /
  M3_WEIGHTS below).** PREREG §7 writes the M2/M3 weight vectors in
  "decile 1 = most popular .. decile 10 = least popular" order. But
  `factual_qa.items_from_records`'s `weights[i]` applies to `deciles[i]` as
  that function builds them: sorted by `s_pop` ASCENDING, so
  `deciles[0]` is the LEAST popular decile and `deciles[9]` is the MOST
  popular one (see that module's docstring, and
  `GRADER_CHARACTERIZATION.md`: "the popularity-mix knob (skewing the
  decile draw toward higher-popularity deciles) is the documented lever").
  So the vectors below are PREREG's listed vectors REVERSED -- written
  directly in the code's ascending-popularity index order, increasing
  toward the popular end -- not copied verbatim in PREREG's decile-1-first
  order. (The `M2_WEIGHTS` name in `tests/test_factual_qa.py` is an
  unrelated synthetic fixture for apportionment-math testing only; it is
  not evidence for this direction question and this module does not import
  it.)

Reuses tested suite code for every piece of grading/apportionment/item
construction logic (`longctx_retrieval.build_items/grade`,
`longctx_retrieval.assert_tokenizer_match`, `factual_qa.items_from_records`
/ `grade`, `generate.run_items`) -- this module adds only the calibration
SEARCH policy, the alias-mapping-for-seed-2718 bootstrap, and I/O
(measurements.jsonl / summary.json), per the overnight work order (task 1a).

CLI:
    uv run python scripts/calibrate_f16.py \\
        --model-id qwen2.5-1.5b-instruct \\
        --f16 models/f16/Qwen2.5-1.5B-Instruct-f16.gguf \\
        --hf-tokenizer models/hf/Qwen2.5-1.5B-Instruct \\
        --suite both \\
        --out calibration/qwen2.5-1.5b-instruct/

Resumable: measurements already present in `<out>/measurements.jsonl`
(matched by suite + the knob values + n_items + seed) are skipped, so a
run interrupted mid-suite can simply be re-invoked.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bitcliff_pipeline import generate as gen_mod  # noqa: E402
from bitcliff_pipeline.config import GenSettings  # noqa: E402
from bitcliff_pipeline.hashing import sha256_file  # noqa: E402
from bitcliff_pipeline.suites import factual_qa, longctx_retrieval  # noqa: E402
from bitcliff_pipeline.vendor import generate_multivalue2 as mv2  # noqa: E402

# ---------------------------------------------------------------------------
# Registered constants
# ---------------------------------------------------------------------------

BAND_LOW = 0.6
BAND_HIGH = 0.85
MONOTONICITY_NOISE = 0.05

CORPUS_PATH = REPO_ROOT / "corpora" / "pg1184-monte-cristo.txt"
CORPUS_SHA256 = "0a21a13834b5215876bd4019af8fbc436abbfbb61b2826db62223eb990071443"

LONGCTX_LADDER: tuple[str, ...] = mv2.DIFFICULTY_ORDER
LONGCTX_N_ITEMS = 96
LONGCTX_SEED = 2024
LONGCTX_ANSWER_BUDGET = 32
LONGCTX_TARGET_TOKENS = (4096, 8192)
LONGCTX_N_CTX_HEADROOM = 1024  # n_ctx = target_tokens + this
PROVISIONAL_LONGCTX_NOTE = (
    "provisional (pending 2b n/seed registration — OPEN_QUESTIONS §2)"
)

FACTUAL_QA_N_ITEMS = 500
FACTUAL_QA_SEED = 2718
FACTUAL_QA_CHARACTERIZATION_SEED = 7411  # NOT used for item construction here
FACTUAL_QA_ANSWER_BUDGET = 64
FACTUAL_QA_N_CTX = 2048

# See the module docstring's "Weight-vector direction note": these are
# PREREG §7's listed vectors, reversed into the ascending-`s_pop` decile
# index order that `factual_qa.items_from_records`'s `weights[i]` actually
# applies to (index 0 = least-popular decile, index 9 = most-popular).
M1_WEIGHTS: tuple[float, ...] = (0.1,) * 10
M2_WEIGHTS: tuple[float, ...] = tuple((i + 1) / 55 for i in range(10))
M3_WEIGHTS: tuple[float, ...] = (0.04,) * 5 + (0.16,) * 5
FACTUAL_QA_MIXES: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("M1", M1_WEIGHTS),
    ("M2", M2_WEIGHTS),
    ("M3", M3_WEIGHTS),
)

FETCH_ALIASES_SCRIPT = REPO_ROOT / "scripts" / "fetch_wikidata_aliases.py"
ALIAS_MAPPING_PATH_SEED2718 = REPO_ROOT / "data" / "popqa_wikidata_aliases_seed2718.json"

GEN_SEED = 42  # generation seed (distinct from every item-construction seed)


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without a model or network)
# ---------------------------------------------------------------------------


def in_band(acc: float, low: float = BAND_LOW, high: float = BAND_HIGH) -> bool:
    return low <= acc <= high


def item_set_sha256(items) -> str:
    """Deterministic hash over an item set: sorted item ids + each item's
    token content (prompt_tokens if the item was built in token space,
    else its text prompt) + its gold answer(s). Order-independent (sorted
    by id first) so the same item set hashes identically regardless of
    build/iteration order.
    """
    import hashlib

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


def calibrate_ladder(
    ladder: tuple[str, ...],
    measure: Callable[[str], float],
    low: float = BAND_LOW,
    high: float = BAND_HIGH,
    noise: float = MONOTONICITY_NOISE,
) -> dict:
    """PREREG §7 / user-directed search policy for a totally-ordered knob
    ladder (easiest -> hardest), given `measure(entry) -> F16 accuracy`.

    1. Binary search assuming accuracy is monotone non-increasing along the
       ladder, for the hardest entry with `measure(entry)` in [low, high].
    2. Confirm the boundary: measure the found entry's immediate harder
       neighbor, which must be out-of-band-low.
    3. If at any point two measured entries violate monotonicity by more
       than `noise` (a harder entry scoring higher than an easier one),
       abandon the binary search and fall back to measuring EVERY entry in
       the ladder, easiest to hardest (a linear walk) -- the final answer
       then comes from that complete measurement set.

    Returns:
        {"chosen": str | None, "measurements": {entry: acc, ...},
         "method": "binary" | "linear-fallback", "notes": [str, ...]}
    """
    measurements: dict[str, float] = {}
    notes: list[str] = []

    def m(entry: str) -> float:
        if entry not in measurements:
            measurements[entry] = measure(entry)
        return measurements[entry]

    def monotonicity_violated() -> bool:
        by_ladder_order = sorted(measurements.items(), key=lambda kv: ladder.index(kv[0]))
        for a in range(len(by_ladder_order)):
            for b in range(a + 1, len(by_ladder_order)):
                _, acc_easier = by_ladder_order[a]
                _, acc_harder = by_ladder_order[b]
                if acc_harder > acc_easier + noise:
                    return True
        return False

    def finalize(method: str) -> dict:
        in_band_entries = [e for e in ladder if e in measurements and in_band(measurements[e], low, high)]
        chosen = in_band_entries[-1] if in_band_entries else None
        return {
            "chosen": chosen,
            "measurements": dict(measurements),
            "method": method,
            "notes": list(notes),
        }

    def linear_fallback(reason: str) -> dict:
        notes.append(reason)
        for entry in ladder:
            m(entry)
        return finalize("linear-fallback")

    lo, hi = 0, len(ladder) - 1
    hardest_in_band_idx: int | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        acc = m(ladder[mid])
        if monotonicity_violated():
            return linear_fallback(
                f"monotonicity violated while measuring {ladder[mid]!r} "
                f"(acc={acc}); falling back to a linear walk"
            )
        if acc > high:
            lo = mid + 1
        elif acc < low:
            hi = mid - 1
        else:
            hardest_in_band_idx = mid
            lo = mid + 1

    if hardest_in_band_idx is None:
        notes.append("no in-band entry found by binary search")
        return finalize("binary")

    hardest = ladder[hardest_in_band_idx]
    if hardest_in_band_idx + 1 < len(ladder):
        neighbor = ladder[hardest_in_band_idx + 1]
        nacc = m(neighbor)
        if monotonicity_violated():
            return linear_fallback(
                f"monotonicity violated confirming neighbor {neighbor!r} "
                f"(acc={nacc}) of hardest-in-band {hardest!r}"
            )
        if in_band(nacc, low, high) or nacc > high:
            notes.append(
                f"boundary confirmation failed: harder neighbor {neighbor!r} of "
                f"{hardest!r} scored {nacc} (expected out-of-band-low); falling "
                f"back to a linear walk"
            )
            return linear_fallback(notes[-1])
    return finalize("binary")


def choose_target_tokens(
    acc_at_4096: float,
    acc_at_8192: float | None,
    low: float = BAND_LOW,
    high: float = BAND_HIGH,
) -> int:
    """§7 tie-break for the SAME variant across target_tokens: 8192 wins
    iff it is also in-band; otherwise 4096 (the default/first-tried value)
    stands. `acc_at_8192=None` means it was not measured."""
    del acc_at_4096  # kept for signature clarity / future use; not needed for the rule
    if acc_at_8192 is not None and in_band(acc_at_8192, low, high):
        return 8192
    return 4096


def select_factual_qa_mix(
    results: dict[str, float], low: float = BAND_LOW, high: float = BAND_HIGH
) -> dict:
    """§7: most tail-heavy in-band candidate wins, in order M1 > M2 > M3.
    If none land in-band, M3 is selected with disclosure of its
    out-of-band value (registered fallback)."""
    for name, _ in FACTUAL_QA_MIXES:
        if name in results and in_band(results[name], low, high):
            return {"chosen": name, "in_band": True, "note": None}
    if "M3" in results:
        return {
            "chosen": "M3",
            "in_band": False,
            "note": (
                f"no candidate mix landed in [{low}, {high}]; falling back to "
                f"M3 per §7, disclosing out-of-band F16 accuracy "
                f"{results['M3']}"
            ),
        }
    return {"chosen": None, "in_band": False, "note": "M3 was never measured"}


# ---------------------------------------------------------------------------
# Measurement store (resumable JSONL)
# ---------------------------------------------------------------------------


class MeasurementStore:
    """Append-only measurements.jsonl with lookup-by-key for resumability."""

    def __init__(self, path: Path):
        self.path = path
        self.records: list[dict] = []
        if path.exists():
            self.records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def find(self, **key) -> dict | None:
        for r in self.records:
            if all(r.get(k) == v for k, v in key.items()):
                return r
        return None

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        self.records.append(record)


# ---------------------------------------------------------------------------
# Corpus / alias-mapping bootstrap
# ---------------------------------------------------------------------------


def load_and_verify_corpus(path: Path = CORPUS_PATH, expected_sha256: str = CORPUS_SHA256) -> str:
    # NOTE: read as raw bytes and decode explicitly, NOT `Path.read_text()`.
    # `CORPUS_MANIFEST.md` §1 registers this corpus with its original `\r\n`
    # line endings preserved verbatim ("no other normalization"); text-mode
    # reading applies universal-newline translation (`\r\n` -> `\n`), which
    # silently changes the byte content and therefore the sha256 -- this
    # was caught by a real mismatch while smoke-testing this function
    # (`\r\n`-collapsed text hashed to 93e0c742...  instead of the
    # registered 0a21a138...).
    text = path.read_bytes().decode("utf-8")
    digest = __import__("hashlib").sha256(text.encode("utf-8")).hexdigest()
    if digest != expected_sha256:
        raise AssertionError(
            f"corpus {path} sha256 is {digest}, expected {expected_sha256}: refusing "
            f"to run PREREG §3.1 config 2b on an unverified corpus"
        )
    return text


def _load_fetch_wikidata_aliases_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fetch_wikidata_aliases", FETCH_ALIASES_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fetch_aliases_resilient(
    fwa, qids: list[str], max_retries: int = 8, base_sleep: float = 2.0
) -> tuple[dict[str, list[str]], int]:
    """Batch-fetch aliases exactly like `fwa.fetch_aliases`, using its own
    `entities_url`/`_get_json`/`parse_entities_response`/`chunk` helpers
    unmodified, but with exponential backoff on transient HTTP errors
    (429 Too Many Requests in particular -- the plain `fwa.fetch_aliases`
    has no retry and dies on the first one, which this deliberately wide
    841-QID union run hits reliably)."""
    import urllib.error

    aliases: dict[str, list[str]] = {}
    n_missing = 0
    batches = fwa.chunk(qids, fwa.BATCH_SIZE)
    for i, batch in enumerate(batches):
        url = fwa.entities_url(batch)
        for attempt in range(max_retries):
            try:
                data = fwa._get_json(url)
                break
            except urllib.error.HTTPError as e:
                if attempt == max_retries - 1:
                    raise
                sleep_s = base_sleep * (2 ** attempt)
                print(
                    f"  batch {i + 1}/{len(batches)}: {e} -- retrying in {sleep_s:.0f}s "
                    f"(attempt {attempt + 1}/{max_retries})",
                    file=sys.stderr,
                )
                time.sleep(sleep_s)
        batch_aliases = fwa.parse_entities_response(data)
        aliases.update(batch_aliases)
        n_missing += len(batch) - len(batch_aliases)
        print(f"  batch {i + 1}/{len(batches)}: {len(batch)} ids -> {len(batch_aliases)} resolved", file=sys.stderr)
        if i < len(batches) - 1:
            time.sleep(fwa.SLEEP_SECONDS)
    return aliases, n_missing


def ensure_alias_mapping(
    n_items: int = FACTUAL_QA_N_ITEMS,
    seed: int = FACTUAL_QA_SEED,
    out_path: Path = ALIAS_MAPPING_PATH_SEED2718,
    mixes: tuple[tuple[str, tuple[float, ...] | None], ...] = (
        ("M1", None), ("M2", M2_WEIGHTS), ("M3", M3_WEIGHTS),
    ),
) -> Path:
    """PREREG §3.4 branch (b), reparameterized by seed per the overnight
    work order: build the Wikidata alias-augmentation mapping for the
    confirmatory-seed record draw, via the SAME mechanical, output-blind
    fetch/parse machinery already used (and committed) for the
    characterization seed 7411 -- `scripts/fetch_wikidata_aliases.py`'s
    `fetch_aliases`/`parse_entities_response`, loaded and called directly
    (not subclassed or reimplemented) so this never drifts from that
    module's logic. Idempotent: does nothing if `out_path` already exists.
    MUST be called before any factual_qa generation for the confirmatory
    seed (output-blind rule).

    **Why this covers the UNION of all three §7 mixes, not just one draw**
    (a widening of `fetch_wikidata_aliases.py`'s own single-draw
    `build_mapping`, discovered while building this harness): §7's mixes
    change the PER-DECILE DRAW COUNT (`weights` -> `_apportion_counts`
    -> `takes`), and `picked_records` draws each decile with
    `rng.sample(decile, take)` against one shared, sequentially-advancing
    `random.Random(seed)` -- a different `take` for decile i changes how
    much of that shared RNG stream decile i consumes, which shifts every
    later decile's draw too. So M1/M2/M3 at the SAME seed pick materially
    DIFFERENT 500-record subsets (measured: only ~120-142 of M2's/M3's
    ~450/444 unique object QIDs overlap with M1's). Building the mapping
    for only one mix (as the seed-7411 precedent does, since that draw is
    never re-mixed) would leave most of M2's and M3's items unaugmented.
    This function instead unions `picked_records(seed=seed, weights=w)`
    over every registered mix `w` before fetching -- still one uniform
    rule ("fetch every sampled record's object entity's English label +
    aliases"), still zero per-item selection, still run and committed
    before any model output is read; only the SET of QIDs handed to that
    rule is widened to match what this calibration actually grades.
    """
    if out_path.exists():
        return out_path

    fwa = _load_fetch_wikidata_aliases_module()

    from datasets import load_dataset

    print(f"loading akariasai/PopQA test split (for alias mapping, seed={seed})...", file=sys.stderr)
    ds = load_dataset("akariasai/PopQA", split="test")

    qids: set[str] = set()
    per_mix_counts: dict[str, int] = {}
    for name, weights in mixes:
        picked = factual_qa.picked_records(ds, n_items, seed, weights=weights)
        mix_qids = {factual_qa.object_qid(r) for r in picked}
        per_mix_counts[name] = len(mix_qids)
        qids |= mix_qids
    qids_sorted = sorted(qids)
    print(
        f"seed={seed}: union of {[name for name, _ in mixes]} draws -> "
        f"{len(qids_sorted)} unique object QIDs (per-mix: {per_mix_counts})",
        file=sys.stderr,
    )

    aliases, n_missing = _fetch_aliases_resilient(fwa, qids_sorted)
    n_with_aliases = len(aliases)
    n_aliases_total = sum(len(v) for v in aliases.values())
    n_aliases_mean = round(n_aliases_total / n_with_aliases, 3) if n_with_aliases else 0.0

    header = {
        "retrieval_date": datetime.date.today().isoformat(),
        "rule": fwa.AUGMENTATION_RULE,
        "endpoint": (
            f"{fwa.WIKIDATA_API}?action=wbgetentities&ids=...&props=aliases|labels"
            f"&languages=en&format=json"
        ),
        "batch_size": fwa.BATCH_SIZE,
        "n_items_sampled": n_items,
        "seed": seed,
        "n_qids": len(qids_sorted),
        "n_qids_resolved": n_with_aliases,
        "n_qids_missing": n_missing,
        "n_aliases_total": n_aliases_total,
        "n_aliases_mean_per_qid": n_aliases_mean,
        "mixes_unioned": [name for name, _ in mixes],
        "per_mix_qid_counts": per_mix_counts,
        "note": (
            "Widened from a single-draw mapping (the seed-7411 precedent) to "
            "the UNION of picked_records(seed, weights=w) over every "
            "registered §7 mix: different mixes draw materially different "
            "record subsets at the same seed (different per-decile take "
            "counts shift the shared RNG stream). See "
            "scripts/calibrate_f16.py::ensure_alias_mapping docstring."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"_meta": header, "aliases": aliases}, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {out_path} ({n_with_aliases}/{len(qids_sorted)} QIDs resolved, "
        f"{n_aliases_total} aliases total)",
        file=sys.stderr,
    )
    return out_path


# ---------------------------------------------------------------------------
# Runtime driver -- longctx_retrieval
# ---------------------------------------------------------------------------


def _n_ctx_for(target_tokens: int) -> int:
    return target_tokens + LONGCTX_N_CTX_HEADROOM


class LongctxRunner:
    """Owns the (lazily-created, one-per-n_ctx) llama-cpp model handles and
    the HF tokenizer; runs the PREREG §3.1 tokenizer-match gate exactly
    once per process before the first generation; measures a
    (variant, target_tokens) cell, resuming from `store` when possible."""

    def __init__(self, f16_path: Path, hf_tokenizer, store: MeasurementStore, model_sha256: str):
        self.f16_path = f16_path
        self.hf_tokenizer = hf_tokenizer
        self.store = store
        self.model_sha256 = model_sha256
        self._llm_by_ctx: dict[int, object] = {}
        self.tokenizer_checked = False
        self.aborted = False
        self.abort_reason: str | None = None
        self.corpus_text = load_and_verify_corpus()

    def _llm_for(self, n_ctx: int):
        if n_ctx not in self._llm_by_ctx:
            gen = GenSettings(
                seed=GEN_SEED, temperature=0.0, top_k=1,
                max_tokens=LONGCTX_ANSWER_BUDGET, n_ctx=n_ctx,
            )
            self._llm_by_ctx[n_ctx] = gen_mod.make_llm(self.f16_path, gen)
        return self._llm_by_ctx[n_ctx]

    def close(self) -> None:
        """Explicitly free every llama-cpp model handle this runner
        created. llama-cpp-python's Metal backend can abort the process if
        a Llama object is instead freed by __del__ during interpreter
        shutdown (observed while smoke-testing this module); calling
        `.close()` deterministically, before the process starts tearing
        down, avoids that."""
        for llm in self._llm_by_ctx.values():
            llm.close()
        self._llm_by_ctx.clear()

    def _check_tokenizer_once(self, items) -> None:
        if self.tokenizer_checked or self.aborted:
            return
        self.tokenizer_checked = True
        # Registered 20-string sample: question strings of the run's first
        # 20 items, in id order.
        sample = [it.prompt for it in sorted(items, key=lambda i: i.id)[:20]]
        llm = self._llm_for(_n_ctx_for(LONGCTX_TARGET_TOKENS[0]))
        # BOS/special handling: HF side uses add_special_tokens=False (no
        # BOS, no special tokens); the matching llama-cpp call is
        # add_bos=False, special=False -- both sides then tokenize the raw
        # string with no extra tokens injected, which is what makes them
        # comparable at all.
        llama_tokenize = lambda s: llm.tokenize(s.encode("utf-8"), add_bos=False, special=False)  # noqa: E731
        try:
            longctx_retrieval.assert_tokenizer_match(self.hf_tokenizer, llama_tokenize, sample)
        except AssertionError as e:
            self.aborted = True
            self.abort_reason = str(e)
            raise

    def measure(self, variant: str, target_tokens: int) -> float | None:
        """Returns F16 accuracy for (variant, target_tokens), or None if
        the tokenizer-match gate has already aborted this model's longctx
        run."""
        if self.aborted:
            return None

        cached = self.store.find(
            suite="longctx_retrieval", variant=variant, target_tokens=target_tokens,
            n_items=LONGCTX_N_ITEMS, seed=LONGCTX_SEED,
        )
        if cached is not None:
            print(f"  [cache] longctx {variant} t{target_tokens}: acc={cached['accuracy']}")
            return cached["accuracy"]

        items = longctx_retrieval.build_items(
            self.hf_tokenizer, self.corpus_text, CORPUS_SHA256,
            n_items=LONGCTX_N_ITEMS, seed=LONGCTX_SEED,
            variant=variant, target_tokens=target_tokens,
        )
        self._check_tokenizer_once(items)
        if self.aborted:
            return None

        n_ctx = _n_ctx_for(target_tokens)
        llm = self._llm_for(n_ctx)
        gen = GenSettings(
            seed=GEN_SEED, temperature=0.0, top_k=1,
            max_tokens=LONGCTX_ANSWER_BUDGET, n_ctx=n_ctx,
        )
        print(f"  generating longctx {variant} t{target_tokens} (n={len(items)}, n_ctx={n_ctx})...")
        t0 = time.time()
        records = gen_mod.run_items(llm, items, "F16", self.model_sha256, gen)
        wall = time.time() - t0

        by_id = {it.id: it for it in items}
        n_correct = sum(
            1 for r in records if longctx_retrieval.grade(by_id[r.item_id], r.text) == "correct"
        )
        acc = n_correct / len(items)

        record = {
            "suite": "longctx_retrieval",
            "variant": variant,
            "target_tokens": target_tokens,
            "n_items": LONGCTX_N_ITEMS,
            "seed": LONGCTX_SEED,
            "gen_seed": GEN_SEED,
            "answer_budget": LONGCTX_ANSWER_BUDGET,
            "n_ctx": n_ctx,
            "accuracy": acc,
            "n_correct": n_correct,
            "item_set_sha256": item_set_sha256(items),
            "wall_time_s": round(wall, 2),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "provisional_note": PROVISIONAL_LONGCTX_NOTE,
        }
        self.store.append(record)
        print(f"    acc={acc:.3f} ({n_correct}/{len(items)}) in {wall:.1f}s")
        return acc


def run_longctx_calibration(f16_path: Path, hf_tokenizer, store: MeasurementStore, model_sha256: str) -> dict:
    runner = LongctxRunner(f16_path, hf_tokenizer, store, model_sha256)
    try:
        return _run_longctx_calibration(runner)
    finally:
        # Deterministic cleanup: see LongctxRunner.close's docstring for
        # why this matters (a Metal-backend __del__-at-shutdown abort seen
        # while smoke-testing this module).
        runner.close()


def _run_longctx_calibration(runner: "LongctxRunner") -> dict:
    def measure_at_4096(variant: str) -> float:
        acc = runner.measure(variant, 4096)
        if acc is None:
            # tokenizer-match gate aborted; treat as "impossibly hard" so
            # the ladder search terminates immediately without a chosen
            # variant, rather than raising mid-search.
            raise TokenizerMismatchAbort(runner.abort_reason or "tokenizer mismatch")
        return acc

    try:
        search_4096 = calibrate_ladder(LONGCTX_LADDER, measure_at_4096)
    except TokenizerMismatchAbort as e:
        return {
            "aborted": True,
            "abort_reason": str(e),
            "chosen_variant": None,
            "chosen_target_tokens": None,
            "provisional_note": PROVISIONAL_LONGCTX_NOTE,
        }

    result = {
        "aborted": False,
        "search_4096": search_4096,
        "provisional_note": PROVISIONAL_LONGCTX_NOTE,
    }

    hardest = search_4096["chosen"]
    if hardest is None:
        # §7: "the search runs at target_tokens 4096 first, then 8192" --
        # if NOTHING on the ladder is in-band at 4096 (every variant,
        # including the hardest, scored above the band -- too easy even at
        # the ladder's end), that is not the end of the search: run the
        # SAME full ladder search at t=8192, since a longer document can
        # itself make retrieval harder. (Observed for real: qwen2.5-7b-
        # instruct scored >=0.98 on every measured t=4096 variant up to
        # and including multivalue4.)
        def measure_at_8192_full(variant: str) -> float:
            acc = runner.measure(variant, 8192)
            if acc is None:
                raise TokenizerMismatchAbort(runner.abort_reason or "tokenizer mismatch")
            return acc

        search_8192_full = calibrate_ladder(LONGCTX_LADDER, measure_at_8192_full)
        result["search_8192_full_ladder"] = search_8192_full
        if search_8192_full["chosen"] is not None:
            result["chosen_variant"] = search_8192_full["chosen"]
            result["chosen_target_tokens"] = 8192
            return result

        # Nothing in-band anywhere in the registered candidate space
        # (variant ladder x target_tokens in {4096, 8192}). §7 registers a
        # fallback for factual_qa's out-of-band case (M3 + disclosure) but
        # NONE for longctx -- this is a genuine registration gap, not a
        # decision this harness can make on its own (OPEN_QUESTIONS.md
        # §5). Record the terminal state gracefully instead of picking a
        # setting or crashing.
        result["chosen_variant"] = None
        result["chosen_target_tokens"] = None
        hardest_setting_variant = LONGCTX_LADDER[-1]
        hardest_accuracy = search_8192_full["measurements"].get(hardest_setting_variant)
        if hardest_accuracy is None:
            # Defensive: the binary search always reaches the ladder's last
            # entry when every measurement is "too easy" (see this
            # function's comment above), but measure it directly if for
            # any reason it wasn't touched.
            hardest_accuracy = runner.measure(hardest_setting_variant, 8192)
        if hardest_accuracy is not None and hardest_accuracy > BAND_HIGH:
            result["terminal_state"] = "above_band_everywhere"
            result["hardest_setting"] = {
                "variant": hardest_setting_variant, "target_tokens": 8192,
            }
            result["hardest_accuracy"] = hardest_accuracy
            result["note"] = (
                "Entire registered candidate space (variant ladder x "
                "target_tokens in {4096, 8192}) scored above the "
                f"[{BAND_LOW}, {BAND_HIGH}] band; §7 registers no longctx "
                "fallback for this terminal state (only factual_qa has "
                "one). Not decided here -- see OPEN_QUESTIONS.md §5."
            )
        return result

    acc_4096 = search_4096["measurements"][hardest]

    # §7: test THE SAME variant at t=8192.
    acc_8192 = runner.measure(hardest, 8192)
    chosen_t = choose_target_tokens(acc_4096, acc_8192)
    result["acc_at_4096"] = acc_4096
    result["acc_at_8192_same_variant"] = acc_8192
    result["chosen_target_tokens_before_8192_probe"] = chosen_t

    # Also probe whether a HARDER variant is in-band at 8192: linear-probe
    # forward from `hardest` while in-band results keep coming, at t=8192.
    hardest_idx = LONGCTX_LADDER.index(hardest)
    probe_8192: dict[str, float] = {hardest: acc_8192} if acc_8192 is not None else {}
    best_at_8192 = hardest if (acc_8192 is not None and in_band(acc_8192)) else None
    if acc_8192 is not None and in_band(acc_8192):
        for idx in range(hardest_idx + 1, len(LONGCTX_LADDER)):
            v = LONGCTX_LADDER[idx]
            a = runner.measure(v, 8192)
            if a is None:
                break
            probe_8192[v] = a
            if in_band(a):
                best_at_8192 = v
            else:
                break
    result["probe_8192"] = probe_8192

    if best_at_8192 is not None:
        result["chosen_variant"] = best_at_8192
        result["chosen_target_tokens"] = 8192
    elif chosen_t == 8192:
        result["chosen_variant"] = hardest
        result["chosen_target_tokens"] = 8192
    else:
        result["chosen_variant"] = hardest
        result["chosen_target_tokens"] = 4096
    return result


class TokenizerMismatchAbort(Exception):
    pass


# ---------------------------------------------------------------------------
# Runtime driver -- factual_qa
# ---------------------------------------------------------------------------


def run_factual_qa_calibration(f16_path: Path, store: MeasurementStore, model_sha256: str) -> dict:
    ensure_alias_mapping()
    alias_mapping = json.loads(ALIAS_MAPPING_PATH_SEED2718.read_text())["aliases"]

    from datasets import load_dataset

    ds = load_dataset("akariasai/PopQA", split="test")

    gen = GenSettings(
        seed=GEN_SEED, temperature=0.0, top_k=1,
        max_tokens=FACTUAL_QA_ANSWER_BUDGET, n_ctx=FACTUAL_QA_N_CTX,
    )
    llm = None  # created lazily, only if any mix is uncached

    results: dict[str, float] = {}
    for name, weights in FACTUAL_QA_MIXES:
        cached = store.find(
            suite="factual_qa", mix=name, n_items=FACTUAL_QA_N_ITEMS, seed=FACTUAL_QA_SEED,
        )
        if cached is not None:
            print(f"  [cache] factual_qa {name}: acc={cached['accuracy']}")
            results[name] = cached["accuracy"]
            continue

        items = factual_qa.items_from_records(
            ds, FACTUAL_QA_N_ITEMS, FACTUAL_QA_SEED, weights=weights,
            alias_augmentation=alias_mapping,
        )
        if llm is None:
            llm = gen_mod.make_llm(f16_path, gen)
        print(f"  generating factual_qa {name} (n={len(items)})...")
        t0 = time.time()
        records = gen_mod.run_items(llm, items, "F16", model_sha256, gen)
        wall = time.time() - t0

        by_id = {it.id: it for it in items}
        n_correct = sum(
            1 for r in records if factual_qa.grade(by_id[r.item_id], r.text) == "correct"
        )
        acc = n_correct / len(items)
        record = {
            "suite": "factual_qa",
            "mix": name,
            "n_items": FACTUAL_QA_N_ITEMS,
            "seed": FACTUAL_QA_SEED,
            "gen_seed": GEN_SEED,
            "answer_budget": FACTUAL_QA_ANSWER_BUDGET,
            "accuracy": acc,
            "n_correct": n_correct,
            "item_set_sha256": item_set_sha256(items),
            "wall_time_s": round(wall, 2),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        store.append(record)
        print(f"    acc={acc:.3f} ({n_correct}/{len(items)}) in {wall:.1f}s")
        results[name] = acc

    if llm is not None:
        # Deterministic cleanup -- see LongctxRunner.close's docstring.
        llm.close()

    selection = select_factual_qa_mix(results)
    return {"measurements": results, **selection}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--f16", required=True, type=Path)
    ap.add_argument("--hf-tokenizer", required=True, type=Path)
    ap.add_argument("--suite", choices=["longctx", "factual_qa", "both"], default="both")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    store = MeasurementStore(out_dir / "measurements.jsonl")
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary["model_id"] = args.model_id
    summary["f16_path"] = str(args.f16)
    model_sha256 = sha256_file(args.f16)
    summary["f16_sha256"] = model_sha256

    if args.suite in ("longctx", "both"):
        from transformers import AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(str(args.hf_tokenizer))
        print(f"=== longctx_retrieval calibration for {args.model_id} ===")
        summary["longctx_retrieval"] = run_longctx_calibration(args.f16, hf_tokenizer, store, model_sha256)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary["longctx_retrieval"], indent=2, default=str))

    if args.suite in ("factual_qa", "both"):
        print(f"=== factual_qa calibration for {args.model_id} ===")
        summary["factual_qa"] = run_factual_qa_calibration(args.f16, store, model_sha256)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary["factual_qa"], indent=2, default=str))

    print(f"summary written: {summary_path}")


if __name__ == "__main__":
    main()
