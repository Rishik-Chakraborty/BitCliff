"""The single home of every PREREG-registered constant used across the
pipeline (pre-rerun hardening, user ruling 2026-09-04).

Why this module exists: `configs/0b/*.yaml` hand-copied PREREG §7's M3
popularity-mix weight vector in PREREG's own prose order (decile 1 = most
popular), but `factual_qa.items_from_records` indexes `weights[i]` against
deciles it builds in ASCENDING `s_pop` order (index 0 = least popular).
`scripts/calibrate_f16.py` got the direction right (it reverses the vector);
the 0B configs and `tests/test_0b_configs.py`'s own cross-check constant did
not, so the CI check silently validated a config against its own bug and the
cloud confirmatory run sampled a non-registered factual_qa item set
(OPEN_QUESTIONS §8, 2026-09-04). The structural fix is this module: every
registered numeric/hash constant lives in exactly ONE place, every consumer
imports it (never retypes it), and a boot-time gate
(`__main__`'s generate stage) recomputes each suite's item-set hash and
refuses to generate if it doesn't match the value pinned here.

No I/O. Plain constants only, each cited to the PREREG/Amendment section (or
external file) it comes from. Values were re-verified against PREREG.md and
its Amendments 1-2 (and, for the two constants PREREG never registers,
computed locally — see `DERIVED_ITEM_SET_SHA256` below) while writing this
module, not copied blind from any prior draft.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

# PREREG §6: "Deterministic decoding: greedy, temperature 0.0, top_k 1,
# seed 42." The generation seed -- distinct from every item-construction
# seed below.
GEN_SEED = 42

# PREREG §3.2: "confirmatory item set ... sampled with fixed seed 3141."
ARITHMETIC_SEED = 3141

# PREREG §3.3: the instantiated twin set is built with seed 1301.
TWINS_SEED = 1301

# PREREG §3.1 Amendment 1 §A: "2b longctx_retrieval registered n=96,
# seed=2024."
LONGCTX_SEED = 2024

# PREREG §3.4: "confirmatory fixed seed 2718."
FACTUAL_QA_SEED = 2718

# OPEN_QUESTIONS §7, RESOLVED 2026-09-04 (user ruling): "seed 8271
# RATIFIED", together with the derived per-cell RNG rule
# (`8271:{run_id}:{quant_label}:{suite}`) and per-pair rule
# (`8271:pair:{name_a}:{name_b}:{suite}`) `scripts/analyze_0b.py`
# documents. PREREG §8 itself registers no bootstrap RNG seed -- this is a
# disclosed post-hoc ratification, not a §3 sampling seed. Kept as a
# string (not int): every consumer builds `random.Random(f"{SEED}:...")`
# from it, never uses it as a bare int seed.
BOOTSTRAP_SEED = "8271"

# ---------------------------------------------------------------------------
# Item counts (n's)
# ---------------------------------------------------------------------------

# PREREG §3.2: n=500.
ARITHMETIC_N = 500

# PREREG §3.4: n=500.
FACTUAL_QA_N = 500

# PREREG §3.1 Amendment 1 §A: 2b n=96.
LONGCTX_N = 96

# PREREG §3.3: all 47 verified templates -> 94 (original, twin) items. Not a
# config knob -- fixed by the verified template set itself.
TWINS_N = 94

# ---------------------------------------------------------------------------
# factual_qa popularity-mix weight vectors (PREREG §7)
# ---------------------------------------------------------------------------

# PREREG §7 registers three candidate popularity-decile mixes for the
# factual_qa calibration knob, written there in "decile 1 = most popular ..
# decile 10 = least popular" prose order. BUT
# `factual_qa.items_from_records` builds its ten deciles sorted by `s_pop`
# ASCENDING (decile 0 = least popular, decile 9 = most popular) and applies
# `weights[i]` to `deciles[i]` as built -- so every vector below is written
# in that ASCENDING-popularity index order (PREREG's listed vector,
# reversed), which is what `items_from_records` actually needs to reproduce
# the registered mix. Using PREREG's prose order un-reversed is exactly the
# bug OPEN_QUESTIONS §8 documents: it silently draws a different,
# non-registered 500-item set. `scripts/calibrate_f16.py`'s M1_WEIGHTS /
# M2_WEIGHTS / M3_WEIGHTS (pre-existing, convention-correct) are moved here
# verbatim; calibrate_f16.py now imports them instead of redefining them.
M1_WEIGHTS: tuple[float, ...] = (0.1,) * 10  # uniform
M2_WEIGHTS: tuple[float, ...] = tuple((i + 1) / 55 for i in range(10))  # linear-tail-heavy
M3_WEIGHTS_CONVENTION: tuple[float, ...] = (0.04,) * 5 + (0.16,) * 5  # step-tail-heavy

# ---------------------------------------------------------------------------
# Difficulty-calibration band (PREREG §7)
# ---------------------------------------------------------------------------

BAND_LOW = 0.6
BAND_HIGH = 0.85

# ---------------------------------------------------------------------------
# PREREG §8: margin, alpha, bootstrap resamples
# ---------------------------------------------------------------------------

# "Margin: M = 3 percentage points absolute accuracy, per suite."
MARGIN = 0.03

# The dual multiplicity rule's per-cell/headline significance level:
# "Per-cell verdicts are descriptive, at fixed alpha = 0.05."
ALPHA = 0.05

# "two-sided 95% CI on Delta-accuracy by paired bootstrap (10,000
# resamples)."
N_RESAMPLES = 10_000

# ---------------------------------------------------------------------------
# Per-suite generation budgets (PREREG §6)
# ---------------------------------------------------------------------------

LONGCTX_MAX_TOKENS = 32  # "longctx_retrieval 32 tokens (paper-equivalent, §3.1)"
FACTUAL_QA_MAX_TOKENS = 64  # "factual_qa 64 tokens"
GLOBAL_MAX_TOKENS = 1024  # "Global length budget: max_tokens: 1024"

# ---------------------------------------------------------------------------
# Corpus hashes (PREREG §3.1)
# ---------------------------------------------------------------------------

# Configuration 2b (site/dataset corpus): PG-1184 ("The Count of Monte
# Cristo"), STRIPPED-text sha256. "The generator's corpus-hash gate accepts
# exactly this hash for 2b."
CORPUS_2B_SHA256 = "0a21a13834b5215876bd4019af8fbc436abbfbb61b2826db62223eb990071443"

# Configuration 2a (paper-comparability corpus, never published): the
# `sgoel9/paul_graham_essays` corpus pinned at this sha256.
CORPUS_2A_SHA256 = "b6135331a3132d08cb84262870ae8f9d9acb6bae4cd7f0278926a64c38f9329e"

# ---------------------------------------------------------------------------
# F16 baseline file hashes -- runs-cloud/models/f16/computed-sha256.txt
# (cross-checked against runs-cloud/f16-recon-provenance.txt's matching HF
# revision pins; local F16 conversions of the gated/official repos).
# ---------------------------------------------------------------------------

LLAMA_F16_SHA256 = "139c255d857940bc945b2a3242fbbb2641b59191d79413bcad9b74fa1784c7b0"
QWEN7B_F16_SHA256 = "970ccec3ad83bb62aa25ce585bed4ebd297963257442afe697d350465933f2c5"

# ---------------------------------------------------------------------------
# Registered item-set hashes -- Amendment 1 §C's per-cell calibration
# results. Keyed (suite, model_id); a model-INDEPENDENT suite (factual_qa:
# "item construction is model-independent", Amendment 1 §C) is keyed
# (suite, None) and applies to every model.
# ---------------------------------------------------------------------------

ITEM_SET_SHA256: dict[tuple[str, str | None], str] = {
    # Amendment 1 §C: M3 item set, identical across all three models.
    ("factual_qa", None): "2e53ca0e73e9cbdd5d7bac672857ba918ff7b74d2405ddf62f37aa9a630091c6",
    # Amendment 1 §C: longctx_retrieval item construction depends on the
    # model's own tokenizer, so it IS model-keyed. qwen2.5-1.5b-instruct and
    # qwen2.5-7b-instruct share a tokenizer family and land on the identical
    # hash; llama-3.1-8b-instruct's differs.
    ("longctx_retrieval", "qwen2.5-1.5b-instruct"): (
        "ed18db130e4a035eef4bd581a8c53c7a17daf63085499d9123497991bd257f9c"
    ),
    ("longctx_retrieval", "qwen2.5-7b-instruct"): (
        "ed18db130e4a035eef4bd581a8c53c7a17daf63085499d9123497991bd257f9c"
    ),
    ("longctx_retrieval", "llama-3.1-8b-instruct"): (
        "9973be4a98d860e84f8002be91a80b2808a371466c0f60d6a2f073e79ad59bff"
    ),
}

# PREREG §3.3: the instantiated twin set's FILE sha256 (the embargoed
# `private/twins/twin_set_seed1301.jsonl`, 47 records) -- the integrity
# anchor committed in PREREG for a file that is itself excluded from the
# repo. This is a hash of that FILE's bytes, distinct in kind from an
# item_set_sha256 value (`hashing.item_set_sha256`, a hash over an
# in-memory EvalItem list) -- named accordingly, and never used as an
# ITEM_SET_SHA256 / DERIVED_ITEM_SET_SHA256 gate value.
TWINS_SET_SHA256 = "2002447536db8560c7160fc80ec7ba4f1c56074a1a949c1322114e8263d9e14b"

# ---------------------------------------------------------------------------
# Derived (non-registered) item-set hashes -- arithmetic and arithmetic_twins
# have NO registered item-set hash anywhere in PREREG or its Amendments.
# These two values were computed ONCE, locally, deterministically, from the
# registered seed/n via the real suite builders
# (`suites.arithmetic.load_gsm8k_items` / `suites.arithmetic_twins.
# load_pair_items`) and `hashing.item_set_sha256` -- see
# `tests/test_registered.py::test_derived_hashes_actually_reproduce_from_
# the_registered_seed_and_builders`, which rebuilds both sets from HF and
# re-asserts these exact values every test run.
#
# derived 2026-09-04 from the registered seed/n as a regression gate; not
# itself a registered value.
# ---------------------------------------------------------------------------

DERIVED_ITEM_SET_SHA256: dict[tuple[str, str | None], str] = {
    ("arithmetic", None): "27d341e90625c9350cc49b91b6cfa5268d5579a07ece37c6f85bbcf6420913f8",
    ("arithmetic_twins", None): "34841501bcefb084ab0de0729d681948758bb2764389b18bbe57cab0d11ec8fd",
}


def expected_item_set_sha256(suite: str, model_id: str) -> str | None:
    """The expected item-set hash for `suite` under `model_id`, or None if
    none is pinned (registered OR derived) -- the caller (the boot-time
    gate in `__main__`) must treat None as "unknown, refuse to generate".

    Lookup order: an exact (suite, model_id) registered pin, then a
    model-independent (suite, None) registered pin, then the same two for
    the derived (non-registered) table.
    """
    for table in (ITEM_SET_SHA256, DERIVED_ITEM_SET_SHA256):
        if (suite, model_id) in table:
            return table[(suite, model_id)]
        if (suite, None) in table:
            return table[(suite, None)]
    return None
