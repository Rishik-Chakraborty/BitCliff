"""Tests for bitcliff_pipeline.registered — the single home of every
PREREG-registered constant (pre-rerun hardening, user ruling 2026-09-04,
OPEN_QUESTIONS §8). Values are cross-checked against PREREG.md / Amendment 1
/ Amendment 2 (the ratified bootstrap seed, OPEN_QUESTIONS §7) and the
committed runs-cloud F16 provenance files — this test file is the
regression gate for the module's verbatim-copied values.
"""

import pytest

from bitcliff_pipeline import registered


# ---------------------------------------------------------------------------
# Seeds — PREREG §3.1 (Amendment 1 §A), §3.2, §3.3, §3.4; OPEN_QUESTIONS §7.
# ---------------------------------------------------------------------------


def test_seeds_match_prereg():
    assert registered.GEN_SEED == 42
    assert registered.ARITHMETIC_SEED == 3141
    assert registered.TWINS_SEED == 1301
    assert registered.LONGCTX_SEED == 2024
    assert registered.FACTUAL_QA_SEED == 2718


def test_bootstrap_seed_ratified_2026_09_04():
    # OPEN_QUESTIONS §7: "seed 8271 RATIFIED" — kept as the string form the
    # existing analysis driver (scripts/analyze_0b.py) already builds its
    # per-cell RNG rule strings from (f"{SEED}:{run_id}:...").
    assert registered.BOOTSTRAP_SEED == "8271"


# ---------------------------------------------------------------------------
# Item counts — PREREG §3.1 Amendment 1 §A (2b n=96), §3.2 (500), §3.3 (all
# 47 pairs = 94 items, fixed), §3.4 (500).
# ---------------------------------------------------------------------------


def test_ns_match_prereg():
    assert registered.ARITHMETIC_N == 500
    assert registered.FACTUAL_QA_N == 500
    assert registered.LONGCTX_N == 96
    assert registered.TWINS_N == 94


# ---------------------------------------------------------------------------
# factual_qa popularity-mix weight vectors — PREREG §7, in the
# items_from_records convention (ascending s_pop; index 0 = least popular).
# See calibrate_f16.py's module docstring "Weight-vector direction note" —
# the source of truth for the direction bug this whole rerun exists to fix
# (OPEN_QUESTIONS §8).
# ---------------------------------------------------------------------------


def test_m1_weights_uniform():
    assert registered.M1_WEIGHTS == (0.1,) * 10
    assert sum(registered.M1_WEIGHTS) == pytest.approx(1.0)


def test_m2_weights_linear_tail_heavy_ascending():
    assert registered.M2_WEIGHTS == tuple((i + 1) / 55 for i in range(10))
    assert abs(sum(registered.M2_WEIGHTS) - 1.0) < 1e-9
    assert list(registered.M2_WEIGHTS) == sorted(registered.M2_WEIGHTS)


def test_m3_weights_convention_matches_registered_step_function():
    assert registered.M3_WEIGHTS_CONVENTION == (0.04,) * 5 + (0.16,) * 5
    assert sum(registered.M3_WEIGHTS_CONVENTION) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Calibration band + §8 statistics constants.
# ---------------------------------------------------------------------------


def test_calibration_band():
    assert registered.BAND_LOW == 0.6
    assert registered.BAND_HIGH == 0.85


def test_section_8_statistics_constants():
    assert registered.MARGIN == 0.03
    assert registered.ALPHA == 0.05
    assert registered.N_RESAMPLES == 10_000


# ---------------------------------------------------------------------------
# Per-suite answer budgets — PREREG §6.
# ---------------------------------------------------------------------------


def test_per_suite_budgets():
    assert registered.LONGCTX_MAX_TOKENS == 32
    assert registered.FACTUAL_QA_MAX_TOKENS == 64
    assert registered.GLOBAL_MAX_TOKENS == 1024


# ---------------------------------------------------------------------------
# Corpus hashes — PREREG §3.1.
# ---------------------------------------------------------------------------


def test_corpus_hashes():
    assert registered.CORPUS_2B_SHA256 == (
        "0a21a13834b5215876bd4019af8fbc436abbfbb61b2826db62223eb990071443"
    )
    assert registered.CORPUS_2A_SHA256 == (
        "b6135331a3132d08cb84262870ae8f9d9acb6bae4cd7f0278926a64c38f9329e"
    )
    assert len(registered.CORPUS_2B_SHA256) == 64
    assert len(registered.CORPUS_2A_SHA256) == 64


# ---------------------------------------------------------------------------
# F16 file hashes — runs-cloud/models/f16/computed-sha256.txt.
# ---------------------------------------------------------------------------


def test_f16_sha256s():
    assert registered.LLAMA_F16_SHA256 == (
        "139c255d857940bc945b2a3242fbbb2641b59191d79413bcad9b74fa1784c7b0"
    )
    assert registered.QWEN7B_F16_SHA256 == (
        "970ccec3ad83bb62aa25ce585bed4ebd297963257442afe697d350465933f2c5"
    )
    assert len(registered.LLAMA_F16_SHA256) == 64
    assert len(registered.QWEN7B_F16_SHA256) == 64


# ---------------------------------------------------------------------------
# Registered item-set hashes — Amendment 1 §C.
# ---------------------------------------------------------------------------


def test_item_set_sha256_factual_qa_model_independent():
    expected = "2e53ca0e73e9cbdd5d7bac672857ba918ff7b74d2405ddf62f37aa9a630091c6"
    assert registered.ITEM_SET_SHA256[("factual_qa", None)] == expected
    # Model-independent: applies to every model, not just a specific one.
    assert registered.expected_item_set_sha256("factual_qa", "qwen2.5-1.5b-instruct") == expected
    assert registered.expected_item_set_sha256("factual_qa", "qwen2.5-7b-instruct") == expected
    assert registered.expected_item_set_sha256("factual_qa", "llama-3.1-8b-instruct") == expected


def test_item_set_sha256_longctx_per_model():
    qwen_hash = "ed18db130e4a035eef4bd581a8c53c7a17daf63085499d9123497991bd257f9c"
    llama_hash = "9973be4a98d860e84f8002be91a80b2808a371466c0f60d6a2f073e79ad59bff"
    assert registered.ITEM_SET_SHA256[("longctx_retrieval", "qwen2.5-1.5b-instruct")] == qwen_hash
    assert registered.ITEM_SET_SHA256[("longctx_retrieval", "qwen2.5-7b-instruct")] == qwen_hash
    assert registered.ITEM_SET_SHA256[("longctx_retrieval", "llama-3.1-8b-instruct")] == llama_hash
    assert registered.expected_item_set_sha256("longctx_retrieval", "qwen2.5-1.5b-instruct") == qwen_hash
    assert registered.expected_item_set_sha256("longctx_retrieval", "llama-3.1-8b-instruct") == llama_hash


def test_twins_set_sha256_is_a_file_hash_not_an_item_set_hash():
    # PREREG §3.3: sha256 of the instantiated (embargoed) twin file itself,
    # private/twins/twin_set_seed1301.jsonl — distinct from an
    # item_set_sha256 (hashing.item_set_sha256) value, hence the separate
    # name (never keyed into ITEM_SET_SHA256/DERIVED_ITEM_SET_SHA256).
    assert registered.TWINS_SET_SHA256 == (
        "2002447536db8560c7160fc80ec7ba4f1c56074a1a949c1322114e8263d9e14b"
    )
    assert len(registered.TWINS_SET_SHA256) == 64


# ---------------------------------------------------------------------------
# Derived (non-registered, regression-gate-only) item-set hashes for
# arithmetic / arithmetic_twins — PREREG registers no item-set hash for
# either suite. These are computed once locally from the registered
# seed/n via the suite builders and pinned here as a drift gate.
# ---------------------------------------------------------------------------


def test_derived_item_set_sha256_present_for_arithmetic_and_twins():
    assert ("arithmetic", None) in registered.DERIVED_ITEM_SET_SHA256
    assert ("arithmetic_twins", None) in registered.DERIVED_ITEM_SET_SHA256
    for key in (("arithmetic", None), ("arithmetic_twins", None)):
        h = registered.DERIVED_ITEM_SET_SHA256[key]
        assert isinstance(h, str)
        assert len(h) == 64


def test_expected_item_set_sha256_falls_back_to_derived():
    assert registered.expected_item_set_sha256("arithmetic", "llama-3.1-8b-instruct") == (
        registered.DERIVED_ITEM_SET_SHA256[("arithmetic", None)]
    )
    assert registered.expected_item_set_sha256("arithmetic_twins", "qwen2.5-7b-instruct") == (
        registered.DERIVED_ITEM_SET_SHA256[("arithmetic_twins", None)]
    )


def test_expected_item_set_sha256_unknown_suite_or_model_returns_none():
    assert registered.expected_item_set_sha256("spectacle", "llama-3.1-8b-instruct") is None
    assert registered.expected_item_set_sha256("longctx_retrieval", "unknown-model") is None


def test_derived_hashes_actually_reproduce_from_the_registered_seed_and_builders():
    """The derived hashes aren't just asserted as literals here — this
    confirms they are exactly what building the item sets from the
    registered seed/n via the real suite builders (network: HF GSM8K)
    produces, so a future drift in either the builder or the pinned
    constant is caught."""
    from bitcliff_pipeline.hashing import item_set_sha256
    from bitcliff_pipeline.suites import arithmetic, arithmetic_twins

    arith_items = arithmetic.load_gsm8k_items(registered.ARITHMETIC_N, registered.ARITHMETIC_SEED)
    assert len(arith_items) == registered.ARITHMETIC_N
    assert item_set_sha256(arith_items) == registered.DERIVED_ITEM_SET_SHA256[("arithmetic", None)]

    twins_items = arithmetic_twins.load_pair_items(registered.TWINS_SEED)
    assert len(twins_items) == registered.TWINS_N
    assert item_set_sha256(twins_items) == registered.DERIVED_ITEM_SET_SHA256[("arithmetic_twins", None)]
