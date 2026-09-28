"""Config-vs-registered cross-check for `configs/0b-prime/*.yaml` (the 1.5B
confirmatory run, RUN_0B_PRIME.md) -- the same carrier-honesty gate as
`tests/test_0b_configs.py`, for the 1.5B. Every YAML literal is asserted
against `bitcliff_pipeline.registered` (the single home of registered
constants) and, for file pins, independently against the stamped manifest
JSON itself (reference-manifests/qwen2.5-1.5b-bartowski.json, whose own
file sha256 is checked against registered.QWEN15B_MANIFEST_SHA256 first).

The last test builds the 2a items from the ladder config through the real
`__main__.build_items` and the real boot-time item-set gate, offline (real
Qwen2.5-1.5B-Instruct tokenizer under models/hf/, cached
sgoel9/paul_graham_essays corpus) -- no network, no model.
"""

import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from bitcliff_pipeline import registered
from bitcliff_pipeline.config import LadderConfig, load_config

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PIPELINE_ROOT / "configs" / "0b-prime"
LADDER_PATH = CONFIGS_DIR / "qwen2.5-1.5b-ladder.yaml"
SMOKE_PATH = CONFIGS_DIR / "qwen2.5-1.5b-smoke.yaml"
MANIFEST_PATH = PIPELINE_ROOT / "reference-manifests" / "qwen2.5-1.5b-bartowski.json"
CALIBRATE_SCRIPT = PIPELINE_ROOT / "scripts" / "calibrate_f16.py"

MODEL_ID = "qwen2.5-1.5b-instruct"
SUITE_2A = "longctx_retrieval_2a"
SUITE_2B = "longctx_retrieval"

# Amendment 4 §C: the 7 registered rungs (F16 is config.f16_path, not a
# quants entry) -- derived from registered.py, never restated.
LADDER_LABELS = [label for label in registered.QWEN15B_LADDER_SHA256 if label != "F16"]
assert LADDER_LABELS == ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K", "IQ2_M"]


def _load(path: Path) -> LadderConfig:
    return load_config(path)


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def test_both_0b_prime_configs_exist_and_load():
    assert {p.name for p in CONFIGS_DIR.glob("*.yaml")} == {LADDER_PATH.name, SMOKE_PATH.name}
    for path in (LADDER_PATH, SMOKE_PATH):
        assert _load(path).model_id == MODEL_ID


def test_manifest_file_is_the_stamped_one():
    # Amendment 4 §D: the manifest these pins are cross-checked against is
    # the stamped file, byte-for-byte.
    assert hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest() == registered.QWEN15B_MANIFEST_SHA256


# ---------------------------------------------------------------------------
# Ladder: model/repo/F16/rungs -- Amendment 4 §C, manifest-pinned.
# ---------------------------------------------------------------------------


def test_ladder_model_repo_and_f16_path():
    cfg = _load(LADDER_PATH)
    manifest = _manifest()
    assert cfg.model_id == MODEL_ID
    assert cfg.hf_repo == manifest["repo"] == "bartowski/Qwen2.5-1.5B-Instruct-GGUF"
    assert str(cfg.f16_path) == manifest["f16"]["path"]
    assert cfg.f16_path.name == "Qwen2.5-1.5B-Instruct-f16.gguf"
    assert manifest["f16"]["sha256"] == registered.QWEN15B_LADDER_SHA256["F16"]


def test_ladder_has_exactly_the_amendment_4_rungs_in_order():
    cfg = _load(LADDER_PATH)
    assert [q.label for q in cfg.quants] == LADDER_LABELS
    assert LADDER_LABELS == _manifest()["confirmatory_ladder"]


def test_ladder_filenames_and_sha256s_match_registered_and_the_manifest():
    cfg = _load(LADDER_PATH)
    by_path = {f["path"]: f for f in _manifest()["gguf_files"]}
    for q in cfg.quants:
        assert q.sha256 == registered.QWEN15B_LADDER_SHA256[q.label]
        assert q.filename in by_path, f"{q.filename!r} not in the manifest"
        row = by_path[q.filename]
        assert q.sha256 == row["sha256"]
        assert row["rung"] == q.label and row["in_ladder"] is True
        assert q.imatrix is row["imatrix"] is True
        assert q.uploader == "bartowski"
        assert q.hf_repo is None and q.extra_files == () and q.spectacle_only is False
        assert q.filename.endswith(f"-{q.label}.gguf")


# ---------------------------------------------------------------------------
# Generation -- PREREG §6, identical to the 0B configs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [LADDER_PATH, SMOKE_PATH], ids=lambda p: p.name)
def test_generation_settings_identical_to_the_0b_configs(path):
    cfg = _load(path)
    assert cfg.generation.seed == registered.GEN_SEED
    assert cfg.generation.temperature == 0.0
    assert cfg.generation.top_k == 1
    assert cfg.generation.max_tokens == registered.GLOBAL_MAX_TOKENS
    assert cfg.generation == _load(PIPELINE_ROOT / "configs" / "0b" / "0b-qwen-7b-ladder.yaml").generation


# ---------------------------------------------------------------------------
# Suites -- registered n / seeds / knobs.
# ---------------------------------------------------------------------------


def test_ladder_carries_exactly_the_five_registered_suites():
    cfg = _load(LADDER_PATH)
    assert set(cfg.suites) == {SUITE_2A, SUITE_2B, "arithmetic", "arithmetic_twins", "factual_qa"}
    assert cfg.exploratory is False


def test_ladder_registered_ns_and_seeds():
    s = _load(LADDER_PATH).suites
    assert s[SUITE_2A]["n_items"] == registered.LONGCTX_2A_N
    assert s[SUITE_2A]["seed"] == registered.LONGCTX_SEED
    assert s[SUITE_2B]["n_items"] == registered.LONGCTX_N
    assert s[SUITE_2B]["seed"] == registered.LONGCTX_SEED
    assert s["arithmetic"]["n_items"] == registered.ARITHMETIC_N
    assert s["arithmetic"]["seed"] == registered.ARITHMETIC_SEED
    assert s["arithmetic_twins"] == {"seed": registered.TWINS_SEED}  # n fixed at 94, no knob
    assert s["factual_qa"]["n_items"] == registered.FACTUAL_QA_N
    assert s["factual_qa"]["seed"] == registered.FACTUAL_QA_SEED


def test_ladder_2a_block_matches_prereg_3_1_configuration_2a():
    s = _load(LADDER_PATH).suites[SUITE_2A]
    assert s["variant"] == "multivalue2"
    assert s["target_tokens"] == 4096
    assert s["corpus_dataset"] == "sgoel9/paul_graham_essays"
    assert "corpus_path" not in s
    assert s["corpus_sha256"] == registered.CORPUS_2A_SHA256
    assert Path(s["tokenizer_path"]) == Path("models/hf/Qwen2.5-1.5B-Instruct")
    assert s["max_tokens"] == registered.LONGCTX_MAX_TOKENS


def test_ladder_2a_corpus_dataset_is_the_vendored_generators_corpus():
    from bitcliff_pipeline.vendor import generate_multivalue2 as mv2

    assert _load(LADDER_PATH).suites[SUITE_2A]["corpus_dataset"] == mv2.CORPUS_DATASET


def test_ladder_2b_block_matches_amendment_1_for_the_1_5b():
    s = _load(LADDER_PATH).suites[SUITE_2B]
    # Amendment 1 §C: the 1.5B's in-band 2b setting is multivalue4 @ 8192.
    assert s["variant"] == "multivalue4"
    assert s["target_tokens"] == 8192
    assert s["corpus_sha256"] == registered.CORPUS_2B_SHA256
    assert (PIPELINE_ROOT / s["corpus_path"]).name == "pg1184-monte-cristo.txt"
    assert (PIPELINE_ROOT / s["corpus_path"]).exists()
    assert Path(s["tokenizer_path"]) == Path("models/hf/Qwen2.5-1.5B-Instruct")
    assert s["max_tokens"] == registered.LONGCTX_MAX_TOKENS


def test_ladder_factual_qa_m3_order_matches_calibrate_f16_convention():
    spec = importlib.util.spec_from_file_location("calibrate_f16_for_0b_prime", CALIBRATE_SCRIPT)
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)

    s = _load(LADDER_PATH).suites["factual_qa"]
    weights = tuple(s["weights"])
    assert weights == pytest.approx(cal.M3_WEIGHTS)
    assert weights == pytest.approx(registered.M3_WEIGHTS_CONVENTION)
    assert weights[0] < weights[-1]  # ascending popularity: tail deciles first (OPEN_QUESTIONS §8)
    assert s["max_tokens"] == registered.FACTUAL_QA_MAX_TOKENS
    aug = PIPELINE_ROOT / s["alias_augmentation_path"]
    assert aug.name == "popqa_wikidata_aliases_seed2718.json" and aug.exists()


def test_ladder_arithmetic_suites_use_the_global_budget():
    s = _load(LADDER_PATH).suites
    assert "max_tokens" not in s["arithmetic"]
    assert "max_tokens" not in s["arithmetic_twins"]


def test_ladder_nll_block_names_exactly_the_2a_and_2b_suites():
    cfg = _load(LADDER_PATH)
    assert cfg.nll is not None
    assert set(cfg.nll.suites) == {SUITE_2A, SUITE_2B}
    assert len(cfg.nll.suites) == 2


# ---------------------------------------------------------------------------
# Smoke config: exploratory, F16 + one ladder rung, n=5, otherwise the
# ladder's own suite blocks.
# ---------------------------------------------------------------------------


def test_smoke_is_exploratory_and_says_so():
    cfg = _load(SMOKE_PATH)
    assert cfg.exploratory is True
    text = SMOKE_PATH.read_text()
    header = text[: text.index("exploratory: true")]  # the leading comment block
    assert "EXPLORATORY" in header and "NON-REGISTERED" in header


def test_smoke_shares_model_f16_and_pinned_rung_with_the_ladder():
    smoke, ladder = _load(SMOKE_PATH), _load(LADDER_PATH)
    assert (smoke.model_id, smoke.hf_repo, smoke.f16_path) == (ladder.model_id, ladder.hf_repo, ladder.f16_path)
    assert len(smoke.quants) == 1
    assert smoke.quants[0] in ladder.quants


def test_smoke_suites_are_the_ladders_at_n_5_without_twins():
    smoke, ladder = _load(SMOKE_PATH), _load(LADDER_PATH)
    # arithmetic_twins omitted: fixed n=94, no n knob (PREREG §3.3).
    assert set(smoke.suites) == set(ladder.suites) - {"arithmetic_twins"}
    for name, block in smoke.suites.items():
        assert block["n_items"] == 5
        assert {**block, "n_items": ladder.suites[name]["n_items"]} == ladder.suites[name]
    assert smoke.nll == ladder.nll


# ---------------------------------------------------------------------------
# The real thing: build the 2a items from the ladder config and pass them
# through the real boot-time item-set gate (offline).
# ---------------------------------------------------------------------------


def test_2a_items_built_from_the_ladder_config_pass_the_real_item_set_gate():
    from transformers import AutoTokenizer

    from bitcliff_pipeline.__main__ import assert_item_sets_match_registered, build_items
    from bitcliff_pipeline.hashing import item_set_sha256

    cfg = _load(LADDER_PATH)
    only_2a = dataclasses.replace(cfg, suites={SUITE_2A: cfg.suites[SUITE_2A]})
    tokenizer = AutoTokenizer.from_pretrained(
        str(PIPELINE_ROOT / cfg.suites[SUITE_2A]["tokenizer_path"]), local_files_only=True,
    )
    specs: list = []
    items = build_items(only_2a, PIPELINE_ROOT, longctx_tokenizer=tokenizer, longctx_answer_specs=specs)

    assert len(items) == registered.LONGCTX_2A_N
    assert {i.suite for i in items} == {SUITE_2A}
    assert [s.item_id for s in specs] == [i.id for i in items]
    assert item_set_sha256(items) == registered.ITEM_SET_SHA256[(SUITE_2A, MODEL_ID)]
    assert_item_sets_match_registered(items, cfg.model_id)  # must not raise


def test_2b_items_built_from_the_ladder_config_pass_the_real_item_set_gate():
    """Sibling of the 2a test for 2b (suite key `longctx_retrieval`,
    multivalue4 @ 8192 over the committed Monte Cristo corpus, Amendment 1
    §C): built through the real `build_items` and the real boot-time gate,
    offline."""
    from transformers import AutoTokenizer

    from bitcliff_pipeline.__main__ import assert_item_sets_match_registered, build_items
    from bitcliff_pipeline.hashing import item_set_sha256

    cfg = _load(LADDER_PATH)
    only_2b = dataclasses.replace(cfg, suites={SUITE_2B: cfg.suites[SUITE_2B]})
    tokenizer = AutoTokenizer.from_pretrained(
        str(PIPELINE_ROOT / cfg.suites[SUITE_2B]["tokenizer_path"]), local_files_only=True,
    )
    specs: list = []
    items = build_items(only_2b, PIPELINE_ROOT, longctx_tokenizer=tokenizer, longctx_answer_specs=specs)

    assert len(items) == registered.LONGCTX_N
    assert {i.suite for i in items} == {SUITE_2B}
    assert [s.item_id for s in specs] == [i.id for i in items]
    assert item_set_sha256(items) == registered.ITEM_SET_SHA256[(SUITE_2B, MODEL_ID)]
    assert_item_sets_match_registered(items, cfg.model_id)  # must not raise
