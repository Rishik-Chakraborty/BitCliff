"""Config-vs-PREREG cross-check (RUN_0B.md §2 P3): the carrier-honesty
gate. Every `configs/0b/*.yaml` run config is asserted against constants
QUOTED FROM PREREG.md / Amendment 1 / the committed reference-manifests --
a config drifting from any of these fails CI. Every assertion below cites
the PREREG/RUN_0B/manifest section it enforces.

No network, no model: this test only loads YAML and committed manifest
JSON files already in the repo.
"""

import json
from pathlib import Path

import pytest

from bitcliff_pipeline.config import LadderConfig, load_config

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_0B_DIR = PIPELINE_ROOT / "configs" / "0b"
MANIFESTS_DIR = PIPELINE_ROOT / "reference-manifests"

# RUN_0B.md §5 "RULED 2026-08-30 -- canonical registered ladder": the
# per-model confirmatory rung set, identical for both reference models.
CANONICAL_LADDER_LABELS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K", "IQ2_M"]

# PREREG §7's registered factual_qa popularity-mix candidates (most
# tail-heavy first). M3 is the "step-tail-heavy" mix chosen for all three
# models per Amendment 1 §C ("None in [0.6, 0.85]... Chosen mix: M3").
M3_WEIGHTS = (0.16, 0.16, 0.16, 0.16, 0.16, 0.04, 0.04, 0.04, 0.04, 0.04)

# PREREG §3.1 "Configuration 2b": the STRIPPED-text sha256 ("The
# generator's corpus-hash gate accepts exactly this hash for 2b").
CORPUS_SHA256_2B = "0a21a13834b5215876bd4019af8fbc436abbfbb61b2826db62223eb990071443"

ALL_0B_CONFIG_PATHS = sorted(CONFIGS_0B_DIR.glob("*.yaml"))


def _load(path: Path) -> LadderConfig:
    return load_config(path)


def _manifest(name: str) -> dict:
    return json.loads((MANIFESTS_DIR / name).read_text())


def _files_by_path(manifest: dict) -> dict[str, dict]:
    return {f["path"]: f for f in manifest["gguf_files"]}


# ---------------------------------------------------------------------------
# Basic shape: every configs/0b/*.yaml file loads, and this test actually
# found all four RUN_0B.md §5 run-unit configs (a regression here would
# silently shrink the cross-check's coverage to nothing).
# ---------------------------------------------------------------------------


def test_all_four_0b_configs_exist_and_load():
    names = {p.name for p in ALL_0B_CONFIG_PATHS}
    assert names == {
        "0b-llama-8b-ladder.yaml",
        "0b-qwen-7b-ladder.yaml",
        "0b-shootout-arm1.yaml",
        "0b-arm2-official.yaml",
    }
    for path in ALL_0B_CONFIG_PATHS:
        _load(path)  # must not raise


LADDER_CONFIGS = [CONFIGS_0B_DIR / "0b-llama-8b-ladder.yaml", CONFIGS_0B_DIR / "0b-qwen-7b-ladder.yaml"]
ALL_CONFIGS_WITH_SUITES = ALL_0B_CONFIG_PATHS  # every 0B config carries the full suite block


# ---------------------------------------------------------------------------
# Generation settings -- PREREG §6 "Fixed generation parameters":
# "Global length budget: max_tokens: 1024", "Deterministic decoding:
# greedy, temperature 0.0, top_k 1, seed 42."
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ALL_0B_CONFIG_PATHS, ids=lambda p: p.name)
def test_generation_settings_match_prereg_section_6(path):
    cfg = _load(path)
    assert cfg.generation.seed == 42
    assert cfg.generation.temperature == 0.0
    assert cfg.generation.top_k == 1
    assert cfg.generation.max_tokens == 1024  # PREREG §6 global budget


@pytest.mark.parametrize("path", ALL_0B_CONFIG_PATHS, ids=lambda p: p.name)
def test_n_ctx_is_large_enough_for_the_longctx_t8192_prompt(path):
    # RUN_0B.md §2 P3: "n_ctx -- set 16384: longctx t=8192 prompts ~= 8300
    # tokens + answers." Not itself a PREREG-registered constant (n_ctx is
    # an engineering setting, not a registered value) -- asserted as a
    # floor with headroom over the documented ~8300-token prompt + the
    # largest per-suite answer budget, rather than pinned to one exact
    # integer, so this doesn't silently need editing if a config later
    # widens the margin.
    cfg = _load(path)
    assert cfg.generation.n_ctx >= 16384


# ---------------------------------------------------------------------------
# Per-suite answer budgets -- PREREG §6: "longctx_retrieval 32 tokens
# (paper-equivalent, §3.1); factual_qa 64 tokens."
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ALL_0B_CONFIG_PATHS, ids=lambda p: p.name)
def test_per_suite_answer_budgets_match_prereg_section_6(path):
    cfg = _load(path)
    assert cfg.suites["longctx_retrieval"]["max_tokens"] == 32
    assert cfg.suites["factual_qa"]["max_tokens"] == 64
    # arithmetic / arithmetic_twins use the global budget (PREREG §6: "arithmetic
    # suites use the global budget") -- no suite-level max_tokens override.
    assert "max_tokens" not in cfg.suites["arithmetic"]
    assert "max_tokens" not in cfg.suites["arithmetic_twins"]


# ---------------------------------------------------------------------------
# Seeds -- every one of these is registered as NEVER floating (PREREG §3.1
# 2b per Amendment 1 §A, §3.2, §3.3, §3.4; §7: "every sampling seed... may
# NOT be amended").
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ALL_0B_CONFIG_PATHS, ids=lambda p: p.name)
def test_registered_seeds_match_prereg(path):
    cfg = _load(path)
    # Amendment 1 §A: 2b longctx_retrieval registered n=96, seed=2024.
    assert cfg.suites["longctx_retrieval"]["seed"] == 2024
    # PREREG §3.2: "confirmatory item set... sampled with fixed seed 3141".
    assert cfg.suites["arithmetic"]["seed"] == 3141
    # PREREG §3.3: instantiated twin set is seed 1301.
    assert cfg.suites["arithmetic_twins"]["seed"] == 1301
    # PREREG §3.4: "confirmatory fixed seed 2718".
    assert cfg.suites["factual_qa"]["seed"] == 2718


# ---------------------------------------------------------------------------
# n's -- PREREG §3.1 Amendment 1 §A (2b n=96), §3.2 (arithmetic n=500),
# §3.3 (47 verified templates -> 94 pair items, fixed, not a config knob),
# §3.4 (factual_qa n=500).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ALL_0B_CONFIG_PATHS, ids=lambda p: p.name)
def test_registered_ns_match_prereg(path):
    cfg = _load(path)
    assert cfg.suites["longctx_retrieval"]["n_items"] == 96  # Amendment 1 §A
    assert cfg.suites["arithmetic"]["n_items"] == 500  # PREREG §3.2
    assert cfg.suites["factual_qa"]["n_items"] == 500  # PREREG §3.4
    # arithmetic_twins has no n_items config key: PREREG §3.3's n (47
    # template pairs = 94 items) is fixed by the verified template set
    # itself, not a config-supplied count (__main__.build_items passes
    # only `seed` through to arithmetic_twins.load_pair_items).
    assert "n_items" not in cfg.suites["arithmetic_twins"]


# ---------------------------------------------------------------------------
# longctx_retrieval knobs -- Amendment 1 §C/§D: chosen setting for all
# three models (including both 0B reference models) is
# variant=multivalue4, target_tokens=8192 (the §A2 out-of-band-high
# fallback, both reference models score above [0.6, 0.85] everywhere in
# the registered candidate space).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ALL_0B_CONFIG_PATHS, ids=lambda p: p.name)
def test_longctx_variant_and_target_tokens_match_amendment_1(path):
    cfg = _load(path)
    s = cfg.suites["longctx_retrieval"]
    assert s["variant"] == "multivalue4"
    assert s["target_tokens"] == 8192


@pytest.mark.parametrize("path", ALL_0B_CONFIG_PATHS, ids=lambda p: p.name)
def test_longctx_corpus_sha256_matches_prereg_section_3_1(path):
    cfg = _load(path)
    assert cfg.suites["longctx_retrieval"]["corpus_sha256"] == CORPUS_SHA256_2B


@pytest.mark.parametrize("path", ALL_0B_CONFIG_PATHS, ids=lambda p: p.name)
def test_longctx_corpus_path_points_at_the_registered_committed_corpus(path):
    cfg = _load(path)
    corpus_path = PIPELINE_ROOT / cfg.suites["longctx_retrieval"]["corpus_path"]
    assert corpus_path.name == "pg1184-monte-cristo.txt"  # PG-1184, PREREG §3.1
    assert corpus_path.exists()


# ---------------------------------------------------------------------------
# factual_qa M3 weights vector -- PREREG §7's registered M3 candidate
# ("step-tail-heavy... fixed 20% tail floor"), the mix Amendment 1 §C
# chose for every model (none landed in-band; M3 is most-tail-heavy
# in-band-eligible... none in-band -> M3 fallback per §7).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ALL_0B_CONFIG_PATHS, ids=lambda p: p.name)
def test_factual_qa_weights_match_prereg_section_7_m3(path):
    cfg = _load(path)
    weights = tuple(cfg.suites["factual_qa"]["weights"])
    assert weights == pytest.approx(M3_WEIGHTS)
    assert sum(weights) == pytest.approx(1.0)


@pytest.mark.parametrize("path", ALL_0B_CONFIG_PATHS, ids=lambda p: p.name)
def test_factual_qa_alias_augmentation_points_at_the_seed_2718_mapping(path):
    # Amendment 1 §B: the seed-2718 mapping (union of M1/M2/M3 QIDs), not
    # the seed-7411 characterization mapping.
    cfg = _load(path)
    aug_path = PIPELINE_ROOT / cfg.suites["factual_qa"]["alias_augmentation_path"]
    assert aug_path.name == "popqa_wikidata_aliases_seed2718.json"
    assert aug_path.exists()


# ---------------------------------------------------------------------------
# Canonical 7-rung ladder + exact filenames/sha256s -- RUN_0B.md §5
# "RULED 2026-08-30", cross-checked against the committed
# reference-manifests/*.json (the file pins PREREG §4 registers).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config_path,manifest_name",
    [
        (CONFIGS_0B_DIR / "0b-llama-8b-ladder.yaml", "llama-3.1-8b-bartowski.json"),
        (CONFIGS_0B_DIR / "0b-qwen-7b-ladder.yaml", "qwen2.5-7b-bartowski.json"),
    ],
    ids=["llama-8b-ladder", "qwen-7b-ladder"],
)
def test_ladder_config_has_exactly_the_canonical_7_rungs(config_path, manifest_name):
    cfg = _load(config_path)
    assert [q.label for q in cfg.quants] == CANONICAL_LADDER_LABELS


@pytest.mark.parametrize(
    "config_path,manifest_name",
    [
        (CONFIGS_0B_DIR / "0b-llama-8b-ladder.yaml", "llama-3.1-8b-bartowski.json"),
        (CONFIGS_0B_DIR / "0b-qwen-7b-ladder.yaml", "qwen2.5-7b-bartowski.json"),
    ],
    ids=["llama-8b-ladder", "qwen-7b-ladder"],
)
def test_ladder_config_filenames_and_sha256s_match_the_committed_manifest(config_path, manifest_name):
    cfg = _load(config_path)
    manifest = _manifest(manifest_name)
    files_by_path = _files_by_path(manifest)
    for q in cfg.quants:
        assert q.filename in files_by_path, (
            f"{config_path.name}: {q.filename!r} is not a file listed in "
            f"reference-manifests/{manifest_name}"
        )
        assert q.sha256 == files_by_path[q.filename]["sha256"], (
            f"{config_path.name}: {q.filename!r} sha256 does not match "
            f"reference-manifests/{manifest_name}"
        )
        assert q.uploader == "bartowski"
        assert q.imatrix is True


@pytest.mark.parametrize(
    "config_path,manifest_name",
    [
        (CONFIGS_0B_DIR / "0b-llama-8b-ladder.yaml", "llama-3.1-8b-bartowski.json"),
        (CONFIGS_0B_DIR / "0b-qwen-7b-ladder.yaml", "qwen2.5-7b-bartowski.json"),
    ],
    ids=["llama-8b-ladder", "qwen-7b-ladder"],
)
def test_ladder_config_filenames_actually_end_with_their_rung_label(config_path, manifest_name):
    # A config-authoring slip (e.g. Q2_K's filename accidentally pointing
    # at the Q2_K_L file) would still pass the manifest sha256 cross-check
    # above if the sha256 were copied from the wrong row too. This test
    # independently confirms every filename's own suffix names its rung.
    cfg = _load(config_path)
    for q in cfg.quants:
        assert q.filename.endswith(f"-{q.label}.gguf"), (
            f"{config_path.name}: quant labeled {q.label!r} has filename "
            f"{q.filename!r}, which does not end with '-{q.label}.gguf'"
        )


# ---------------------------------------------------------------------------
# Shootout Arm 1 (Llama-8B, unsloth/mradermacher-static/mradermacher-i1) --
# PREREG §5 Arm 1, file pins from reference-manifests/shootout-8b.json.
# ---------------------------------------------------------------------------


def test_shootout_arm1_has_exactly_the_registered_6_files():
    cfg = _load(CONFIGS_0B_DIR / "0b-shootout-arm1.yaml")
    assert cfg.model_id == "llama-3.1-8b-instruct"
    assert [q.label for q in cfg.quants] == [
        "unsloth_Q4_K_M", "unsloth_Q3_K_M",
        "mradermacher_static_Q4_K_M", "mradermacher_static_Q3_K_M",
        "mradermacher_i1_Q4_K_M", "mradermacher_i1_Q3_K_M",
    ]


def test_shootout_arm1_filenames_shas_and_repos_match_shootout_8b_manifest():
    cfg = _load(CONFIGS_0B_DIR / "0b-shootout-arm1.yaml")
    manifest = _manifest("shootout-8b.json")
    # Build filename -> (sha256, repo, uploader, imatrix) from every repo
    # entry in shootout-8b.json's "repos" list.
    by_filename = {}
    for repo_entry in manifest["repos"]:
        for f in repo_entry["gguf_files"]:
            by_filename[f["path"]] = {
                "sha256": f["sha256"],
                "repo": repo_entry["repo"],
                "uploader": f["uploader"],
                "imatrix": f["imatrix"],
            }
    assert len(cfg.quants) == 6
    for q in cfg.quants:
        assert q.filename in by_filename, f"{q.filename!r} not in shootout-8b.json"
        pin = by_filename[q.filename]
        assert q.sha256 == pin["sha256"]
        assert q.hf_repo == pin["repo"]
        assert q.uploader == pin["uploader"]
        assert q.imatrix == pin["imatrix"]


def test_shootout_arm1_quant_filter_is_q4_k_m_and_q3_k_m_only():
    # PREREG §5 Arm 1: "Q4_K_M and Q3_K_M only". Every label must end with
    # one of those two rungs.
    cfg = _load(CONFIGS_0B_DIR / "0b-shootout-arm1.yaml")
    for q in cfg.quants:
        assert q.label.endswith("Q4_K_M") or q.label.endswith("Q3_K_M")


def test_shootout_arm1_covers_all_three_repos_from_the_manifest():
    cfg = _load(CONFIGS_0B_DIR / "0b-shootout-arm1.yaml")
    manifest = _manifest("shootout-8b.json")
    expected_repos = {r["repo"] for r in manifest["repos"]}
    assert {q.hf_repo for q in cfg.quants} == expected_repos


# ---------------------------------------------------------------------------
# Arm 2 official (Qwen-7B) -- PREREG §5 Arm 2, file pins from
# reference-manifests/qwen2.5-7b-official.json. q4_k_m ships as 2 shards;
# `filename` must be the FIRST shard, the second recorded as extra_files.
# ---------------------------------------------------------------------------


def test_arm2_official_has_exactly_q4_k_m_and_q3_k_m():
    cfg = _load(CONFIGS_0B_DIR / "0b-arm2-official.yaml")
    assert cfg.model_id == "qwen2.5-7b-instruct"
    assert [q.label for q in cfg.quants] == ["Q4_K_M", "Q3_K_M"]


def test_arm2_official_q4_k_m_uses_the_first_shard_as_the_primary_filename():
    cfg = _load(CONFIGS_0B_DIR / "0b-arm2-official.yaml")
    q4 = next(q for q in cfg.quants if q.label == "Q4_K_M")
    assert q4.filename == "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    assert len(q4.extra_files) == 1
    assert q4.extra_files[0].filename == "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"


def test_arm2_official_filenames_and_shas_match_the_committed_manifest():
    cfg = _load(CONFIGS_0B_DIR / "0b-arm2-official.yaml")
    manifest = _manifest("qwen2.5-7b-official.json")
    files_by_path = _files_by_path(manifest)

    q4 = next(q for q in cfg.quants if q.label == "Q4_K_M")
    assert q4.sha256 == files_by_path[q4.filename]["sha256"]
    assert q4.extra_files[0].sha256 == files_by_path[q4.extra_files[0].filename]["sha256"]

    q3 = next(q for q in cfg.quants if q.label == "Q3_K_M")
    assert q3.filename == "qwen2.5-7b-instruct-q3_k_m.gguf"
    assert q3.sha256 == files_by_path[q3.filename]["sha256"]
    assert q3.extra_files == ()


# ---------------------------------------------------------------------------
# F16 baseline paths -- must point at the local conversions RUN_0B.md §5
# lists ("local conversion") and that actually exist on disk (models/f16).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config_path,expected_f16_name",
    [
        (CONFIGS_0B_DIR / "0b-llama-8b-ladder.yaml", "Llama-3.1-8B-Instruct-f16.gguf"),
        (CONFIGS_0B_DIR / "0b-shootout-arm1.yaml", "Llama-3.1-8B-Instruct-f16.gguf"),
        (CONFIGS_0B_DIR / "0b-qwen-7b-ladder.yaml", "Qwen2.5-7B-Instruct-f16.gguf"),
        (CONFIGS_0B_DIR / "0b-arm2-official.yaml", "Qwen2.5-7B-Instruct-f16.gguf"),
    ],
    ids=["llama-ladder", "shootout-arm1", "qwen-ladder", "arm2-official"],
)
def test_f16_baseline_path_matches_across_same_model_run_units(config_path, expected_f16_name):
    # RUN_0B.md §5: the shootout/official arms share the SAME F16 baseline
    # as their model's ladder run -- not a separately-converted file.
    cfg = _load(config_path)
    assert cfg.f16_path.name == expected_f16_name


def test_llama_configs_share_the_identical_f16_path():
    llama = _load(CONFIGS_0B_DIR / "0b-llama-8b-ladder.yaml")
    shootout = _load(CONFIGS_0B_DIR / "0b-shootout-arm1.yaml")
    assert llama.f16_path == shootout.f16_path


def test_qwen_configs_share_the_identical_f16_path():
    ladder = _load(CONFIGS_0B_DIR / "0b-qwen-7b-ladder.yaml")
    arm2 = _load(CONFIGS_0B_DIR / "0b-arm2-official.yaml")
    assert ladder.f16_path == arm2.f16_path


# ---------------------------------------------------------------------------
# tokenizer_path -- must match the model actually being run (a copy-paste
# slip here would silently gate the WRONG model's tokenizer against the
# GGUF, defeating PREREG §3.1's tokenizer-match assertion).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config_path,expected_tokenizer_dir",
    [
        (CONFIGS_0B_DIR / "0b-llama-8b-ladder.yaml", "Llama-3.1-8B-Instruct"),
        (CONFIGS_0B_DIR / "0b-shootout-arm1.yaml", "Llama-3.1-8B-Instruct"),
        (CONFIGS_0B_DIR / "0b-qwen-7b-ladder.yaml", "Qwen2.5-7B-Instruct"),
        (CONFIGS_0B_DIR / "0b-arm2-official.yaml", "Qwen2.5-7B-Instruct"),
    ],
    ids=["llama-ladder", "shootout-arm1", "qwen-ladder", "arm2-official"],
)
def test_tokenizer_path_matches_the_run_units_model(config_path, expected_tokenizer_dir):
    cfg = _load(config_path)
    tok_path = Path(cfg.suites["longctx_retrieval"]["tokenizer_path"])
    assert tok_path.name == expected_tokenizer_dir


# ---------------------------------------------------------------------------
# No stray sha256/hf_repo/extra_files typos: every ladder/shootout/arm2
# quant either has a sha256 pin or is explicitly exempt (none are, in
# these 0B configs -- every rung is manifest-pinned).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ALL_0B_CONFIG_PATHS, ids=lambda p: p.name)
def test_every_quant_in_every_0b_config_has_a_sha256_pin(path):
    cfg = _load(path)
    for q in cfg.quants:
        assert q.sha256, f"{path.name}: quant {q.label!r} has no config-pinned sha256"
        assert len(q.sha256) == 64  # a hex sha256 digest
        for extra in q.extra_files:
            assert extra.sha256, (
                f"{path.name}: quant {q.label!r}'s extra file "
                f"{extra.filename!r} has no config-pinned sha256"
            )
