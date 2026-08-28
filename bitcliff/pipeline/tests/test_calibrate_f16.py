"""Unit tests for scripts/calibrate_f16.py's PURE logic: the calibration
search policy over a fake accuracy oracle, item-set hashing, the
target_tokens tie-break, and the factual_qa mix selection rule. No model,
no network, no llama-cpp/transformers/datasets import.

Loaded by file path (same pattern as tests/test_fetch_wikidata_aliases.py):
the script has no package `__init__.py`.
"""

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "calibrate_f16.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("calibrate_f16", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cal = _load_module()


# ---------------------------------------------------------------------------
# in_band
# ---------------------------------------------------------------------------


def test_in_band_true_inside_range():
    assert cal.in_band(0.7) is True
    assert cal.in_band(0.6) is True  # inclusive lower
    assert cal.in_band(0.85) is True  # inclusive upper


def test_in_band_false_outside_range():
    assert cal.in_band(0.59) is False
    assert cal.in_band(0.86) is False


# ---------------------------------------------------------------------------
# item_set_sha256
# ---------------------------------------------------------------------------


class _FakeItem:
    def __init__(self, id, prompt, expected, prompt_tokens=None):
        self.id = id
        self.prompt = prompt
        self.expected = expected
        self.prompt_tokens = prompt_tokens


def test_item_set_sha256_deterministic():
    items = [
        _FakeItem("b", "question b", ("1234",), (1, 2, 3)),
        _FakeItem("a", "question a", ("5678",), (4, 5, 6)),
    ]
    h1 = cal.item_set_sha256(items)
    h2 = cal.item_set_sha256(list(reversed(items)))
    assert h1 == h2  # order-independent (sorted by id internally)
    assert len(h1) == 64  # sha256 hex digest


def test_item_set_sha256_changes_with_different_tokens():
    a = [_FakeItem("x", "q", ("1234",), (1, 2, 3))]
    b = [_FakeItem("x", "q", ("1234",), (1, 2, 4))]
    assert cal.item_set_sha256(a) != cal.item_set_sha256(b)


def test_item_set_sha256_changes_with_different_golds():
    a = [_FakeItem("x", "q", ("1234",), (1, 2, 3))]
    b = [_FakeItem("x", "q", ("9999",), (1, 2, 3))]
    assert cal.item_set_sha256(a) != cal.item_set_sha256(b)


def test_item_set_sha256_text_prompt_items_hash_too():
    # factual_qa items have no prompt_tokens; the prompt text stands in.
    a = [_FakeItem("x", "what is it", ("answer",), None)]
    b = [_FakeItem("x", "what is it", ("answer",), None)]
    assert cal.item_set_sha256(a) == cal.item_set_sha256(b)


# ---------------------------------------------------------------------------
# calibrate_ladder — the search-policy pure function over a fake oracle.
# ---------------------------------------------------------------------------

LADDER = ("single", "multikey4", "multikey8", "multikey12", "multivalue2",
          "multiquery2", "multiquery3", "multivalue3", "multiquery4", "multivalue4")


def test_calibrate_ladder_monotone_decreasing_finds_hardest_in_band_via_binary():
    # Smooth monotone-decreasing accuracy curve with a clean in-band window.
    acc_by_variant = {
        "single": 0.98, "multikey4": 0.95, "multikey8": 0.90, "multikey12": 0.87,
        "multivalue2": 0.80, "multiquery2": 0.70, "multiquery3": 0.62,
        "multivalue3": 0.50, "multiquery4": 0.30, "multivalue4": 0.10,
    }
    result = cal.calibrate_ladder(LADDER, lambda v: acc_by_variant[v])
    assert result["chosen"] == "multiquery3"  # hardest with 0.6 <= acc <= 0.85
    assert result["method"] == "binary"
    assert not result["notes"] or "no in-band" not in result["notes"][0]
    # Binary search should not need to measure every single entry.
    assert len(result["measurements"]) < len(LADDER)


def test_calibrate_ladder_non_monotone_triggers_linear_fallback():
    # A harder variant scores meaningfully higher than an easier one --
    # this must be discovered during the search and trigger the fallback.
    acc_by_variant = {
        "single": 0.95, "multikey4": 0.90, "multikey8": 0.85, "multikey12": 0.80,
        "multivalue2": 0.75, "multiquery2": 0.70, "multiquery3": 0.65,
        "multivalue3": 0.60, "multiquery4": 0.98,  # non-monotone spike
        "multivalue4": 0.05,
    }
    result = cal.calibrate_ladder(LADDER, lambda v: acc_by_variant[v])
    assert result["method"] == "linear-fallback"
    assert any("monotonicity" in n.lower() for n in result["notes"])
    # Linear fallback measures the whole ladder.
    assert set(result["measurements"]) == set(LADDER)
    # Hardest in-band across the full (now exhaustive) measurement set.
    assert result["chosen"] == "multivalue3"


def test_calibrate_ladder_no_in_band_entry_returns_none_chosen():
    # Everything is either too easy or too hard -- no in-band window at all.
    acc_by_variant = {
        "single": 0.99, "multikey4": 0.97, "multikey8": 0.95, "multikey12": 0.90,
        "multivalue2": 0.20, "multiquery2": 0.15, "multiquery3": 0.10,
        "multivalue3": 0.05, "multiquery4": 0.02, "multivalue4": 0.01,
    }
    result = cal.calibrate_ladder(LADDER, lambda v: acc_by_variant[v])
    assert result["chosen"] is None


def test_calibrate_ladder_boundary_confirmation_neighbor_in_band_triggers_fallback():
    # Binary search's probing might land on an in-band point whose harder
    # neighbor is ALSO in-band (i.e. binary search under-called the true
    # boundary) -- this must be caught by the confirmation step.
    acc_by_variant = {
        "single": 0.95, "multikey4": 0.90, "multikey8": 0.84, "multikey12": 0.82,
        "multivalue2": 0.80, "multiquery2": 0.78, "multiquery3": 0.76,
        "multivalue3": 0.74, "multiquery4": 0.30, "multivalue4": 0.10,
    }
    result = cal.calibrate_ladder(LADDER, lambda v: acc_by_variant[v])
    # Whatever path it takes, the final chosen entry's harder neighbor (if
    # any) must not be in-band -- i.e. the result is self-consistent.
    chosen = result["chosen"]
    if chosen is not None:
        idx = LADDER.index(chosen)
        if idx + 1 < len(LADDER):
            neighbor = LADDER[idx + 1]
            assert neighbor in result["measurements"]
            assert not cal.in_band(result["measurements"][neighbor])


def test_calibrate_ladder_is_deterministic():
    acc_by_variant = {v: (len(LADDER) - i) / len(LADDER) for i, v in enumerate(LADDER)}
    a = cal.calibrate_ladder(LADDER, lambda v: acc_by_variant[v])
    b = cal.calibrate_ladder(LADDER, lambda v: acc_by_variant[v])
    assert a == b


# ---------------------------------------------------------------------------
# _run_longctx_calibration — the orchestration around calibrate_ladder,
# exercised with a fake runner (only needs .measure(variant, target_tokens)
# -> float|None and .abort_reason; no real model/tokenizer/corpus).
# ---------------------------------------------------------------------------


class _FakeLongctxRunner:
    def __init__(self, acc_by_variant_and_t):
        self.acc_by_variant_and_t = acc_by_variant_and_t
        self.abort_reason = None
        self.calls = []

    def measure(self, variant, target_tokens):
        self.calls.append((variant, target_tokens))
        return self.acc_by_variant_and_t.get((variant, target_tokens))


def test_run_longctx_calibration_falls_through_to_8192_when_whole_4096_ladder_too_easy():
    # Real scenario observed on qwen2.5-7b-instruct: every t=4096 variant
    # the binary search touches (up to and including the ladder's hardest
    # entry) scores above the band -- nothing in-band at 4096 at all. §7:
    # "the search runs at target_tokens 4096 first, then 8192" -- this
    # must not silently give up; it must search the full ladder at 8192.
    acc = {}
    for v in LADDER:
        acc[(v, 4096)] = 0.99  # too easy everywhere at 4096
    # At 8192 the task gets harder; multivalue4 (hardest) lands in-band.
    for v in LADDER:
        acc[(v, 8192)] = 0.95
    acc[("multivalue4", 8192)] = 0.70

    runner = _FakeLongctxRunner(acc)
    result = cal._run_longctx_calibration(runner)

    assert result["aborted"] is False
    assert result["search_4096"]["chosen"] is None
    assert result["chosen_variant"] == "multivalue4"
    assert result["chosen_target_tokens"] == 8192
    assert result["search_8192_full_ladder"]["chosen"] == "multivalue4"


def test_run_longctx_calibration_stays_none_when_nothing_in_band_at_either_target():
    acc = {(v, 4096): 0.99 for v in LADDER}
    acc.update({(v, 8192): 0.95 for v in LADDER})  # still too easy everywhere

    runner = _FakeLongctxRunner(acc)
    result = cal._run_longctx_calibration(runner)

    assert result["chosen_variant"] is None
    assert result["chosen_target_tokens"] is None


def test_run_longctx_calibration_records_above_band_everywhere_terminal_state():
    # Real scenario hit on qwen2.5-7b-instruct: every setting in the
    # registered candidate space, including the hardest
    # (multivalue4 @ 8192), scores above 0.85. §7 has no longctx fallback
    # for this (unlike factual_qa's M3 rule) -- must record a clean
    # terminal state, not crash or silently pick a setting.
    acc = {(v, 4096): 0.99 for v in LADDER}
    acc.update({(v, 8192): 1.0 for v in LADDER})

    runner = _FakeLongctxRunner(acc)
    result = cal._run_longctx_calibration(runner)

    assert result["chosen_variant"] is None
    assert result["chosen_target_tokens"] is None
    assert result["terminal_state"] == "above_band_everywhere"
    assert result["hardest_setting"] == {"variant": "multivalue4", "target_tokens": 8192}
    assert result["hardest_accuracy"] == 1.0
    assert "OPEN_QUESTIONS" in result["note"]


def test_run_longctx_calibration_no_terminal_state_label_when_below_band_not_above():
    # Nothing in-band, but the hardest setting is BELOW the band, not
    # above it -- this is a different (already-handled-elsewhere-in-
    # principle) situation and must NOT be mislabeled
    # "above_band_everywhere".
    acc = {(v, 4096): 0.99 for v in LADDER}
    acc.update({(v, 8192): 0.99 for v in LADDER})
    acc[("multivalue4", 8192)] = 0.10  # hardest setting: below band, not above

    runner = _FakeLongctxRunner(acc)
    result = cal._run_longctx_calibration(runner)

    assert result["chosen_variant"] is None
    assert "terminal_state" not in result


def test_run_longctx_calibration_normal_path_when_4096_finds_hardest_in_band():
    # Sanity check that the ordinary (4096-succeeds) path is untouched by
    # the new fallback: hardest-in-band found directly at 4096, same
    # variant out-of-band-low at 8192 -> 4096 stands.
    acc_by_variant = {
        "single": 0.98, "multikey4": 0.95, "multikey8": 0.90, "multikey12": 0.87,
        "multivalue2": 0.80, "multiquery2": 0.70, "multiquery3": 0.62,
        "multivalue3": 0.50, "multiquery4": 0.30, "multivalue4": 0.10,
    }
    acc = {(v, 4096): a for v, a in acc_by_variant.items()}
    # multiquery3 (the hardest-in-band-at-4096 variant) at 8192:
    acc[("multiquery3", 8192)] = 0.20  # out of band -> 4096 should stand

    runner = _FakeLongctxRunner(acc)
    result = cal._run_longctx_calibration(runner)

    assert "search_8192_full_ladder" not in result
    assert result["chosen_variant"] == "multiquery3"
    assert result["chosen_target_tokens"] == 4096


# ---------------------------------------------------------------------------
# choose_target_tokens — the 4096-vs-8192 tie-break.
# ---------------------------------------------------------------------------


def test_choose_target_tokens_8192_wins_when_in_band():
    assert cal.choose_target_tokens(0.70, 0.65) == 8192


def test_choose_target_tokens_4096_stands_when_8192_out_of_band():
    assert cal.choose_target_tokens(0.70, 0.30) == 4096
    assert cal.choose_target_tokens(0.70, 0.95) == 4096


def test_choose_target_tokens_4096_stands_when_8192_not_measured():
    assert cal.choose_target_tokens(0.70, None) == 4096


def test_choose_target_tokens_8192_wins_at_exact_band_edges():
    assert cal.choose_target_tokens(0.70, 0.6) == 8192
    assert cal.choose_target_tokens(0.70, 0.85) == 8192


# ---------------------------------------------------------------------------
# select_factual_qa_mix — §7 M1 > M2 > M3 preference + fallback disclosure.
# ---------------------------------------------------------------------------


def test_select_factual_qa_mix_prefers_m1_when_in_band():
    result = cal.select_factual_qa_mix({"M1": 0.70, "M2": 0.75, "M3": 0.80})
    assert result["chosen"] == "M1"
    assert result["in_band"] is True
    assert result["note"] is None


def test_select_factual_qa_mix_falls_through_to_m2_when_m1_out_of_band():
    result = cal.select_factual_qa_mix({"M1": 0.30, "M2": 0.72, "M3": 0.80})
    assert result["chosen"] == "M2"
    assert result["in_band"] is True


def test_select_factual_qa_mix_falls_through_to_m3():
    result = cal.select_factual_qa_mix({"M1": 0.20, "M2": 0.40, "M3": 0.65})
    assert result["chosen"] == "M3"
    assert result["in_band"] is True


def test_select_factual_qa_mix_none_in_band_falls_back_to_m3_with_disclosure():
    result = cal.select_factual_qa_mix({"M1": 0.10, "M2": 0.20, "M3": 0.92})
    assert result["chosen"] == "M3"
    assert result["in_band"] is False
    assert result["note"] is not None
    assert "0.92" in result["note"]


# ---------------------------------------------------------------------------
# Registered weight vectors — sanity checks (normalize to 1.0, correct
# shape) and the direction note in the module docstring: weight increases
# toward the ascending-`s_pop` (i.e. popular) end for M2/M3, since
# `factual_qa.items_from_records`'s weights[i] applies to deciles[i] built
# by ascending s_pop.
# ---------------------------------------------------------------------------


def test_m1_weights_uniform():
    assert cal.M1_WEIGHTS == (0.1,) * 10


def test_weights_sum_to_one():
    assert abs(sum(cal.M1_WEIGHTS) - 1.0) < 1e-9
    assert abs(sum(cal.M2_WEIGHTS) - 1.0) < 1e-9
    assert abs(sum(cal.M3_WEIGHTS) - 1.0) < 1e-9


def test_m2_weights_increase_toward_popular_ascending_index():
    # Strictly increasing: more weight on higher-s_pop (more popular) deciles.
    assert list(cal.M2_WEIGHTS) == sorted(cal.M2_WEIGHTS)
    assert cal.M2_WEIGHTS[0] < cal.M2_WEIGHTS[-1]


def test_m3_weights_step_shape_favors_popular_half():
    assert cal.M3_WEIGHTS[:5] == (0.04,) * 5
    assert cal.M3_WEIGHTS[5:] == (0.16,) * 5


# ---------------------------------------------------------------------------
# Corpus verification
# ---------------------------------------------------------------------------


def test_load_and_verify_corpus_rejects_wrong_hash(tmp_path):
    bad = tmp_path / "corpus.txt"
    bad.write_text("not the count of monte cristo")
    import pytest

    with pytest.raises(AssertionError, match="sha256"):
        cal.load_and_verify_corpus(bad, expected_sha256="0" * 64)


def test_load_and_verify_corpus_accepts_matching_hash(tmp_path):
    text = "hello world"
    p = tmp_path / "corpus.txt"
    p.write_text(text)
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert cal.load_and_verify_corpus(p, expected_sha256=digest) == text


def test_load_and_verify_corpus_preserves_crlf_line_endings(tmp_path):
    # CORPUS_MANIFEST.md §1: the registered corpus keeps its original \r\n
    # line endings verbatim ("no other normalization"). A naive
    # `Path.read_text()` applies universal-newline translation (\r\n ->
    # \n), silently changing the byte content and therefore the sha256 --
    # this must NOT happen: reading and re-hashing must reproduce the
    # exact registered hash of the raw bytes.
    raw = b"line one\r\nline two\r\n"
    p = tmp_path / "corpus.txt"
    p.write_bytes(raw)
    import hashlib

    expected = hashlib.sha256(raw).hexdigest()
    text = cal.load_and_verify_corpus(p, expected_sha256=expected)
    assert text.encode("utf-8") == raw
    assert "\r\n" in text


# ---------------------------------------------------------------------------
# MeasurementStore — resumability
# ---------------------------------------------------------------------------


def test_measurement_store_find_returns_none_when_empty(tmp_path):
    store = cal.MeasurementStore(tmp_path / "measurements.jsonl")
    assert store.find(suite="longctx_retrieval", variant="single") is None


def test_measurement_store_append_and_find(tmp_path):
    path = tmp_path / "measurements.jsonl"
    store = cal.MeasurementStore(path)
    store.append({"suite": "factual_qa", "mix": "M1", "accuracy": 0.7})
    assert store.find(suite="factual_qa", mix="M1") == {"suite": "factual_qa", "mix": "M1", "accuracy": 0.7}
    assert store.find(suite="factual_qa", mix="M2") is None


def test_measurement_store_resumes_across_instances(tmp_path):
    path = tmp_path / "measurements.jsonl"
    cal.MeasurementStore(path).append({"suite": "longctx_retrieval", "variant": "single", "target_tokens": 4096, "accuracy": 0.9})
    reloaded = cal.MeasurementStore(path)
    found = reloaded.find(suite="longctx_retrieval", variant="single", target_tokens=4096)
    assert found is not None
    assert found["accuracy"] == 0.9
