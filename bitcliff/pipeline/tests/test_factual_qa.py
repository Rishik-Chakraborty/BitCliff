import json

import pytest

from bitcliff_pipeline.items import EvalItem
from bitcliff_pipeline.suites.factual_qa import (
    grade,
    items_from_records,
    load_popqa_items,
    object_qid,
    picked_records,
)

PROMPT_PREFIX = "Answer with just the answer: "


def _record(i, pop):
    return {
        "question": f"Q{i}: what is entity {i} known for?",
        "possible_answers": json.dumps([f"answer-{i}", f"alt answer {i}"]),
        "s_pop": pop,
    }


# 200 records, popularity == index, already sorted ascending -> 10 deciles
# of exactly 20 records each (200 / 10 = 20, no remainder in decile sizing).
RECORDS = [_record(i, pop=i) for i in range(200)]


def _decile_of(question: str) -> int:
    # "Q{i}: ..." -> i; popularity == i above, so i // 20 gives the decile.
    i = int(question.split(":")[0][1:])
    return i // 20


def _deciles_hit(items):
    deciles = [0] * 10
    for item in items:
        question = item.prompt[len(PROMPT_PREFIX):]
        deciles[_decile_of(question)] += 1
    return deciles


def test_items_are_deterministic():
    a = items_from_records(RECORDS, n_items=50, seed=7)
    b = items_from_records(RECORDS, n_items=50, seed=7)
    assert a == b


def test_n_items_exact():
    items = items_from_records(RECORDS, n_items=50, seed=7)
    assert len(items) == 50


def test_decile_coverage_even_split():
    items = items_from_records(RECORDS, n_items=50, seed=7)
    assert _deciles_hit(items) == [5] * 10


def test_remainder_goes_to_earliest_deciles():
    items = items_from_records(RECORDS, n_items=23, seed=7)
    assert len(items) == 23
    assert _deciles_hit(items) == [3, 3, 3, 2, 2, 2, 2, 2, 2, 2]


def test_different_seed_changes_selection():
    a = items_from_records(RECORDS, n_items=50, seed=1)
    b = items_from_records(RECORDS, n_items=50, seed=2)
    assert a != b


# ---------------------------------------------------------------------------
# Non-uniform mix (PREREG §7): per-decile counts by largest-remainder
# apportionment of weight_i * n, ties toward the lower decile index, then
# the seeded per-decile draw exactly as in the uniform case.
# ---------------------------------------------------------------------------

# PREREG §7 M2 shape, w_i = (11-i)/55, expressed in this function's decile
# order (index 0..9): non-integer raw counts at any n not divisible by 55.
M2_WEIGHTS = tuple((10 - i) / 55 for i in range(10))


def test_weights_none_is_byte_identical_to_uniform_default():
    assert items_from_records(RECORDS, n_items=50, seed=7, weights=None) == \
        items_from_records(RECORDS, n_items=50, seed=7)


def test_m2_style_weights_produce_largest_remainder_apportionment():
    items = items_from_records(RECORDS, n_items=50, seed=7, weights=M2_WEIGHTS)
    assert len(items) == 50
    # raw = 50 * (10..1)/55 = (9.09, 8.18, 7.27, 6.36, 5.45, 4.55, 3.64,
    # 2.73, 1.82, 0.91); floors sum to 45, the 5 largest remainders are
    # deciles 9, 8, 7, 6, 5.
    assert _deciles_hit(items) == [9, 8, 7, 6, 5, 5, 4, 3, 2, 1]


def test_apportionment_ties_break_toward_lower_decile_index():
    weights = (0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05)
    items = items_from_records(RECORDS, n_items=10, seed=7, weights=weights)
    # raw = (1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5): shortfall 4,
    # eight deciles tie at remainder .5 -> the four lowest indices win.
    assert _deciles_hit(items) == [2, 2, 2, 2, 1, 1, 0, 0, 0, 0]


def test_weighted_draw_is_deterministic_per_seed():
    a = items_from_records(RECORDS, n_items=50, seed=7, weights=M2_WEIGHTS)
    b = items_from_records(RECORDS, n_items=50, seed=7, weights=M2_WEIGHTS)
    assert a == b


def test_explicit_uniform_weights_match_the_default_rule():
    # M1 (uniform) through the apportionment path lands on the same counts
    # as the registered remainders-to-earliest-deciles default.
    a = items_from_records(RECORDS, n_items=23, seed=7, weights=(0.1,) * 10)
    b = items_from_records(RECORDS, n_items=23, seed=7)
    assert a == b


def test_ids_and_shape():
    items = items_from_records(RECORDS[:20], n_items=10, seed=3)
    assert items[0].id == "factual_qa-3-0000"
    assert items[-1].id == "factual_qa-3-0009"
    assert all(item.suite == "factual_qa" for item in items)
    assert all(item.expected is not None for item in items)
    assert all(item.prompt.startswith(PROMPT_PREFIX) for item in items)


def _item(aliases):
    return EvalItem(
        id="factual_qa-1-0000", suite="factual_qa", prompt="p", expected=tuple(aliases)
    )


def test_grade_alias_hit_correct():
    item = _item(["Paris"])
    assert grade(item, "The answer is Paris") == "correct"


def test_grade_case_insensitive():
    item = _item(["Paris"])
    assert grade(item, "the answer is paris") == "correct"


def test_grade_alias_inside_longer_word_is_wrong():
    item = _item(["Paris"])
    assert grade(item, "The Parisian streets are lovely.") == "wrong"


def test_grade_punctuation_adjacent_alias_is_correct():
    item = _item(["Paris"])
    assert grade(item, "It's Paris.") == "correct"


def test_grade_multi_word_alias():
    item = _item(["political leader"])
    assert grade(item, "He is a political leader in the region.") == "correct"
    assert grade(item, "He is a leader, political in nature.") == "wrong"


def test_grade_no_alias_is_wrong():
    item = _item(["Paris"])
    assert grade(item, "The answer is London") == "wrong"


def _full_item(question, aliases):
    return EvalItem(
        id="factual_qa-1-0000",
        suite="factual_qa",
        prompt=PROMPT_PREFIX + question,
        expected=tuple(aliases),
    )


def test_grade_subject_echo_alias_in_question_is_ignored():
    # The Asti pattern: "Asti" is a valid alias AND the question's subject.
    # An output that echoes the subject but gives a different, wrong actual
    # answer must not be scored correct just because "Asti" appears.
    item = _full_item(
        "What is Asti the capital of?",
        ["Province of Asti", "Asti", "provincia di Asti", "Asti province"],
    )
    assert grade(item, "Asti is the capital of Piedmont, Italy.") == "wrong"


def test_grade_normal_item_unaffected_by_subject_echo_guard():
    # Alias absent from the question, present in the output -> still correct.
    item = _full_item("What is the capital of France?", ["Paris"])
    assert grade(item, "The answer is Paris") == "correct"


def test_grade_all_aliases_in_question_forces_wrong():
    # Every alias already appears in the question -> auto-wrong regardless
    # of what the model outputs (empty effective-alias set).
    item = _full_item("Is Paris the capital of France, Paris?", ["Paris"])
    assert grade(item, "Paris") == "wrong"
    assert grade(item, "London") == "wrong"


# ---------------------------------------------------------------------------
# Mechanical Wikidata alias augmentation (PREREG §3.4 branch (b)).
#
# The object entity's Wikidata QID is derived from the record's `o_uri`
# field ("http://www.wikidata.org/entity/Q82955" -> "Q82955"). NOTE: PopQA's
# `obj_id` column is an internal numeric id, NOT the Wikidata QID (verified
# against datasets-server: obj_id=2834605 for a record whose o_uri encodes
# Q82955 -- the numbers don't correspond) -- `o_uri` is the reliable source.
# ---------------------------------------------------------------------------


def _aug_record(i, pop, qid, aliases):
    return {
        "question": f"Q{i}: what is entity {i} known for?",
        "possible_answers": json.dumps(aliases),
        "s_pop": pop,
        "o_uri": f"http://www.wikidata.org/entity/{qid}",
    }


# 10 records, popularity == index, already sorted ascending -> 10 deciles of
# exactly 1 record each -> the seeded draw is forced regardless of seed,
# giving a fully deterministic picked-record order for these tests.
AUG_RECORDS = [_aug_record(i, pop=i, qid=f"Q{700 + i}", aliases=[f"answer-{i}"]) for i in range(10)]


def test_object_qid_extracts_from_o_uri():
    assert object_qid({"o_uri": "http://www.wikidata.org/entity/Q82955"}) == "Q82955"


def test_object_qid_raises_on_unrecognized_uri():
    with pytest.raises(ValueError):
        object_qid({"o_uri": "not-a-wikidata-uri"})


def test_picked_records_matches_items_from_records_order():
    picked = picked_records(AUG_RECORDS, n_items=10, seed=3)
    items = items_from_records(AUG_RECORDS, n_items=10, seed=3)
    assert [PROMPT_PREFIX + r["question"] for r in picked] == [i.prompt for i in items]


def test_alias_augmentation_none_is_byte_identical():
    a = items_from_records(AUG_RECORDS, n_items=10, seed=3, alias_augmentation=None)
    b = items_from_records(AUG_RECORDS, n_items=10, seed=3)
    assert a == b


def test_alias_augmentation_omitted_matches_explicit_none():
    a = items_from_records(AUG_RECORDS, n_items=10, seed=3)
    b = items_from_records(AUG_RECORDS, n_items=10, seed=3, alias_augmentation=None)
    assert a == b


def test_alias_augmentation_merges_new_aliases_preserving_original_order():
    records = [_aug_record(0, pop=0, qid="Q500", aliases=["Original One", "original two"])]
    augmentation = {"Q500": ["Extra Alias", "original ONE"]}  # dup differs only in case
    items = items_from_records(records, n_items=1, seed=1, alias_augmentation=augmentation)
    # original aliases keep their order and casing first; "original ONE" is
    # a case-insensitive dupe of "Original One" and is dropped; the
    # remaining new alias is appended.
    assert items[0].expected == ("Original One", "original two", "Extra Alias")


def test_alias_augmentation_no_entry_for_qid_leaves_expected_unchanged():
    records = [_aug_record(0, pop=0, qid="Q999", aliases=["Solo"])]
    items = items_from_records(
        records, n_items=1, seed=1, alias_augmentation={"Q1": ["Other"]}
    )
    assert items[0].expected == ("Solo",)


def test_augmented_alias_matching_output_is_correct():
    records = [_aug_record(0, pop=0, qid="Q500", aliases=["Original"])]
    augmentation = {"Q500": ["Augmented Alias"]}
    items = items_from_records(records, n_items=1, seed=1, alias_augmentation=augmentation)
    item = items[0]
    assert grade(item, "The answer is Augmented Alias.") == "correct"
    assert grade(item, "unrelated text") == "wrong"


def test_load_popqa_items_reads_alias_augmentation_file(monkeypatch, tmp_path):
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda name, split: AUG_RECORDS)

    aug_path = tmp_path / "aliases.json"
    aug_path.write_text(json.dumps({"aliases": {"Q705": ["Bonus Alias"]}}))

    items = load_popqa_items(10, seed=3, alias_augmentation_path=aug_path)
    assert len(items) == 10
    item5 = items[5]  # decile 5 -> record index 5 -> qid Q705 (see AUG_RECORDS)
    assert item5.expected == ("answer-5", "Bonus Alias")


def test_load_popqa_items_without_alias_augmentation_path_is_unaugmented(monkeypatch):
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda name, split: AUG_RECORDS)
    items = load_popqa_items(10, seed=3)
    assert items[5].expected == ("answer-5",)


def test_load_popqa_items_forwards_weights_to_items_from_records(monkeypatch):
    """0B P3 wiring gap: `load_popqa_items` must forward an explicit
    `weights` vector (PREREG §7's popularity-mix knob) through to
    `items_from_records`, not silently default to the uniform mix."""
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda name, split: RECORDS)
    m3 = (0.16, 0.16, 0.16, 0.16, 0.16, 0.04, 0.04, 0.04, 0.04, 0.04)

    weighted = load_popqa_items(50, seed=7, weights=m3)
    unweighted = load_popqa_items(50, seed=7)
    direct = items_from_records(RECORDS, n_items=50, seed=7, weights=m3)

    assert [i.id for i in weighted] == [i.id for i in direct]
    assert weighted != unweighted
