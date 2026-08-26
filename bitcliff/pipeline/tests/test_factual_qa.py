import json

from bitcliff_pipeline.items import EvalItem
from bitcliff_pipeline.suites.factual_qa import grade, items_from_records

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
