from bitcliff_pipeline.suites.arithmetic import (
    extract_final_number,
    grade,
    items_from_records,
)

RECORDS = [
    {"question": f"Q{i}: If x = {i} and y = {i}, what is x + y?",
     "answer": f"x + y = {i} + {i} = {2 * i}\n#### {2 * i}"}
    for i in range(1, 21)
]


def test_extract_prefers_hash_marker():
    assert extract_final_number("some steps 3 + 4\n#### 7") == "7"


def test_extract_normalizes_commas_and_floats():
    assert extract_final_number("#### 1,000") == "1000"
    assert extract_final_number("#### 6.0") == "6"
    assert extract_final_number("#### 2.5") == "2.5"


def test_extract_falls_back_to_last_number():
    assert extract_final_number("The answer is 12, no wait, 14.") == "14"


def test_extract_none_when_no_number():
    assert extract_final_number("I cannot solve this.") is None


def test_items_are_deterministic_and_shaped():
    a = items_from_records(RECORDS, n_items=5, seed=1301)
    b = items_from_records(RECORDS, n_items=5, seed=1301)
    assert a == b
    assert all(item.suite == "arithmetic" for item in a)
    assert all(item.expected is not None for item in a)
    assert "####" in a[0].prompt  # instructs the answer format


def test_grade():
    (item,) = items_from_records(RECORDS[:1], n_items=1, seed=1)
    gold = item.expected[0]
    assert grade(item, f"steps...\n#### {gold}") == "correct"
    assert grade(item, "#### 999999") == "wrong"
    assert grade(item, "no answer at all") == "wrong"
