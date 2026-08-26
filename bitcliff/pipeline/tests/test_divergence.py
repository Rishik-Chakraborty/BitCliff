from bitcliff_pipeline.divergence import divergence_for_records, first_divergence_index
from bitcliff_pipeline.generate import OutputRecord


def rec(text):
    return OutputRecord(
        item_id="x", suite="retrieval", quant_label="Q", model_sha256="h",
        prompt="p", text=text, finish_reason="stop", gen_settings={}, machine="m",
    )


def test_identical_sequences_return_none():
    assert first_divergence_index([1, 2, 3], [1, 2, 3]) is None


def test_first_mismatch_index():
    assert first_divergence_index([1, 2, 3], [1, 9, 3]) == 1


def test_prefix_returns_shorter_length():
    assert first_divergence_index([1, 2], [1, 2, 3]) == 2


def test_divergence_for_records_with_word_tokenizer():
    base = rec("the cat sat on the mat")
    other = rec("the cat slept on the mat")
    assert divergence_for_records(base, other, tokenize=str.split) == 2
    assert divergence_for_records(base, rec("the cat sat on the mat"), tokenize=str.split) is None
