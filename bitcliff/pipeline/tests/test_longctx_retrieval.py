"""Tests for the longctx_retrieval suite: the bitcliff_pipeline adapter around
the vendored multivalue2 generator, and the paper's verbatim grading rule.

No network, no real tokenizer: a stub tokenizer exposes exactly the surface
the vendored generator needs (see `_split_chat_template`, `_build_document_ids`
and `_token_stream` in bitcliff_pipeline.vendor.generate_multivalue2):
  - `tokenizer(text, add_special_tokens=False)["input_ids"] -> list[int]`
  - `tokenizer.decode(ids) -> str`
  - `tokenizer.apply_chat_template(messages, tokenize=False,
     add_generation_prompt=True) -> str`
  - `tokenizer.name_or_path -> str`
"""

import hashlib

import pytest

from bitcliff_pipeline.items import EvalItem
from bitcliff_pipeline.suites import longctx_retrieval
from bitcliff_pipeline.vendor import generate_multivalue2 as mv2


class StubTokenizer:
    """Deterministic word<->id tokenizer. Splits on a single literal space,
    so token count only tracks word count, not word content — the vendored
    generator's needle/question templates keep a fixed word shape regardless
    of which key/value is drawn, which is what makes filler-length arithmetic
    ("document is exactly target_tokens") checkable with this stub.
    """

    def __init__(self, name: str = "stub-tokenizer"):
        self.name_or_path = name
        self._word_to_id: dict[str, int] = {}
        self._id_to_word: dict[int, str] = {}

    def _id_for(self, word: str) -> int:
        if word not in self._word_to_id:
            i = len(self._word_to_id)
            self._word_to_id[word] = i
            self._id_to_word[i] = word
        return self._word_to_id[word]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        ids = [self._id_for(w) for w in text.split(" ")]
        return {"input_ids": ids}

    def decode(self, ids) -> str:
        return " ".join(self._id_to_word[i] for i in ids)

    def apply_chat_template(
        self, messages, tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        content = messages[0]["content"]
        return f"<|user|>{content}<|assistant|>"


def make_tokenizer() -> StubTokenizer:
    return StubTokenizer()


CORPUS_TEXT = " ".join(f"corpusword{i}" for i in range(500))
CORPUS_SHA256 = hashlib.sha256(CORPUS_TEXT.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# grading — the paper rule, verbatim (GRADING.md)
# ---------------------------------------------------------------------------


def _item(expected):
    return EvalItem("longctx_retrieval-x", "longctx_retrieval", "desc", expected)


def test_grade_correct_when_all_present():
    item = _item(("4404", "5614"))
    assert longctx_retrieval.grade(item, "The passcodes are 4404 and 5614.") == "correct"


def test_grade_correct_order_insensitive():
    item = _item(("4404", "5614"))
    assert longctx_retrieval.grade(item, "5614, 4404") == "correct"


def test_grade_wrong_when_one_missing():
    item = _item(("4404", "5614"))
    assert longctx_retrieval.grade(item, "Only 4404 here.") == "wrong"


def test_grade_wrong_when_none_present():
    item = _item(("4404", "5614"))
    assert longctx_retrieval.grade(item, "no idea") == "wrong"


def test_grade_unanchored_substring_is_a_documented_laxity():
    # GRADING.md: "Substring match is not boundary-anchored. A gold `4404`
    # inside a longer digit run ... counts as a match." Lock this laxity.
    item = _item(("4404", "5614"))
    assert longctx_retrieval.grade(item, "The number was 44045614 apparently.") == "correct"


# ---------------------------------------------------------------------------
# EvalItem back-compat
# ---------------------------------------------------------------------------


def test_eval_item_4arg_construction_still_works():
    item = EvalItem("id", "suite", "prompt", ("a",))
    assert item.prompt_tokens is None


def test_eval_item_5arg_construction():
    item = EvalItem("id", "suite", "prompt", ("a",), (1, 2, 3))
    assert item.prompt_tokens == (1, 2, 3)


# ---------------------------------------------------------------------------
# build_items
# ---------------------------------------------------------------------------


def test_build_items_returns_n_items():
    items = longctx_retrieval.build_items(
        make_tokenizer(), CORPUS_TEXT, CORPUS_SHA256, n_items=3, seed=1, target_tokens=64,
    )
    assert len(items) == 3


def test_build_items_id_shape():
    (item,) = longctx_retrieval.build_items(
        make_tokenizer(), CORPUS_TEXT, CORPUS_SHA256, n_items=1, seed=7, target_tokens=64,
    )
    assert item.id == "longctx_retrieval-multivalue2-t64-s7-0000"
    assert item.suite == "longctx_retrieval"


def test_build_items_prompt_is_short_descriptor_not_document():
    (item,) = longctx_retrieval.build_items(
        make_tokenizer(), CORPUS_TEXT, CORPUS_SHA256, n_items=1, seed=7, target_tokens=64,
    )
    assert "corpusword" not in item.prompt
    assert len(item.prompt) < 200


def test_build_items_expected_is_match_strings_tuple():
    (item,) = longctx_retrieval.build_items(
        make_tokenizer(), CORPUS_TEXT, CORPUS_SHA256, n_items=1, seed=7, target_tokens=64,
    )
    assert isinstance(item.expected, tuple)
    assert len(item.expected) == 2  # multivalue2: two passcodes
    assert all(v.isdigit() and len(v) == 4 for v in item.expected)


def test_build_items_prompt_tokens_is_full_token_prompt():
    (item,) = longctx_retrieval.build_items(
        make_tokenizer(), CORPUS_TEXT, CORPUS_SHA256, n_items=1, seed=7, target_tokens=64,
    )
    assert isinstance(item.prompt_tokens, tuple)
    assert all(isinstance(t, int) for t in item.prompt_tokens)
    assert len(item.prompt_tokens) > 64  # document tokens + chat-wrapper tokens


def test_build_items_document_is_exactly_target_tokens():
    # The vendored generator's own guarantee: the document (needles + filler)
    # is exactly target_tokens tokens. Exercise the raw generator dict (which
    # exposes "document_tokens") the same way the adapter populates the
    # stream cache, to lock this exact-length property.
    tok = make_tokenizer()
    token_ids = tok(CORPUS_TEXT)["input_ids"]
    mv2._STREAM_CACHE[tok.name_or_path] = token_ids
    (raw,) = mv2.build_items(tok, "multivalue2", n_items=1, target_tokens=64, seed=7)
    assert raw["document_tokens"] == 64


def test_build_items_deterministic_across_two_calls():
    a = longctx_retrieval.build_items(
        make_tokenizer(), CORPUS_TEXT, CORPUS_SHA256, n_items=3, seed=42, target_tokens=64,
    )
    b = longctx_retrieval.build_items(
        make_tokenizer(), CORPUS_TEXT, CORPUS_SHA256, n_items=3, seed=42, target_tokens=64,
    )
    assert a == b


def test_build_items_different_seed_differs():
    a = longctx_retrieval.build_items(
        make_tokenizer(), CORPUS_TEXT, CORPUS_SHA256, n_items=3, seed=42, target_tokens=64,
    )
    b = longctx_retrieval.build_items(
        make_tokenizer(), CORPUS_TEXT, CORPUS_SHA256, n_items=3, seed=43, target_tokens=64,
    )
    assert a != b


def test_build_items_corpus_hash_mismatch_raises():
    with pytest.raises(AssertionError, match="sha256"):
        longctx_retrieval.build_items(
            make_tokenizer(), CORPUS_TEXT, "0" * 64, n_items=1, seed=1, target_tokens=64,
        )


# ---------------------------------------------------------------------------
# build_items_with_answer_spec — the P3 carried item (a) sidecar. EvalItem
# stays untouched; answer_ids/n_answer_tokens come back as an aligned
# AnswerSpec list instead.
# ---------------------------------------------------------------------------


def test_build_items_with_answer_spec_same_length_as_items():
    items, specs = longctx_retrieval.build_items_with_answer_spec(
        make_tokenizer(), CORPUS_TEXT, CORPUS_SHA256, n_items=5, seed=7, target_tokens=64,
    )
    assert len(items) == 5
    assert len(specs) == 5


def test_build_items_with_answer_spec_aligned_by_id_and_index():
    items, specs = longctx_retrieval.build_items_with_answer_spec(
        make_tokenizer(), CORPUS_TEXT, CORPUS_SHA256, n_items=5, seed=7, target_tokens=64,
    )
    for item, spec in zip(items, specs):
        assert item.id == spec.item_id


def test_build_items_with_answer_spec_answer_ids_match_raw_generator_output():
    tok = make_tokenizer()
    items, specs = longctx_retrieval.build_items_with_answer_spec(
        tok, CORPUS_TEXT, CORPUS_SHA256, n_items=3, seed=7, target_tokens=64,
    )
    # Independently rebuild the raw generator items (same seed/variant/
    # target_tokens) and confirm the sidecar's answer_ids/n_answer_tokens
    # match the vendored generator's own fields exactly -- not just "some"
    # tuple of ints.
    raw_items = mv2.build_items(tok, "multivalue2", n_items=3, target_tokens=64, seed=7)
    for spec, raw in zip(specs, raw_items):
        assert spec.answer_ids == tuple(raw["answer_ids"])
        assert spec.n_answer_tokens == raw["n_answer_tokens"]
        assert spec.n_answer_tokens == len(spec.answer_ids)


def test_build_items_with_answer_spec_prompt_tokens_plus_answer_ids_is_nll_input_ids():
    """The exact pairing nll_scorer.score_answer_span needs: EvalItem.
    prompt_tokens (gen_prompt_ids) + AnswerSpec.answer_ids reconstructs
    ANSWER_TOKENS.md's nll_input_ids without rebuilding anything."""
    items, specs = longctx_retrieval.build_items_with_answer_spec(
        make_tokenizer(), CORPUS_TEXT, CORPUS_SHA256, n_items=1, seed=7, target_tokens=64,
    )
    (item,), (spec,) = items, specs
    nll_input_ids = item.prompt_tokens + spec.answer_ids
    assert nll_input_ids[: len(item.prompt_tokens)] == item.prompt_tokens
    assert nll_input_ids[len(item.prompt_tokens):] == spec.answer_ids


def test_build_items_is_unchanged_by_the_answer_spec_refactor():
    """build_items (the pre-existing public API) must keep returning
    exactly what it always did -- build_items_with_answer_spec is an
    additive wrapper, not a behavior change."""
    a = longctx_retrieval.build_items(
        make_tokenizer(), CORPUS_TEXT, CORPUS_SHA256, n_items=3, seed=7, target_tokens=64,
    )
    items, _specs = longctx_retrieval.build_items_with_answer_spec(
        make_tokenizer(), CORPUS_TEXT, CORPUS_SHA256, n_items=3, seed=7, target_tokens=64,
    )
    assert a == items


# ---------------------------------------------------------------------------
# assert_tokenizer_match — the PREREG §3.1 build-time GGUF/HF tokenizer
# equivalence gate. Pure given two callables: an HF-style tokenizer and the
# llama-cpp model's tokenize function.
# ---------------------------------------------------------------------------


def _llama_tokenize_from(stub: StubTokenizer):
    """A llama-cpp-shaped tokenize callable (text -> list[int]) that agrees
    with `stub` exactly."""
    return lambda text: stub(text, add_special_tokens=False)["input_ids"]


def test_assert_tokenizer_match_passes_when_encodings_agree():
    hf = make_tokenizer()
    samples = [f"needle sentence number {i} with passcode {1000 + i}" for i in range(20)]
    longctx_retrieval.assert_tokenizer_match(hf, _llama_tokenize_from(hf), samples)
    # no raise = pass


def test_assert_tokenizer_match_raises_on_any_mismatch():
    hf = make_tokenizer()
    good = _llama_tokenize_from(hf)

    def drifting_tokenize(text):
        ids = good(text)
        # a GGUF-side divergence on one sample only (e.g. different merges)
        return ids + [999999] if "sample-7" in text else ids

    samples = [f"sample-{i} text" for i in range(20)]
    with pytest.raises(AssertionError, match="sample 7"):
        longctx_retrieval.assert_tokenizer_match(hf, drifting_tokenize, samples)


def test_assert_tokenizer_match_raises_on_empty_sample_list():
    hf = make_tokenizer()
    with pytest.raises(AssertionError, match="empty"):
        longctx_retrieval.assert_tokenizer_match(hf, _llama_tokenize_from(hf), [])
