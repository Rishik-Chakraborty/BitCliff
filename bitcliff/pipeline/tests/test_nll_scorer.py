"""Tests for the teacher-forced answer-span NLL scorer (P2).

No network, no real model: fake `logits_provider`/`llm` objects with KNOWN,
hand-computable logit distributions. Every expected value below is derived
independently from softmax-by-hand (raw logits -> exp -> normalize -> -log),
never by calling the module's own `_log_softmax_at`/`score_answer_span` --
so these tests would catch a wrong implementation, not just echo it.

Per the user ruling (RUN_0B.md §2 P2): this module implements PREREG §3.1's
registered Q2 definition (teacher-forced NLL on the full-precision
trajectory) exactly, and these tests assert that -- the alignment
convention, the full-span primary, the digits-only sensitivity, and
identical-input pairing -- not merely "it returns numbers."
"""

import math

import pytest

from bitcliff_pipeline.nll_scorer import (
    NLLItem,
    digit_token_ids_from_decode,
    make_llama_logits_provider,
    read_records,
    score_answer_span,
    score_records,
    write_records,
)


# ---------------------------------------------------------------------------
# Shared fake logits_provider: position -> row, by absolute position index
# in the FULL nll_input_ids sequence. Positions not explicitly given fall
# back to an all-zero ("uniform") row. This satisfies the LogitsProvider
# contract exactly: given input_ids, return one row per position, in
# position order.
# ---------------------------------------------------------------------------


def make_fake_provider(rows_by_position: dict, vocab_size: int):
    default_row = [0.0] * vocab_size

    def provider(input_ids):
        return [rows_by_position.get(i, list(default_row)) for i in range(len(input_ids))]

    return provider


# ---------------------------------------------------------------------------
# (1) Alignment: the first (and every) answer token is scored from the
# LAST PROMPT TOKEN's state (n_prompt - 1), not from its own position or
# any other off-by-one. The provider gives DIFFERENT, hand-computable rows
# at position (n_prompt - 1) [correct], (n_prompt) [an "off by one, uses
# its own position" bug], and (n_prompt - 2) [an "off by two" bug] -- so a
# misaligned implementation would produce a distinguishably different
# number than the one asserted here.
# ---------------------------------------------------------------------------


def test_alignment_first_answer_token_scored_from_last_prompt_token():
    vocab = 6
    n_prompt = 3
    answer_ids = [3]  # single answer token, id 3
    nll_input_ids = [100, 101, 102, 3]

    # correct state row: position n_prompt - 1 == 2.
    # logits = [0, 0, 0, ln7, 0, 0] -> exp = [1,1,1,7,1,1], sum=12
    # p(token 3) = 7/12  =>  NLL = -ln(7/12) = ln(12/7)
    row_correct = [0.0, 0.0, 0.0, math.log(7), 0.0, 0.0]

    # "off by one" bug row (would-be-used position n_prompt == 3): boosts a
    # DIFFERENT token (5), so token 3's probability there is the pure
    # background value 1/14, giving ln(14) -- clearly distinguishable from
    # ln(12/7) above.
    row_off_by_one = [0.0, 0.0, 0.0, 0.0, 0.0, math.log(9)]

    # "off by two" bug row (would-be-used position n_prompt - 2 == 1):
    # uniform, background p(token 3) = 1/6 -> NLL = ln(6). Also
    # distinguishable.
    row_off_by_two = [0.0] * vocab

    provider = make_fake_provider(
        {2: row_correct, 3: row_off_by_one, 1: row_off_by_two}, vocab
    )

    result = score_answer_span(provider, nll_input_ids, n_prompt, answer_ids)

    expected_correct = math.log(12 / 7)
    expected_off_by_one = math.log(14)
    expected_off_by_two = math.log(6)
    # sanity: the three hand-computed candidates really are distinguishable
    assert len({round(expected_correct, 6), round(expected_off_by_one, 6),
                round(expected_off_by_two, 6)}) == 3

    assert result["per_token_nll"] == pytest.approx([expected_correct])
    assert result["span_mean_nll"] == pytest.approx(expected_correct)
    assert result["n_answer_tokens"] == 1


# ---------------------------------------------------------------------------
# (2) + (3): full-span primary includes separator positions (hand-computed
# mean over ALL positions); digits-only mean excludes exactly the
# non-digit positions.
#
# 4-token answer span mimicking ANSWER_TOKENS.md's digit/separator
# structure: [digit, separator, digit, digit]. digit_token_ids = {2,3,4};
# separator token id = 0.
# ---------------------------------------------------------------------------


VOCAB = 6
N_PROMPT = 3
ANSWER_IDS = [2, 0, 3, 4]  # digit, sep, digit, digit
DIGIT_IDS = frozenset({1, 2, 3, 4})
NLL_INPUT_IDS = [10, 11, 12] + ANSWER_IDS  # 7 tokens total, positions 0..6

# state rows at positions n_prompt-1=2 .. n_prompt+2=5 (one per answer token)
ROW_K0 = [0.0, 0.0, math.log(5), 0.0, 0.0, 0.0]  # correct token 2: p=5/10 -> NLL=ln2
ROW_K1 = [math.log(3), 0.0, 0.0, 0.0, 0.0, 0.0]  # correct token 0: p=3/8 -> NLL=ln(8/3)
ROW_K2 = [0.0, 0.0, 0.0, math.log(9), 0.0, 0.0]  # correct token 3: p=9/14 -> NLL=ln(14/9)
ROW_K3 = [0.0, 0.0, 0.0, 0.0, math.log(4), 0.0]  # correct token 4: p=4/9 -> NLL=ln(9/4)

ROWS_BY_POSITION = {2: ROW_K0, 3: ROW_K1, 4: ROW_K2, 5: ROW_K3}

NLL0 = math.log(2)
NLL1 = math.log(8 / 3)
NLL2 = math.log(14 / 9)
NLL3 = math.log(9 / 4)


def test_full_span_primary_includes_separator_positions():
    provider = make_fake_provider(ROWS_BY_POSITION, VOCAB)
    result = score_answer_span(provider, NLL_INPUT_IDS, N_PROMPT, ANSWER_IDS)

    assert result["per_token_nll"] == pytest.approx([NLL0, NLL1, NLL2, NLL3])
    # hand-computed mean over ALL 4 positions, separator (k=1) included
    expected_full_span_mean = (NLL0 + NLL1 + NLL2 + NLL3) / 4
    assert result["span_mean_nll"] == pytest.approx(expected_full_span_mean)
    assert result["n_answer_tokens"] == 4


def test_digits_only_mean_excludes_exactly_the_non_digit_positions():
    provider = make_fake_provider(ROWS_BY_POSITION, VOCAB)
    result = score_answer_span(
        provider, NLL_INPUT_IDS, N_PROMPT, ANSWER_IDS, digit_token_ids=DIGIT_IDS
    )

    # hand-computed mean over positions k=0,2,3 only (token ids 2,3,4 are
    # digits); k=1 (token id 0, the separator) is excluded.
    expected_digits_only_mean = (NLL0 + NLL2 + NLL3) / 3
    assert result["digits_only_mean_nll"] == pytest.approx(expected_digits_only_mean)
    # and the full-span primary is unaffected by supplying digit_token_ids
    expected_full_span_mean = (NLL0 + NLL1 + NLL2 + NLL3) / 4
    assert result["span_mean_nll"] == pytest.approx(expected_full_span_mean)


def test_digits_only_mean_is_none_when_digit_ids_not_supplied():
    provider = make_fake_provider(ROWS_BY_POSITION, VOCAB)
    result = score_answer_span(provider, NLL_INPUT_IDS, N_PROMPT, ANSWER_IDS)
    assert result["digits_only_mean_nll"] is None


def test_digits_only_mean_is_none_when_no_answer_token_is_a_digit():
    provider = make_fake_provider(ROWS_BY_POSITION, VOCAB)
    result = score_answer_span(
        provider, NLL_INPUT_IDS, N_PROMPT, ANSWER_IDS, digit_token_ids=frozenset({99})
    )
    assert result["digits_only_mean_nll"] is None


# ---------------------------------------------------------------------------
# (4) NaN in logits raises.
# ---------------------------------------------------------------------------


def test_nan_in_scored_logits_row_raises():
    bad_row = [0.0, float("nan"), 0.0, 0.0, 0.0, 0.0]
    provider = make_fake_provider({2: bad_row}, VOCAB)
    with pytest.raises(ValueError, match="NaN"):
        score_answer_span(provider, [10, 11, 12, 2], N_PROMPT, [2])


def test_nan_elsewhere_in_the_row_still_raises_even_if_scored_token_is_finite():
    # NaN at an unrelated vocab index in the SAME scored row still poisons
    # the softmax normalizer -- must raise even though logits[token_id]
    # itself is a normal float.
    bad_row = [0.0, 0.0, float("nan"), 0.0, 0.0, 0.0]
    provider = make_fake_provider({2: bad_row}, VOCAB)
    with pytest.raises(ValueError, match="NaN"):
        score_answer_span(provider, [10, 11, 12, 0], N_PROMPT, [0])


# ---------------------------------------------------------------------------
# (5) answer_ids-suffix mismatch raises.
# ---------------------------------------------------------------------------


def test_answer_ids_suffix_mismatch_raises():
    provider = make_fake_provider({}, VOCAB)
    nll_input_ids = [10, 11, 12, 999]  # suffix is [999]
    with pytest.raises(ValueError, match="suffix"):
        score_answer_span(provider, nll_input_ids, N_PROMPT, [3])


def test_length_mismatch_between_nll_input_ids_and_n_prompt_plus_answer_raises():
    provider = make_fake_provider({}, VOCAB)
    nll_input_ids = [10, 11, 12, 3, 4]  # 5 tokens, but n_prompt+len(answer)=4
    with pytest.raises(ValueError, match="len\\(nll_input_ids\\)"):
        score_answer_span(provider, nll_input_ids, N_PROMPT, [3])


def test_empty_answer_ids_raises():
    provider = make_fake_provider({}, VOCAB)
    with pytest.raises(ValueError, match="empty"):
        score_answer_span(provider, [10, 11, 12], N_PROMPT, [])


def test_zero_prompt_length_raises():
    provider = make_fake_provider({}, VOCAB)
    with pytest.raises(ValueError, match="n_prompt"):
        score_answer_span(provider, [3], 0, [3])


def test_logits_provider_row_count_mismatch_raises():
    def short_provider(input_ids):
        return [[0.0] * VOCAB for _ in range(len(input_ids) - 1)]

    with pytest.raises(ValueError, match="rows"):
        score_answer_span(short_provider, [10, 11, 12, 3], N_PROMPT, [3])


# ---------------------------------------------------------------------------
# (6) Identical inputs -> identical outputs (determinism / pairing). This is
# the property that makes F16-vs-quant deltas paired on the same gold
# sequence, per the user ruling this module implements.
# ---------------------------------------------------------------------------


def test_identical_inputs_produce_identical_outputs():
    provider = make_fake_provider(ROWS_BY_POSITION, VOCAB)
    r1 = score_answer_span(
        provider, NLL_INPUT_IDS, N_PROMPT, ANSWER_IDS, digit_token_ids=DIGIT_IDS
    )
    r2 = score_answer_span(
        provider, NLL_INPUT_IDS, N_PROMPT, ANSWER_IDS, digit_token_ids=DIGIT_IDS
    )
    assert r1 == r2


def test_score_records_is_deterministic_across_two_calls_on_identical_items():
    class FakeLlm:
        def __init__(self):
            self._n = 0

        def reset(self):
            pass

        def eval(self, tokens):
            self._n = len(tokens)

        @property
        def eval_logits(self):
            default = [0.0] * VOCAB
            return [ROWS_BY_POSITION.get(i, list(default)) for i in range(self._n)]

    items = [NLLItem("id-1", "longctx_retrieval", (10, 11, 12), tuple(ANSWER_IDS))]
    r1 = score_records(FakeLlm(), items, "Q4_K_M", "sha-abc", digit_token_ids=DIGIT_IDS)
    r2 = score_records(FakeLlm(), items, "Q4_K_M", "sha-abc", digit_token_ids=DIGIT_IDS)
    assert r1 == r2
    assert r1[0].span_mean_nll == pytest.approx((NLL0 + NLL1 + NLL2 + NLL3) / 4)
    assert r1[0].digits_only_mean_nll == pytest.approx((NLL0 + NLL2 + NLL3) / 3)
    assert r1[0].n_answer_tokens == 4


# ---------------------------------------------------------------------------
# (7) JSONL roundtrip.
# ---------------------------------------------------------------------------


def test_jsonl_roundtrip(tmp_path):
    class FakeLlm:
        def __init__(self):
            self._n = 0

        def reset(self):
            pass

        def eval(self, tokens):
            self._n = len(tokens)

        @property
        def eval_logits(self):
            default = [0.0] * VOCAB
            return [ROWS_BY_POSITION.get(i, list(default)) for i in range(self._n)]

    items = [
        NLLItem("id-1", "longctx_retrieval", (10, 11, 12), tuple(ANSWER_IDS)),
        NLLItem("id-2", "longctx_retrieval", (20, 21, 22), tuple(ANSWER_IDS)),
    ]
    records = score_records(FakeLlm(), items, "Q2_K", "sha-def", digit_token_ids=DIGIT_IDS)
    path = tmp_path / "Q2_K.nll.jsonl"
    write_records(records, path)
    loaded = read_records(path)
    assert loaded == records
    assert len(loaded) == 2
    assert loaded[0].item_id == "id-1"
    assert loaded[0].quant_label == "Q2_K"
    assert loaded[0].model_sha256 == "sha-def"
    assert loaded[0].machine  # non-empty fingerprint string
    assert "llama-cpp-python" in loaded[0].machine


# ---------------------------------------------------------------------------
# make_llama_logits_provider: wiring against a fake llama-cpp-shaped object
# (reset/eval/eval_logits), confirming the exact call sequence used.
# ---------------------------------------------------------------------------


def test_make_llama_logits_provider_calls_reset_then_eval_then_reads_eval_logits():
    class RecordingFakeLlm:
        def __init__(self):
            self.calls = []
            self._tokens = []

        def reset(self):
            self.calls.append("reset")

        def eval(self, tokens):
            self.calls.append(("eval", list(tokens)))
            self._tokens = list(tokens)

        @property
        def eval_logits(self):
            self.calls.append("read_eval_logits")
            return [[float(t)] for t in self._tokens]

    llm = RecordingFakeLlm()
    provider = make_llama_logits_provider(llm)
    rows = provider([7, 8, 9])

    assert llm.calls == ["reset", ("eval", [7, 8, 9]), "read_eval_logits"]
    assert rows == [[7.0], [8.0], [9.0]]


def test_make_llama_logits_provider_resets_between_successive_items():
    class RecordingFakeLlm:
        def __init__(self):
            self.reset_count = 0
            self._tokens = []

        def reset(self):
            self.reset_count += 1

        def eval(self, tokens):
            self._tokens = list(tokens)

        @property
        def eval_logits(self):
            return [[0.0] for _ in self._tokens]

    llm = RecordingFakeLlm()
    provider = make_llama_logits_provider(llm)
    provider([1, 2])
    provider([3, 4, 5])
    assert llm.reset_count == 2


# ---------------------------------------------------------------------------
# digit_token_ids_from_decode: generic, tokenizer-agnostic derivation.
# ---------------------------------------------------------------------------


def test_digit_token_ids_from_decode_picks_only_single_ascii_digit_tokens():
    # a tiny fake vocab: ids 0-9 decode to digits '0'-'9', id 10 decodes to
    # a multi-character piece, id 11 decodes to a non-digit character.
    table = {i: str(i) for i in range(10)}
    table[10] = ", "
    table[11] = "x"

    def decode(ids):
        (tid,) = ids
        return table[tid]

    ids = digit_token_ids_from_decode(decode, vocab_size=12)
    assert ids == frozenset(range(10))


def test_digit_token_ids_from_decode_empty_vocab_gives_empty_set():
    ids = digit_token_ids_from_decode(lambda ids: "z", vocab_size=0)
    assert ids == frozenset()
