"""Teacher-forced answer-span NLL scorer (0B pre-requisite P2).

This module implements PREREG.md §3.1's registered Q2 definition exactly:
the answer-span divergence metric is **teacher-forced negative log-
likelihood on the full-precision trajectory** -- the identical gold answer
token sequence (`nll_input_ids`, from
`bitcliff_pipeline.vendor.generate_multivalue2` / ANSWER_TOKENS.md) is fed
through F16 and every quant, so per-item deltas are paired on identical
inputs. This is confirmatory machinery, not a free-form metric: the tests in
`tests/test_nll_scorer.py` assert the registered alignment convention, the
full-span primary aggregation, the digits-only sensitivity, and identical-
input pairing -- not merely "it returns numbers" (per user ruling, RUN_0B.md
§2 P2 / §9.2).

Registered spec, restated here for reference (authoritative text lives in
PREREG.md §3.1 and the vendored bundle docs):

- `ANSWER_TOKENS.md` (construction): for each item,
  `nll_input_ids = gen_prompt_ids + answer_ids`; the scored span is the
  **contiguous suffix** `nll_input_ids[len(gen_prompt_ids):]`.
- `ANSWER_TOKENS.md` / PREREG §3.1 (alignment convention): the token at
  absolute position `p` is scored from the hidden state at `p-1`; logits for
  the first answer token come from the last prompt token. Right-padding
  only, so padding never precedes a scored position (moot at batch=1: no
  padding exists at all -- see `make_llama_logits_provider`).
- `GRADING.md` §2 (the paper's teacher-forced-NLL rule verbatim): one
  forward pass; item score is the mean cross-entropy over the answer-token
  positions; logits computed in float32 at the scored positions only; NaN
  NLL is an assertion failure, never skipped.
- PREREG §3.1 "Answer-token spec": primary aggregation is the mean over the
  **full answer span** (separators included, paper-consistent); the
  **digits-only mean is a registered sensitivity analysis**, explicitly
  labeled as departing from the paper's metric. `n_answer_tokens` is
  tokenizer-bound -- never hardcoded to 10 here; it is whatever
  `len(answer_ids)` is for the actual tokenizer in play.
"""

import dataclasses
import importlib.metadata
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# logits_provider(input_ids) -> one row of raw (pre-softmax) logits per
# input position, in position order, len(rows) == len(input_ids). This is
# exactly the shape llama-cpp-python's `Llama(logits_all=True)` produces
# (see `make_llama_logits_provider`), and is also trivial to fake in tests
# with hand-built rows.
LogitsProvider = Callable[[list[int]], list[list[float]]]


# ---------------------------------------------------------------------------
# Pure math: score_answer_span
# ---------------------------------------------------------------------------


def _log_softmax_at(row: list[float], token_id: int, *, position: int) -> float:
    """log P(token_id) under softmax(row), computed in float with the
    standard max-subtraction for numerical stability.

    Raises ValueError if `row` contains a NaN anywhere -- GRADING.md §2:
    "NaN NLL is an assertion failure, never skipped." Checked over the
    whole row, not just the queried token, because a NaN anywhere in the
    row poisons the softmax normalizer (the sum-of-exponentials) for every
    token computed from that row, not just the one at `token_id`. Raised
    unconditionally (not via `assert`) so it cannot be silently disabled by
    running Python with `-O`.
    """
    values = [float(x) for x in row]
    for x in values:
        if math.isnan(x):
            raise ValueError(
                f"NaN encountered in the logits row for scored position "
                f"{position} -- GRADING.md #2: NaN NLL is an assertion "
                f"failure, never skipped"
            )
    m = max(values)
    log_sum_exp = m + math.log(sum(math.exp(x - m) for x in values))
    return values[token_id] - log_sum_exp


def score_answer_span(
    logits_provider: LogitsProvider,
    nll_input_ids: list[int],
    n_prompt: int,
    answer_ids: list[int],
    digit_token_ids: frozenset[int] | None = None,
) -> dict:
    """Compute teacher-forced per-token NLL over the answer span of one item.

    Contract for `logits_provider`: given the full `nll_input_ids` sequence,
    return a list of exactly `len(nll_input_ids)` rows, one per input
    position in order; row `i` is the model's raw (pre-softmax) logits over
    the vocabulary computed with `nll_input_ids[0:i+1]` as context -- i.e.
    the distribution the model predicts *for the token that would follow
    position i*. This is precisely what llama-cpp-python's
    `Llama(logits_all=True)` exposes as `llm.eval_logits` after one
    `llm.eval(nll_input_ids)` call (see `make_llama_logits_provider`), and
    what a hand-built fake row list looks like in tests. Log-softmax is
    computed here, in this function, not by the provider -- the provider
    hands back raw logits only.

    Alignment (ANSWER_TOKENS.md, PREREG §3.1): answer token `k`
    (0-indexed within the answer span, absolute position `n_prompt + k` in
    `nll_input_ids`) is scored from the logits row at position
    `n_prompt + k - 1`. In particular the first answer token (`k=0`) is
    scored from the *last prompt token's* row (`n_prompt - 1`) -- logits
    for the first answer token come from the last prompt token, not from
    the first answer token's own row.

    Validates, before touching the provider:
    - `answer_ids` is non-empty and `n_prompt >= 1` (a state row must exist
      to score the first answer token from).
    - `len(nll_input_ids) == n_prompt + len(answer_ids)`.
    - `nll_input_ids[n_prompt:] == answer_ids` -- the scored span is the
      contiguous suffix (ANSWER_TOKENS.md's "Construction" section); this
      is what makes F16-vs-quant deltas paired on the identical gold
      sequence, per the user ruling this module implements.
    All of the above raise `ValueError` (structural/contract violations --
    a caller building `nll_input_ids` wrong, not a data/numerical problem).
    A NaN anywhere in a scored logits row also raises `ValueError` (see
    `_log_softmax_at`) -- both classes of failure use the same exception
    type so a caller can catch one thing, but each carries its own
    distinguishing message text.

    Returns a dict:
    - `per_token_nll`: list[float], one NLL per answer-span position, in
      answer order (length `n_answer_tokens`).
    - `span_mean_nll`: float, the **primary** metric -- mean over the FULL
      answer span, separators included (paper-consistent, PREREG §3.1).
    - `digits_only_mean_nll`: float | None -- the **registered sensitivity
      analysis**, mean over only the positions whose `answer_ids` token is
      in `digit_token_ids`. `None` when `digit_token_ids` is not supplied,
      or (defensively) when it is supplied but matches no position in this
      particular answer span.
    - `n_answer_tokens`: int, `len(answer_ids)` -- tokenizer-bound, never a
      hardcoded constant.
    """
    n_ans = len(answer_ids)
    if n_ans == 0:
        raise ValueError("answer_ids is empty -- nothing to score")
    if n_prompt < 1:
        raise ValueError(
            f"n_prompt={n_prompt}: need at least one prompt token so the first "
            f"answer token has a state position (n_prompt-1) to be scored from "
            f"(ANSWER_TOKENS.md: 'logits for the first answer token come from "
            f"the last prompt token')"
        )
    if len(nll_input_ids) != n_prompt + n_ans:
        raise ValueError(
            f"len(nll_input_ids)={len(nll_input_ids)} != n_prompt({n_prompt}) + "
            f"len(answer_ids)({n_ans}); nll_input_ids must be exactly "
            f"gen_prompt_ids + answer_ids (ANSWER_TOKENS.md 'Construction')"
        )
    suffix = list(nll_input_ids[n_prompt:])
    if suffix != list(answer_ids):
        raise ValueError(
            f"answer_ids {list(answer_ids)} does not match the suffix of "
            f"nll_input_ids at [{n_prompt}:] ({suffix}); the scored span must "
            f"be the contiguous suffix nll_input_ids[n_prompt:] "
            f"(ANSWER_TOKENS.md 'Construction')"
        )

    logits = logits_provider(list(nll_input_ids))
    if len(logits) != len(nll_input_ids):
        raise ValueError(
            f"logits_provider returned {len(logits)} rows for "
            f"{len(nll_input_ids)} input tokens -- the contract is exactly one "
            f"logits row per input position"
        )

    per_token_nll: list[float] = []
    for k in range(n_ans):
        pos = n_prompt + k
        state_pos = pos - 1
        token_id = nll_input_ids[pos]
        row = logits[state_pos]
        log_p = _log_softmax_at(row, token_id, position=state_pos)
        per_token_nll.append(-log_p)

    span_mean_nll = sum(per_token_nll) / n_ans

    digits_only_mean_nll = None
    if digit_token_ids is not None:
        digit_nlls = [
            nll for tok, nll in zip(answer_ids, per_token_nll) if tok in digit_token_ids
        ]
        if digit_nlls:
            digits_only_mean_nll = sum(digit_nlls) / len(digit_nlls)

    return {
        "per_token_nll": per_token_nll,
        "span_mean_nll": span_mean_nll,
        "digits_only_mean_nll": digits_only_mean_nll,
        "n_answer_tokens": n_ans,
    }


def digit_token_ids_from_decode(
    decode: Callable[[list[int]], str], vocab_size: int
) -> frozenset[int]:
    """Generic, tokenizer-agnostic derivation of the "digit token" set: a
    token id belongs iff decoding it alone yields exactly one ASCII digit
    character ('0'-'9'). Works for any tokenizer with single-digit tokens
    (Qwen2.5 digitizes per character, per ANSWER_TOKENS.md's worked
    example); does not hardcode any particular vocab size, id list, or
    count -- `n_answer_tokens` and the number of digit positions are always
    whatever the actual tokenizer in play produces (the tokenizer-bound
    comparability rule, PREREG §3.1).

    `decode` is called once per vocabulary id with a single-token list
    (`decode([token_id]) -> str`), matching both HF (`tokenizer.decode`)
    and llama-cpp (`llm.detokenize([token_id]).decode(...)`) shapes once
    wrapped by the caller.
    """
    ids = set()
    for token_id in range(vocab_size):
        text = decode([token_id])
        if len(text) == 1 and text in "0123456789":
            ids.add(token_id)
    return frozenset(ids)


# ---------------------------------------------------------------------------
# llama-cpp-python 0.3.35 integration
# ---------------------------------------------------------------------------
#
# API investigated directly against the pinned 0.3.35 sdist
# (.venv/lib/python3.11/site-packages/llama_cpp/llama.py):
#
#   - Llama(..., logits_all=True) sizes `self.scores` to
#     (n_ctx, n_vocab) instead of (n_batch, n_vocab); without it llama.cpp
#     only ever computes/stores logits for the LAST position of a batch, and
#     every earlier row is stale/undefined -- required for a full-sequence
#     per-position pass.
#   - `llm.reset()` clears `n_tokens` (and any recurrent/hybrid KV state);
#     required so successive items don't accumulate onto each other's KV
#     cache within one process.
#   - `llm.eval(tokens)` runs one or more internal batches (chunked by
#     `n_batch`, transparently, still logically "one forward pass" over the
#     full sequence) and writes every position's logits into `self.scores`
#     when `logits_all=True` (`Llama.eval`, lines ~666-699).
#   - `llm.eval_logits` (a `Deque[List[float]]` property) returns exactly
#     `self.scores[:n_tokens, :].tolist()` -- one raw logits row per
#     position, in position order, length == len(tokens) after a fresh
#     `reset()` + `eval()` pair. This is precisely the `LogitsProvider`
#     contract `score_answer_span` expects.
#
# Single-sequence, batch=1: llama-cpp-python's `eval()` never pads within a
# single call -- padding only becomes relevant when batching multiple
# sequences together, which this module does not do (score_records issues
# one `reset()`+`eval()` pair per item). Per PREREG's answer-token spec
# ("right-padding only, so padding never precedes a scored position") that
# rule is trivially satisfied here: with no batching, no padding exists at
# all.
#
# The lower-level alternative considered and rejected: `create_completion(
# ..., logprobs=N, echo=True)`. It requires `logits_all=True` too (raises
# otherwise, line ~1365), returns *log-probabilities* rather than raw
# logits (folding log-softmax into the C-level/completion path, which this
# module deliberately keeps in Python -- `_log_softmax_at` above -- so it
# stays independently testable/auditable against the registered spec), and
# is shaped around sampling/generation bookkeeping (finish_reason, top-k
# alternatives, etc.) this module has no use for. `llm.eval()` +
# `llm.eval_logits` is the thinnest correct wrapper for this need.


def make_llama_logits_provider(llm) -> LogitsProvider:
    """Wrap a llama-cpp-python `Llama` instance into the `LogitsProvider`
    contract `score_answer_span` expects.

    Requires `llm` to have been constructed with `Llama(..., logits_all=
    True)` -- without it, `llm.eval_logits` only carries the last
    position's logits and every earlier row is meaningless (see the module
    comment above). Each call is one `llm.reset()` (drop any prior
    sequence's KV state) followed by one `llm.eval(input_ids)` (a single
    forward pass over the full sequence, batch=1, no padding) and reads
    back `llm.eval_logits`.
    """

    def provider(input_ids: list[int]) -> list[list[float]]:
        llm.reset()
        llm.eval(list(input_ids))
        n = len(input_ids)
        # Prefer the (n_ctx, n_vocab) numpy `scores` array the real
        # llama-cpp-python object exposes under logits_all=True: a zero-copy
        # row view with identical row semantics to eval_logits (which is
        # built FROM scores). Materializing eval_logits copies every
        # position's row into Python float objects — ~45 GB for a 152k-vocab
        # model over an 8.3k sequence — which OOM-killed the Qwen NLL pass
        # (dmesg 2026-09-03 18:43). score_answer_span only ever reads the
        # ~n_answer+1 rows it scores from.
        scores = getattr(llm, "scores", None)
        if scores is not None:
            return scores[:n]
        return [list(row) for row in llm.eval_logits]

    return provider


# ---------------------------------------------------------------------------
# Per-item records, matching generate.py's OutputRecord / write_records /
# read_records patterns.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NLLItem:
    """One item to teacher-force-score: the gen prompt (same token ids the
    generator used to produce the model's free-running completion) and the
    gold answer token ids (ANSWER_TOKENS.md's `answer_ids`), kept separate
    -- `score_records` builds `nll_input_ids = gen_prompt_ids + answer_ids`
    itself, exactly as ANSWER_TOKENS.md's "Construction" section specifies.
    """

    id: str
    suite: str
    gen_prompt_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]


@dataclass(frozen=True)
class NLLRecord:
    """One item's scored record: id, rung label, model sha, the three NLL
    fields (`score_answer_span`'s primary + sensitivity + per-token detail),
    `n_answer_tokens` (tokenizer-bound), and the machine/library fingerprint
    already carried by every other pipeline output record (`generate.py`'s
    `OutputRecord.machine`; PREREG §6: "every output record carries the
    machine and library fingerprint"). There is no `gen_settings` field --
    teacher forcing has no sampling settings (no temperature/top_k/seed);
    determinism here comes from the fixed gold token sequence, not from
    decode parameters.
    """

    item_id: str
    suite: str
    quant_label: str
    model_sha256: str
    per_token_nll: list[float]
    span_mean_nll: float
    digits_only_mean_nll: float | None
    n_answer_tokens: int
    machine: str


def _machine_fingerprint() -> str:
    """Same fingerprint shape as `generate.py`'s `run_items` (platform +
    arch + llama-cpp-python version) -- duplicated here rather than
    imported so this module has no import-time dependency on `generate.py`
    beyond what it already needs; the two independently match PREREG §6's
    requirement, not because one calls the other.
    """
    try:
        llama_cpp_version = importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        llama_cpp_version = "unknown"
    return (
        f"{platform.platform()} / {platform.machine()} "
        f"/ llama-cpp-python {llama_cpp_version}"
    )


def score_records(
    llm,
    items: list[NLLItem],
    quant_label: str,
    model_sha256: str,
    digit_token_ids: frozenset[int] | None = None,
) -> list[NLLRecord]:
    """Score every item in `items` by teacher-forcing `llm` over
    `gen_prompt_ids + answer_ids` (dependency injection like `generate.py`'s
    `llm_factory`/`run_items`: `llm` is any object exposing
    `reset()`/`eval(tokens)`/`eval_logits`, real llama-cpp-python instance
    or a test fake -- see `tests/test_nll_scorer.py`).

    `digit_token_ids` is passed straight through to `score_answer_span` for
    every item; the caller derives it once per model/tokenizer (e.g. via
    `digit_token_ids_from_decode`) and reuses it across every item and
    every rung (the token-id -> digit-ness mapping is a tokenizer property,
    not a per-item one).
    """
    provider = make_llama_logits_provider(llm)
    machine = _machine_fingerprint()
    records = []
    for item in items:
        gen_prompt_ids = list(item.gen_prompt_ids)
        answer_ids = list(item.answer_ids)
        nll_input_ids = gen_prompt_ids + answer_ids
        result = score_answer_span(
            provider,
            nll_input_ids,
            n_prompt=len(gen_prompt_ids),
            answer_ids=answer_ids,
            digit_token_ids=digit_token_ids,
        )
        records.append(
            NLLRecord(
                item_id=item.id,
                suite=item.suite,
                quant_label=quant_label,
                model_sha256=model_sha256,
                per_token_nll=result["per_token_nll"],
                span_mean_nll=result["span_mean_nll"],
                digits_only_mean_nll=result["digits_only_mean_nll"],
                n_answer_tokens=result["n_answer_tokens"],
                machine=machine,
            )
        )
    return records


def write_records(records: list[NLLRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(dataclasses.asdict(r)) + "\n")


def read_records(path: Path) -> list[NLLRecord]:
    return [NLLRecord(**json.loads(line)) for line in path.read_text().splitlines()]
