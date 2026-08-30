"""longctx_retrieval: needle-in-a-haystack retrieval suite (multivalue2 et al.).

Thin adapter around the vendored `generate_multivalue2` generator
(`bitcliff_pipeline.vendor.generate_multivalue2`, extracted verbatim from the
capability-targeted-quantization paper's source project — see that module's
docstring for provenance). This adapter:

- Takes an explicit `corpus_text` + `corpus_sha256` rather than going through
  the vendored generator's own corpus loader (which downloads from HF and
  checks against the *paper's* hardcoded corpus hash). Instead, this module
  verifies `corpus_text` against the caller-supplied `corpus_sha256` itself,
  then pre-populates (overwrites) the vendored generator's internal
  `_STREAM_CACHE` directly, so `_token_stream` never consults
  `load_corpus`/`CORPUS_SHA256` at all. This lets the same vendored,
  logic-unchanged generator run against any corpus a caller has already
  verified (the real Paul Graham corpus in production, a tiny synthetic
  corpus in tests) without duplicating or modifying its item-construction
  logic.
- Converts the vendored generator's raw item dicts into bitcliff_pipeline
  `EvalItem`s: `prompt` holds a short human-readable descriptor (the
  question text) — never the document, which lives only in `prompt_tokens`
  as the full token-id prompt; `expected` is the `match_strings` tuple.
- Implements the paper's grading rule verbatim (see GRADING.md in the
  multivalue2-bundle): boolean exact match, unanchored substring, order-
  insensitive, no partial credit.
"""

import hashlib
from dataclasses import dataclass

from ..items import EvalItem
from ..vendor import generate_multivalue2 as mv2


@dataclass(frozen=True)
class AnswerSpec:
    """0B P3 carried item (RUN_0B.md §2 P3(a) / P2 report's documented
    gap): the answer-token spec the vendored generator already computes per
    item (`answer_ids`, `n_answer_tokens` — see ANSWER_TOKENS.md), threaded
    through `build_items_with_answer_spec` as a SIDECAR aligned 1:1 with
    the EvalItem list -- same order, same count, `item_id == EvalItem.id`
    at the same index. `EvalItem` deliberately stays untouched (it has no
    concept of an "answer span"; that's a longctx_retrieval-specific,
    tokenizer-bound construction other suites don't share), so this is a
    parallel list rather than an EvalItem field.

    Consumer (P2, `nll_scorer.py`): `nll_scorer.NLLItem(id=spec.item_id,
    suite="longctx_retrieval", gen_prompt_ids=item.prompt_tokens,
    answer_ids=spec.answer_ids)`, pairing an `AnswerSpec` with the
    same-index `EvalItem` from `build_items_with_answer_spec`'s other
    return value -- no tokenizer, corpus, or vendored generator call
    needed a second time; a run's already-built `items.jsonl` (which
    carries `prompt_tokens`, i.e. `gen_prompt_ids`) plus this sidecar is
    everything `score_answer_span` requires.
    """

    item_id: str
    answer_ids: tuple[int, ...]
    n_answer_tokens: int


def build_items_with_answer_spec(
    tokenizer,
    corpus_text: str,
    corpus_sha256: str,
    n_items: int,
    seed: int,
    variant: str = "multivalue2",
    target_tokens: int = 4096,
) -> tuple[list[EvalItem], list[AnswerSpec]]:
    """Build longctx_retrieval EvalItems via the vendored multivalue2
    generator, alongside an aligned `AnswerSpec` sidecar list (see
    `AnswerSpec`'s docstring for why it isn't just added to `EvalItem`).

    `tokenizer` needs exactly the surface the vendored generator uses: a
    callable `tokenizer(text, add_special_tokens=False)["input_ids"]`,
    `tokenizer.decode(ids)`, `tokenizer.apply_chat_template(messages,
    tokenize=False, add_generation_prompt=True)`, and `tokenizer.name_or_path`.

    Raises AssertionError if `corpus_text` does not hash to `corpus_sha256`
    (a different corpus silently changes every document).

    Alignment guarantee (tested in test_longctx_retrieval.py): the returned
    `(items, answer_specs)` have equal length, and for every index `i`,
    `items[i].id == answer_specs[i].item_id`.
    """
    digest = hashlib.sha256(corpus_text.encode("utf-8")).hexdigest()
    assert digest == corpus_sha256, (
        f"corpus sha256 is {digest}, expected {corpus_sha256}: a different "
        f"corpus produces different documents; items would not be comparable "
        f"to a fixed reference build"
    )

    token_ids = tokenizer(corpus_text, add_special_tokens=False)["input_ids"]
    # Overwrite (not merely pre-populate) so a stale entry from a prior
    # corpus under the same tokenizer name_or_path can never leak in.
    mv2._STREAM_CACHE[tokenizer.name_or_path] = token_ids

    raw_items = mv2.build_items(tokenizer, variant, n_items, target_tokens, seed)

    items = []
    answer_specs = []
    for it in raw_items:
        item_id = f"longctx_retrieval-{variant}-t{target_tokens}-s{seed}-{it['index']:04d}"
        items.append(
            EvalItem(
                id=item_id,
                suite="longctx_retrieval",
                prompt=it["question"],
                expected=tuple(it["match_strings"]),
                prompt_tokens=tuple(it["gen_prompt_ids"]),
            )
        )
        answer_specs.append(
            AnswerSpec(
                item_id=item_id,
                answer_ids=tuple(it["answer_ids"]),
                n_answer_tokens=it["n_answer_tokens"],
            )
        )
    return items, answer_specs


def build_items(
    tokenizer,
    corpus_text: str,
    corpus_sha256: str,
    n_items: int,
    seed: int,
    variant: str = "multivalue2",
    target_tokens: int = 4096,
) -> list[EvalItem]:
    """Build longctx_retrieval EvalItems via the vendored multivalue2
    generator. Back-compat wrapper around `build_items_with_answer_spec`
    that drops the answer-spec sidecar; every existing caller (and every
    existing test) keeps this exact signature and return type."""
    items, _ = build_items_with_answer_spec(
        tokenizer, corpus_text, corpus_sha256, n_items, seed, variant, target_tokens,
    )
    return items


def assert_tokenizer_match(hf_tokenizer, llama_tokenize, samples: list[str]) -> None:
    """PREREG §3.1 tokenizer-equivalence gate: assert the HF tokenizer and
    the llama-cpp (GGUF) tokenizer produce identical token ids for every
    sample string; raise AssertionError on the first mismatch (or on an
    empty sample list, which would make the gate vacuous).

    `hf_tokenizer` is HF-style: `hf_tokenizer(text, add_special_tokens=
    False)["input_ids"]`. `llama_tokenize` is the llama-cpp model's tokenize
    as a plain callable `text -> list[int]` — e.g.
    `lambda s: llm.tokenize(s.encode("utf-8"), add_bos=False)` — with
    `add_bos`/special-token handling configured consistently with the HF
    side (`add_special_tokens=False` here, so no BOS on either side). The
    helper is pure given the two callables; per PREREG §3.1 it runs before
    each confirmatory generation run, over the registered 20-string sample
    (the question strings of the run's first 20 items, in id order).
    """
    assert samples, "assert_tokenizer_match called with an empty sample list"
    for i, text in enumerate(samples):
        hf_ids = list(hf_tokenizer(text, add_special_tokens=False)["input_ids"])
        llama_ids = list(llama_tokenize(text))
        assert hf_ids == llama_ids, (
            f"tokenizer mismatch on sample {i}: HF ids {hf_ids} != llama-cpp "
            f"ids {llama_ids} for text {text!r} — the GGUF and HF tokenizers "
            f"disagree; token-space items would not be comparable"
        )


def llama_tokenize_callable(llm):
    """Wrap a llama-cpp model's `.tokenize` into the plain `text ->
    list[int]` callable `assert_tokenizer_match` expects, with BOS/
    special-token handling fixed to match the HF side's
    `add_special_tokens=False` convention used above (no BOS, no special
    tokens injected on either side — required for the two encodings to be
    comparable at all).

    This is the exact glue `scripts/calibrate_f16.py`'s tokenizer gate
    uses (`LongctxRunner._check_tokenizer_once`); extracted here so that
    caller and `bitcliff_pipeline.__main__.run_pipeline`'s confirmatory
    gate share one implementation instead of two divergent copies.
    """
    return lambda s: llm.tokenize(s.encode("utf-8"), add_bos=False, special=False)


def grade(item: EvalItem, text: str) -> str:
    """Paper's verbatim grading rule (GRADING.md §1): an item is correct iff
    every gold match string appears as a plain, unanchored, order-
    insensitive Python substring of `text`. No partial credit, no grader
    model: a match is a match.
    """
    return "correct" if all(s in text for s in item.expected) else "wrong"
