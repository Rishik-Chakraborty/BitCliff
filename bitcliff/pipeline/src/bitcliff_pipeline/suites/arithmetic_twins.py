"""arithmetic_twins suite (PREREG §3.3): all 47 original+twin pairs in-suite.

Thin runnable path over the twins machinery: for every verified template, the
suite evaluates BOTH the real GSM8K original (the templated item itself) and
its deterministic seed-built twin, under identical settings. Prompts carry
the same registered answer instruction as the `arithmetic` suite (§3.2) and
grading reuses `arithmetic.grade` verbatim (wired in `grading.GRADERS`) — the
contamination comparison is within-pair, so both members must be prompted and
graded identically.
"""

from ..items import EvalItem
from ..twins.builder import build_twin
from ..twins.templates import TEMPLATES
from ..twins.verifier import verify_template
from .arithmetic import ANSWER_INSTRUCTION, extract_final_number


def build_pair_items(templates, gsm8k_records, seed: int) -> list[EvalItem]:
    """One (original, twin) item pair per template, in template order.

    Every template is first round-trip verified against its real GSM8K
    record (same integrity contract as `twins.builder.build_twin_set`), so
    the originals evaluated here are provably the exact GSM8K text. Ids:
    `arithmetic_twins-{seed}-orig-{gsm8k_index:03d}` and `-twin-`.
    """
    items: list[EvalItem] = []
    for t in templates:
        record = gsm8k_records[t.gsm8k_index]
        verify_template(t, record["question"], record["answer"])
        twin = build_twin(t, seed)
        items.append(
            EvalItem(
                id=f"arithmetic_twins-{seed}-orig-{t.gsm8k_index:03d}",
                suite="arithmetic_twins",
                prompt=record["question"] + ANSWER_INSTRUCTION,
                expected=(extract_final_number(record["answer"]),),
            )
        )
        items.append(
            EvalItem(
                id=f"arithmetic_twins-{seed}-twin-{t.gsm8k_index:03d}",
                suite="arithmetic_twins",
                prompt=twin["question"] + ANSWER_INSTRUCTION,
                expected=(twin["answer"],),
            )
        )
    return items


def load_pair_items(seed: int) -> list[EvalItem]:
    """Networked convenience for `__main__.build_items`: loads the real
    GSM8K test split and builds the 94 pair items from the committed
    templates."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    return build_pair_items(TEMPLATES, ds, seed)
