import random
import re

from ..items import EvalItem

_FINAL = re.compile(r"####\s*([-+]?[\d,]*\.?\d+)")
_NUM = re.compile(r"[-+]?[\d,]*\.?\d+")

ANSWER_INSTRUCTION = (
    "\n\nSolve step by step, then give the final answer on its own line as: #### <number>"
)


def _normalize(s: str) -> str:
    value = float(s.replace(",", ""))
    return str(int(value)) if value == int(value) else str(value)


def extract_final_number(text: str) -> str | None:
    m = _FINAL.search(text)
    if m:
        return _normalize(m.group(1))
    nums = _NUM.findall(text)
    return _normalize(nums[-1]) if nums else None


def items_from_records(records, n_items: int, seed: int) -> list[EvalItem]:
    rng = random.Random(seed)
    picked = rng.sample(list(records), n_items)
    items = []
    for i, r in enumerate(picked):
        gold = extract_final_number(r["answer"])
        items.append(
            EvalItem(
                id=f"arithmetic-{seed}-{i:03d}",
                suite="arithmetic",
                prompt=r["question"] + ANSWER_INSTRUCTION,
                expected=(gold,),
            )
        )
    return items


def load_gsm8k_items(n_items: int, seed: int) -> list[EvalItem]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    return items_from_records(ds, n_items, seed)


def grade(item: EvalItem, text: str) -> str:
    got = extract_final_number(text)
    return "correct" if got is not None and got == item.expected[0] else "wrong"
