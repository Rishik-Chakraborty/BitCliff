import random

from ..items import EvalItem

FIRST_NAMES = [
    "Alice", "Marcus", "Priya", "Chen", "Fatima", "Diego", "Yuki", "Omar",
    "Ingrid", "Kwame", "Lena", "Rafael", "Sofia", "Tariq", "Mei", "Anders",
    "Zara", "Viktor", "Nadia", "Jamal", "Elena", "Hiro", "Amara", "Luca",
]

PROMPT_TEMPLATE = (
    "Here is a list of people and their two ID codes.\n\n{pairs}\n\n"
    "Question: What are the two ID codes for {key}? "
    "Answer with just the two codes."
)


def generate_items(n_items: int, n_pairs: int, seed: int) -> list[EvalItem]:
    rng = random.Random(seed)
    items = []
    for i in range(n_items):
        names = rng.sample(FIRST_NAMES, n_pairs)
        codes = rng.sample(range(1000, 10000), n_pairs * 2)
        assoc = {
            name: (str(codes[2 * j]), str(codes[2 * j + 1]))
            for j, name in enumerate(names)
        }
        target = rng.choice(names)
        pairs = "\n".join(f"{n}: {a}, {b}" for n, (a, b) in assoc.items())
        items.append(
            EvalItem(
                id=f"retrieval-{seed}-{i:03d}",
                suite="retrieval",
                prompt=PROMPT_TEMPLATE.format(pairs=pairs, key=target),
                expected=assoc[target],
            )
        )
    return items


def grade(item: EvalItem, text: str) -> str:
    hits = sum(1 for code in item.expected if code in text)
    return {2: "correct", 1: "partial"}.get(hits, "wrong")
