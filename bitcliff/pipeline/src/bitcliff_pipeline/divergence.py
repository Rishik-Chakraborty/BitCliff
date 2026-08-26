from .generate import OutputRecord


def first_divergence_index(a: list, b: list) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def divergence_for_records(baseline: OutputRecord, other: OutputRecord, tokenize) -> int | None:
    return first_divergence_index(tokenize(baseline.text), tokenize(other.text))
