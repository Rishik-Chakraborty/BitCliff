import json
import random
from pathlib import Path

from ..hashing import sha256_file
from .verifier import TwinTemplate, _normalize_number, load_solve, load_valid, verify_template

# Fixed pool of names used to resample any string-valued ("name") params.
# Deliberately distinct from names appearing in the first-60 GSM8K items used
# for template authoring, so a twin can never coincidentally reproduce the
# original wording via a resampled name.
NAME_POOL: tuple[str, ...] = (
    "Priya", "Solomon", "Ines", "Tobias", "Naledi", "Quentin", "Farah",
    "Bao", "Elif", "Ronan", "Amara", "Deshi", "Yusuf", "Marisol", "Otieno",
)

MAX_RESAMPLE_ATTEMPTS = 2000


class TwinBuildError(Exception):
    pass


def _resample_value(rng: random.Random, name: str, original, used_names: set):
    if isinstance(original, bool):  # guard: bool is a subclass of int
        raise TwinBuildError(f"param {name!r} has a bool value; not supported")
    if isinstance(original, int):
        lo = max(1, round(original * 0.5))
        hi = max(lo + 1, round(original * 2))
        return rng.randint(lo, hi)
    if isinstance(original, float):
        lo, hi = original * 0.5, original * 2
        return round(rng.uniform(lo, hi), 2)
    if isinstance(original, str):
        choices = [n for n in NAME_POOL if n not in used_names]
        value = rng.choice(choices)
        used_names.add(value)
        return value
    raise TwinBuildError(f"param {name!r} has unsupported type {type(original)!r}")


def build_twin(t: TwinTemplate, seed: int) -> dict:
    """Resample t's numeric params (and any string/name params, from a fixed
    name pool) until constraints_src's valid(...) accepts them, deterministic
    per (template, seed). Returns
    {"question", "answer", "template_index", "seed"}.
    """
    rng = random.Random(f"{t.gsm8k_index}:{seed}")
    valid = load_valid(t.constraints_src)
    solve = load_solve(t.solve_src)

    for _ in range(MAX_RESAMPLE_ATTEMPTS):
        used_names: set = set()
        candidate = {
            name: _resample_value(rng, name, t.original_values[name], used_names)
            for name in t.param_names
        }
        if valid(**candidate):
            answer = solve(**candidate)
            question = t.text_template.format(**candidate)
            return {
                "question": question,
                "answer": _normalize_number(answer),
                "template_index": t.gsm8k_index,
                "seed": seed,
            }

    raise TwinBuildError(
        f"template idx {t.gsm8k_index}: no valid resample found after "
        f"{MAX_RESAMPLE_ATTEMPTS} attempts"
    )


def build_twin_set(templates, gsm8k_records, seed: int, out_path: Path) -> None:
    """Verify every template against its real GSM8K record, build one twin
    per template, write the twins as JSONL to out_path (expected under
    private/twins/, which is gitignored), and print the file's sha256."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for t in templates:
        record = gsm8k_records[t.gsm8k_index]
        verify_template(t, record["question"], record["answer"])

    twins = [build_twin(t, seed) for t in templates]

    with open(out_path, "w") as f:
        for twin in twins:
            f.write(json.dumps(twin, sort_keys=True))
            f.write("\n")

    digest = sha256_file(out_path)
    print(f"wrote {len(twins)} twins to {out_path} sha256={digest}")
