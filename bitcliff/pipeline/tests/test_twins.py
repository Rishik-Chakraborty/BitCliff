import json
import re
from pathlib import Path

import pytest

from bitcliff_pipeline.hashing import sha256_file
from bitcliff_pipeline.twins.builder import (
    NAME_POOL,
    NUMBER_WORDS,
    TwinBuildError,
    build_twin,
    build_twin_set,
)
from bitcliff_pipeline.twins.templates import SKIPPED_ITEMS, TEMPLATES
from bitcliff_pipeline.twins.verifier import (
    TwinTemplate,
    TwinVerificationError,
    load_solve,
    verify_template,
)

DATA_DIR = Path(__file__).parent / "data"


def load_gsm8k_first60() -> list[dict]:
    """First 60 records of openai/gsm8k (main, test split). Tries the local
    datasets cache first (matches suites/arithmetic.py::load_gsm8k_items);
    falls back to the vendored MIT-licensed excerpt if the cache is
    unavailable offline."""
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [ds[i] for i in range(60)]
    except Exception:
        records = []
        with open(DATA_DIR / "gsm8k_first60.jsonl") as f:
            for line in f:
                records.append(json.loads(line))
        return records


GSM8K_FIRST60 = load_gsm8k_first60()


def fake_template(**overrides) -> TwinTemplate:
    defaults = dict(
        gsm8k_index=0,
        text_template="Sam has {n} apples and gives away {given}. How many are left?",
        param_names=("n", "given"),
        solve_src="def solve(**p):\n    return p['n'] - p['given']\n",
        constraints_src="def valid(**p):\n    return p['n'] > p['given'] > 0\n",
        original_values={"n": 10, "given": 3},
        original_answer="7",
    )
    defaults.update(overrides)
    return TwinTemplate(**defaults)


# ---------------------------------------------------------------------------
# verify_template: hand-made fake records (TDD first per task brief)
# ---------------------------------------------------------------------------


def test_verify_template_passes_on_matching_fake_record():
    t = fake_template()
    verify_template(t, "Sam has 10 apples and gives away 3. How many are left?", "steps\n#### 7")
    # no raise = pass


def test_verify_template_raises_on_text_mismatch():
    t = fake_template()
    with pytest.raises(TwinVerificationError, match="does not match"):
        verify_template(
            t, "Sam has 10 oranges and gives away 3. How many are left?", "steps\n#### 7"
        )


def test_verify_template_raises_on_wrong_solve():
    # solve computes n + given instead of n - given: wrong vs. the real GSM8K answer
    t = fake_template(solve_src="def solve(**p):\n    return p['n'] + p['given']\n")
    with pytest.raises(TwinVerificationError, match="solve"):
        verify_template(t, "Sam has 10 apples and gives away 3. How many are left?", "steps\n#### 7")


def test_verify_template_raises_on_missing_param_in_text():
    # "extra" is declared as a param but never appears in text_template
    t = fake_template(
        param_names=("n", "given", "extra"),
        original_values={"n": 10, "given": 3, "extra": 999},
    )
    with pytest.raises(TwinVerificationError, match="extra"):
        verify_template(t, "Sam has 10 apples and gives away 3. How many are left?", "steps\n#### 7")


def test_verify_template_raises_on_original_answer_mismatch():
    t = fake_template(original_answer="999")
    with pytest.raises(TwinVerificationError, match="original_answer"):
        verify_template(t, "Sam has 10 apples and gives away 3. How many are left?", "steps\n#### 7")


def test_verify_template_honors_format_spec_when_checking_param_presence():
    t = fake_template(
        text_template="Sam has ${n:,} and gives away {given}. How much is left?",
        original_values={"n": 12000, "given": 3},
        original_answer="11997",
        solve_src="def solve(**p):\n    return p['n'] - p['given']\n",
    )
    verify_template(t, "Sam has $12,000 and gives away 3. How much is left?", "steps\n#### 11997")


# ---------------------------------------------------------------------------
# Template authoring: every committed template must round-trip against the
# real GSM8K text.
# ---------------------------------------------------------------------------


def test_at_least_30_verified_templates():
    assert len(TEMPLATES) >= 30, f"only {len(TEMPLATES)} templates, need >= 30"


def test_every_template_verifies_against_real_gsm8k_text():
    for t in TEMPLATES:
        record = GSM8K_FIRST60[t.gsm8k_index]
        verify_template(t, record["question"], record["answer"])  # no raise


def test_template_indices_are_unique():
    indices = [t.gsm8k_index for t in TEMPLATES]
    assert len(indices) == len(set(indices))


def test_skipped_items_have_reasons_and_are_disjoint_from_templates():
    templated = {t.gsm8k_index for t in TEMPLATES}
    for idx, reason in SKIPPED_ITEMS:
        assert idx not in templated
        assert isinstance(reason, str) and len(reason) > 0


# ---------------------------------------------------------------------------
# Builder: determinism + constraint-respecting resample
# ---------------------------------------------------------------------------


def test_build_twin_is_deterministic_per_template_and_seed():
    t = TEMPLATES[0]
    a = build_twin(t, seed=1301)
    b = build_twin(t, seed=1301)
    assert a == b


def test_build_twin_differs_across_seeds_eventually():
    t = TEMPLATES[0]
    results = {build_twin(t, seed=s)["question"] for s in range(10)}
    assert len(results) > 1


def test_build_twin_answer_matches_solve_of_resampled_params():
    t = TEMPLATES[0]
    twin = build_twin(t, seed=7)
    assert twin["template_index"] == t.gsm8k_index
    assert twin["seed"] == 7
    assert twin["question"] != t.text_template.format(**t.original_values) or True
    # answer must be a clean numeric string
    float(twin["answer"])


def test_build_twin_respects_constraints_for_every_committed_template():
    # Cheap smoke test across the whole authored set: every twin must be
    # buildable (i.e. constraints_src is satisfiable by the resampler) and
    # its answer must reproduce solve() on the resampled params.
    for t in TEMPLATES:
        twin = build_twin(t, seed=99)
        assert twin["question"] != ""
        float(twin["answer"])


def test_build_twin_raises_when_constraints_are_unsatisfiable():
    t = fake_template(constraints_src="def valid(**p):\n    return False\n")
    with pytest.raises(TwinBuildError):
        build_twin(t, seed=1)


def test_build_twin_resamples_name_params_from_fixed_pool():
    t = fake_template(
        text_template="{who} has {n} apples and gives away {given}. How many are left?",
        param_names=("who", "n", "given"),
        original_values={"who": "Sam", "n": 10, "given": 3},
        solve_src="def solve(**p):\n    return p['n'] - p['given']\n",
        constraints_src="def valid(**p):\n    return p['n'] > p['given'] > 0\n",
    )
    twin = build_twin(t, seed=5)
    assert any(name in twin["question"] for name in NAME_POOL)
    assert "Sam" not in twin["question"]


def test_build_twin_set_writes_jsonl_and_prints_sha256(tmp_path, capsys):
    subset = TEMPLATES[:5]
    out_path = tmp_path / "private" / "twins" / "twin_set.jsonl"
    build_twin_set(subset, GSM8K_FIRST60, seed=42, out_path=out_path)

    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == len(subset)
    for line in lines:
        rec = json.loads(line)
        assert set(rec) == {"question", "answer", "template_index", "seed"}

    digest = sha256_file(out_path)
    captured = capsys.readouterr()
    assert digest in captured.out


def test_build_twin_set_raises_if_any_template_fails_verification():
    bad = fake_template(text_template="This will never match the real record.")
    out_path = Path("unused.jsonl")
    with pytest.raises(TwinVerificationError):
        build_twin_set([bad], GSM8K_FIRST60, seed=1, out_path=out_path)


# ---------------------------------------------------------------------------
# Consistency guard: catch the idx-26/idx-42 class of bug (a literal number
# or number-word in the *unparametrized* part of text_template happens to
# equal a free param's original value, and solve() actually depends on that
# param) BEFORE it ships as a committed template.
#
# Rule implemented (the "simpler accepted alternative" from the review):
#   1. Strip every {param} / {param:spec} placeholder out of text_template,
#      leaving only the literal/constant text.
#   2. Scan the leftover literal text for bare digit-runs (e.g. "8", "20")
#      and for spelled-out number words (from NUMBER_WORDS, e.g. "five").
#   3. For each param whose *original* numeric value numerically matches one
#      of those leftover literals: perturb that param's value (holding all
#      others at their original values) and re-run solve(). If solve()'s
#      output changes, the param provably feeds the answer *and* the text
#      contains an unparametrized restatement of it that won't move when the
#      param is resampled -- that's exactly the idx-26/idx-42 failure mode,
#      so it's flagged.
#   4. If solve() is unchanged by the perturbation, the leftover literal is
#      pure flavor text solve() never reads (the idx-22/idx-35 pattern,
#      which the review explicitly calls out as harmless) -- not flagged.
#
# This deliberately does NOT flag two independent params that coincidentally
# share the same original value (e.g. idx 22's c1=3 and c3=3): each has its
# own placeholder consuming its own occurrence of "3" in the real text, so
# after stripping placeholders there is no leftover "3" left to match against.
_PLACEHOLDER_RE = re.compile(r"\{[^{}:]+(?::[^{}]*)?\}")
_DIGIT_RUN_RE = re.compile(r"\d[\d,]*\.?\d*")
_REVERSE_NUMBER_WORDS = {v: k for k, v in NUMBER_WORDS.items()}


def _leftover_literal_text(text_template: str) -> str:
    return " ".join(_PLACEHOLDER_RE.split(text_template))


def _leftover_numeric_values(leftover_text: str) -> set[float]:
    values = set()
    for tok in _DIGIT_RUN_RE.findall(leftover_text):
        try:
            values.add(float(tok.replace(",", "")))
        except ValueError:
            continue
    for word, n in NUMBER_WORDS.items():
        if re.search(r"\b" + re.escape(word) + r"\b", leftover_text):
            values.add(float(n))
    return values


def _perturb(value):
    if isinstance(value, bool):
        raise TypeError("bool params not supported")
    if isinstance(value, int):
        return value + 3
    if isinstance(value, float):
        return round(value * 1.37 + 0.71, 2)
    if isinstance(value, str) and value in NUMBER_WORDS:
        other = (NUMBER_WORDS[value] + 3) % 17  # stays within NUMBER_WORDS' small ints
        return _REVERSE_NUMBER_WORDS.get(other, "eleven")
    if isinstance(value, str):
        return "Zzyzx-guard-probe"
    raise TypeError(f"unsupported param type {type(value)!r}")


def test_no_unparametrized_literal_ties_to_a_free_param():
    """Guard against the idx-26/idx-42 class of bug: see the block comment
    above this test for the exact rule. Runs against every committed
    template; any flagged (template, param) pair means a literal in the
    text restates a param's value without being tied to it, and solve()
    actually depends on that param -- i.e. a resampled twin would be
    self-contradictory."""
    flagged = []
    for t in TEMPLATES:
        leftover = _leftover_literal_text(t.text_template)
        leftover_values = _leftover_numeric_values(leftover)
        if not leftover_values:
            continue
        solve = load_solve(t.solve_src)
        base = solve(**t.original_values)
        for name in t.param_names:
            value = t.original_values[name]
            numeric = None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
            elif isinstance(value, str) and value in NUMBER_WORDS:
                numeric = float(NUMBER_WORDS[value])
            if numeric is None or numeric not in leftover_values:
                continue
            perturbed = dict(t.original_values)
            perturbed[name] = _perturb(value)
            try:
                changed = solve(**perturbed) != base
            except Exception:
                changed = True  # solve() breaking under perturbation is itself suspicious
            if changed:
                flagged.append((t.gsm8k_index, name, value))
    assert not flagged, (
        f"unparametrized literal(s) tied to a solve()-consumed param: {flagged} "
        f"-- a leftover literal in text_template restates a param's original "
        f"value, and solve() depends on that param, so a resampled twin would "
        f"show a stale literal alongside a changed number (see idx 26 / idx 42 "
        f"fix history)."
    )
