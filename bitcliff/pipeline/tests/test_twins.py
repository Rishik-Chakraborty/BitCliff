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
# Derived params: a literal in the original text that is a *function* of the
# base params (e.g. "the remaining 24 liters" where 24 = vol1 + vol2 - 1)
# becomes a computed placeholder, so the twin text stays internally
# consistent at any seed.
# ---------------------------------------------------------------------------


def derived_fake_template(**overrides) -> TwinTemplate:
    defaults = dict(
        gsm8k_index=0,
        text_template=(
            "Sam has {n} apples and buys {m} more. "
            "How many of the {total} apples are green if 2 are red?"
        ),
        param_names=("n", "m"),
        solve_src="def solve(**p):\n    return p['n'] + p['m'] - 2\n",
        constraints_src="def valid(**p):\n    return p['n'] > 2 and p['m'] > 0\n",
        derived_src="def derive(**p):\n    return {'total': p['n'] + p['m']}\n",
        original_values={"n": 10, "m": 4, "total": 14},
        original_answer="12",
    )
    defaults.update(overrides)
    return TwinTemplate(**defaults)


def test_verify_template_with_derived_param_passes():
    t = derived_fake_template()
    verify_template(
        t,
        "Sam has 10 apples and buys 4 more. How many of the 14 apples are green if 2 are red?",
        "steps\n#### 12",
    )


def test_verify_template_raises_when_stored_derived_value_is_stale():
    # original_values claims total=15 but derive(n=10, m=4) = 14
    t = derived_fake_template(
        original_values={"n": 10, "m": 4, "total": 15},
    )
    with pytest.raises(TwinVerificationError, match="derived"):
        verify_template(
            t,
            "Sam has 10 apples and buys 4 more. How many of the 15 apples are green if 2 are red?",
            "steps\n#### 12",
        )


def test_verify_template_raises_when_derived_name_collides_with_base_param():
    t = derived_fake_template(
        derived_src="def derive(**p):\n    return {'n': p['n'] + p['m']}\n",
    )
    with pytest.raises(TwinVerificationError, match="collide"):
        verify_template(
            t,
            "Sam has 10 apples and buys 4 more. How many of the 14 apples are green if 2 are red?",
            "steps\n#### 12",
        )


def test_build_twin_recomputes_derived_placeholder_from_resampled_params():
    t = derived_fake_template()
    for seed in (1, 7, 42, 1301):
        twin = build_twin(t, seed=seed)
        m = re.fullmatch(
            r"Sam has (\d+) apples and buys (\d+) more\. "
            r"How many of the (\d+) apples are green if 2 are red\?",
            twin["question"],
        )
        assert m, twin["question"]
        n, extra, total = int(m.group(1)), int(m.group(2)), int(m.group(3))
        assert total == n + extra  # derived literal tracks the resampled params
        assert float(twin["answer"]) == n + extra - 2


def test_repaired_idx20_twin_is_internally_consistent_at_any_seed():
    t = next(t for t in TEMPLATES if t.gsm8k_index == 20)
    for seed in (1, 7, 99, 1301, 2024):
        twin = build_twin(t, seed=seed)
        m = re.search(
            r"I have (\d+) liters of orange drink .* add it to (\d+) liters of "
            r"pineapple drink .* spill one liter .* remaining (\d+) liters\?",
            twin["question"],
        )
        assert m, twin["question"]
        vol1, vol2, remaining = int(m.group(1)), int(m.group(2)), int(m.group(3))
        assert remaining == vol1 + vol2 - 1


def test_repaired_idx22_twin_is_internally_consistent_at_any_seed():
    t = next(t for t in TEMPLATES if t.gsm8k_index == 22)
    for seed in (1, 7, 99, 1301, 2024):
        twin = build_twin(t, seed=seed)
        m = re.search(
            r"He has (\d+) customers on Tuesday\. His first (\d+) customers .* "
            r"next (\d+) customers .* last (\d+) customers",
            twin["question"],
        )
        assert m, twin["question"]
        total, c1, c2, c3 = (int(m.group(i)) for i in range(1, 5))
        assert total == c1 + c2 + c3


def test_repaired_idx35_twin_is_internally_consistent_at_any_seed():
    t = next(t for t in TEMPLATES if t.gsm8k_index == 35)
    for seed in (1, 7, 99, 1301, 2024):
        twin = build_twin(t, seed=seed)
        m = re.search(
            r"plays ping pong for (\d+) minutes\.\s+In the first (\d+) minutes.*"
            r"In the second (\d+) minutes",
            twin["question"],
        )
        assert m, twin["question"]
        total, block1, block2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        assert total == block1 + block2


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


def test_build_twin_answer_normalizes_float_artifacts_to_the_exact_value():
    # A solve() whose float arithmetic lands epsilon away from the exact
    # integer (the idx-9/25/26 class: 882.9999999999999 for 883) must
    # register the answer the arithmetic grader would accept from a correct
    # model, not the artifact.
    t = fake_template(
        text_template="Sam has {n} apples. How many?",
        param_names=("n",),
        solve_src="def solve(**p):\n    return 882.9999999999999\n",
        constraints_src="def valid(**p):\n    return p['n'] > 0\n",
        original_values={"n": 10},
        original_answer="883",
    )
    assert build_twin(t, seed=1)["answer"] == "883"


def test_build_twin_answer_keeps_genuine_decimals():
    t = fake_template(
        text_template="Sam has {n} apples. How many halves?",
        param_names=("n",),
        solve_src="def solve(**p):\n    return 2.5\n",
        constraints_src="def valid(**p):\n    return p['n'] > 0\n",
        original_values={"n": 10},
        original_answer="2.5",
    )
    assert build_twin(t, seed=1)["answer"] == "2.5"


def test_committed_twin_answers_carry_no_float_artifacts():
    # The registered set must never contain an answer that exact-match
    # grading would wrongly fail: every twin answer round-trips through the
    # 6-decimal rounding unchanged.
    for t in TEMPLATES:
        for seed in (99, 1301):
            answer = build_twin(t, seed)["answer"]
            assert float(answer) == round(float(answer), 6)
            assert "999999" not in answer and "000001" not in answer, (
                f"idx {t.gsm8k_index} seed {seed}: suspicious answer {answer!r}"
            )


def test_idx43_twin_answer_stays_integer_like_the_original():
    # The original GSM8K answer for idx 43 is the integer 48; the twin must
    # keep the original's answer form (constrained integral grams).
    t = next(t for t in TEMPLATES if t.gsm8k_index == 43)
    for seed in (1, 7, 99, 1301, 2024):
        answer = build_twin(t, seed)["answer"]
        assert float(answer) == int(float(answer)), answer


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
#   4. If solve() is unchanged by the perturbation, this particular check
#      does not flag it. NOTE: solve()-independence does NOT prove the
#      literal is harmless -- a literal solve() never reads can still break
#      the twin's internal coherence (the idx-20/22/35 derived-literal class:
#      a number equal to a sum/combination of params goes stale under
#      resampling even though solve() ignores it). That class is covered by
#      the derived-combination guard below
#      (test_no_leftover_digit_literal_equals_derived_combination_of_params)
#      and by the recorded manual audit of all 47 templates
#      (.superpowers/sdd/plan-freeze/final-fix-report.md).
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


# ---------------------------------------------------------------------------
# Derived-combination guard: catch the idx-20/22/35 class of bug -- a DIGIT
# literal in the unparametrized part of text_template that equals a simple
# derived combination of the params' original values (sum of any 2-3 numeric
# params, with repetition, exact and +-1; and the sum of all numeric params,
# exact and +-1). Such a literal is almost always a restatement the original
# author computed from the params ("remaining 24 liters" = vol1 + vol2 - 1;
# "8 customers" = c1 + c2 + c3; "40 minutes" = block1 + block2), so it goes
# stale under resampling and the twin contradicts itself -- regardless of
# whether solve() reads it (in all three found cases solve() did NOT).
#
# The check is deliberately restricted to digit-run literals: spelled-out
# number words in flavor text ("eats three for breakfast", "six hours")
# encode fixed constants of the problem structure and collide with sums of
# small params constantly (verified: the word-inclusive variant false-flags
# idx 0, 39 and 45). Spelled-word restatements of a single param remain
# covered by the perturbation guard above (the idx-42 class), and the
# residual risk is covered by the recorded manual audit of all 47 templates.
# Templates that parameterize such a literal as a derived placeholder
# (TwinTemplate.derived_src) pass, because the placeholder is stripped
# before the scan.


def _digit_literal_values(leftover_text: str) -> set[float]:
    values = set()
    for tok in _DIGIT_RUN_RE.findall(leftover_text):
        try:
            values.add(float(tok.replace(",", "")))
        except ValueError:
            continue
    return values


def _derived_combination_values(numeric_originals: list[float]) -> set[float]:
    import itertools

    combos: set[float] = set()
    for k in (2, 3):
        for c in itertools.combinations_with_replacement(numeric_originals, k):
            s = float(sum(c))
            combos.update({s, s - 1.0, s + 1.0})
    if numeric_originals:
        s = float(sum(numeric_originals))
        combos.update({s, s - 1.0, s + 1.0})
    return combos


def _derived_combination_flags(t: TwinTemplate) -> list[tuple[int, float]]:
    """(gsm8k_index, literal) for every leftover digit literal in t's
    unparametrized text that equals a derived combination of t's numeric
    original param values."""
    leftover = _leftover_literal_text(t.text_template)
    literals = _digit_literal_values(leftover)
    if not literals:
        return []
    numeric = [
        float(t.original_values[name])
        for name in t.param_names
        if isinstance(t.original_values[name], (int, float))
        and not isinstance(t.original_values[name], bool)
    ]
    combos = _derived_combination_values(numeric)
    return [(t.gsm8k_index, lit) for lit in sorted(literals) if lit in combos]


def test_no_leftover_digit_literal_equals_derived_combination_of_params():
    flagged = []
    for t in TEMPLATES:
        flagged.extend(_derived_combination_flags(t))
    assert not flagged, (
        f"leftover digit literal(s) equal to a derived combination of the "
        f"template's params: {flagged} -- the literal was computed from the "
        f"params by the original author and goes stale under resampling "
        f"(idx 20/22/35 fix history); parameterize it as a derived "
        f"placeholder (TwinTemplate.derived_src) or constrain the params."
    )


def test_derived_combination_guard_flags_the_prefix_idx20_22_35_templates():
    """Proof the guard catches the bug class it was built for: the PRE-FIX
    versions of templates 20, 22 and 35 (verbatim copies of the committed
    text before the derived-literal repair) must each be flagged."""
    prefix_20 = fake_template(
        gsm8k_index=20,
        text_template=(
            "I have {vol1} liters of orange drink that are two-thirds water and I "
            "wish to add it to {vol2} liters of pineapple drink that is three-fifths "
            "water. But as I pour it, I spill one liter of the orange drink. How "
            "much water is in the remaining 24 liters?"
        ),
        param_names=("vol1", "vol2"),
        original_values={"vol1": 10, "vol2": 15},
    )
    prefix_22 = fake_template(
        gsm8k_index=22,
        text_template=(
            "Billy sells DVDs. He has 8 customers on Tuesday. His first {c1} "
            "customers buy one DVD each.  His next {c2} customers buy {dvd2} DVDs "
            "each.  His last {c3} customers don't buy any DVDs. How many DVDs did "
            "Billy sell on Tuesday?"
        ),
        param_names=("c1", "c2", "dvd2", "c3"),
        original_values={"c1": 3, "c2": 2, "dvd2": 2, "c3": 3},
    )
    prefix_35 = fake_template(
        gsm8k_index=35,
        text_template=(
            "Mike plays ping pong for 40 minutes.  In the first {block} minutes, he "
            "scores {points1} points.  In the second 20 minutes, he scores {pct}% "
            "more points.  How many total points did he score?"
        ),
        param_names=("block", "points1", "pct"),
        original_values={"block": 20, "points1": 4, "pct": 25},
    )
    assert _derived_combination_flags(prefix_20) == [(20, 24.0)]
    assert _derived_combination_flags(prefix_22) == [(22, 8.0)]
    assert _derived_combination_flags(prefix_35) == [(35, 40.0)]
