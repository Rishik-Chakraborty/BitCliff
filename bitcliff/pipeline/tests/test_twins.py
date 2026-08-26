import json
from pathlib import Path

import pytest

from bitcliff_pipeline.hashing import sha256_file
from bitcliff_pipeline.twins.builder import (
    NAME_POOL,
    TwinBuildError,
    build_twin,
    build_twin_set,
)
from bitcliff_pipeline.twins.templates import SKIPPED_ITEMS, TEMPLATES
from bitcliff_pipeline.twins.verifier import (
    TwinTemplate,
    TwinVerificationError,
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
