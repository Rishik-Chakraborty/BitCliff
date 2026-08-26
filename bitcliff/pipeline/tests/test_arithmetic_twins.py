"""arithmetic_twins suite: the PREREG §3.3 runnable path — all 47
original+twin pairs run in-suite, graded by the §3.2 arithmetic rule."""

import json
import re
from pathlib import Path

import pytest

from bitcliff_pipeline.config import GenSettings, LadderConfig, QuantFile
from bitcliff_pipeline.grading import GRADERS
from bitcliff_pipeline.items import EvalItem
from bitcliff_pipeline.suites import arithmetic, arithmetic_twins
from bitcliff_pipeline.suites.arithmetic import ANSWER_INSTRUCTION, extract_final_number
from bitcliff_pipeline.twins.builder import build_twin
from bitcliff_pipeline.twins.templates import TEMPLATES

DATA_DIR = Path(__file__).parent / "data"


def load_gsm8k_first60() -> list[dict]:
    """Same loader rule as test_twins.py: real datasets cache first, vendored
    MIT-licensed excerpt as the offline fallback."""
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [ds[i] for i in range(60)]
    except Exception:
        with open(DATA_DIR / "gsm8k_first60.jsonl") as f:
            return [json.loads(line) for line in f]


GSM8K_FIRST60 = load_gsm8k_first60()

SEED = 1301
PAIR_ITEMS = arithmetic_twins.build_pair_items(TEMPLATES, GSM8K_FIRST60, seed=SEED)


def test_build_pair_items_produces_94_items_for_47_templates():
    assert len(TEMPLATES) == 47
    assert len(PAIR_ITEMS) == 94


def test_pair_item_ids_and_suite():
    id_re = re.compile(rf"arithmetic_twins-{SEED}-(orig|twin)-(\d{{3}})")
    for item in PAIR_ITEMS:
        assert item.suite == "arithmetic_twins"
        m = id_re.fullmatch(item.id)
        assert m, item.id
    # every template contributes exactly one orig and one twin, adjacent
    for t, (orig, twin) in zip(TEMPLATES, zip(PAIR_ITEMS[::2], PAIR_ITEMS[1::2])):
        assert orig.id == f"arithmetic_twins-{SEED}-orig-{t.gsm8k_index:03d}"
        assert twin.id == f"arithmetic_twins-{SEED}-twin-{t.gsm8k_index:03d}"


def test_original_items_carry_the_real_gsm8k_question_and_answer():
    for t, orig in zip(TEMPLATES, PAIR_ITEMS[::2]):
        record = GSM8K_FIRST60[t.gsm8k_index]
        assert orig.prompt == record["question"] + ANSWER_INSTRUCTION
        assert orig.expected == (extract_final_number(record["answer"]),)


def test_twin_items_match_the_deterministic_twin_builder():
    for t, twin_item in zip(TEMPLATES, PAIR_ITEMS[1::2]):
        twin = build_twin(t, seed=SEED)
        assert twin_item.prompt == twin["question"] + ANSWER_INSTRUCTION
        assert twin_item.expected == (twin["answer"],)


def test_every_prompt_ends_with_the_registered_answer_instruction():
    assert all(item.prompt.endswith(ANSWER_INSTRUCTION) for item in PAIR_ITEMS)


def test_grader_dispatch_reuses_the_arithmetic_rule():
    assert GRADERS["arithmetic_twins"] is arithmetic.grade
    item = PAIR_ITEMS[0]
    gold = item.expected[0]
    assert GRADERS["arithmetic_twins"](item, f"steps\n#### {gold}") == "correct"
    assert GRADERS["arithmetic_twins"](item, "steps\n#### 999999999") == "wrong"


def test_build_pair_items_rejects_records_that_fail_template_verification():
    from bitcliff_pipeline.twins.verifier import TwinVerificationError

    bad_records = [dict(r) for r in GSM8K_FIRST60]
    bad_records[TEMPLATES[0].gsm8k_index] = {
        "question": "This does not match any template.",
        "answer": "#### 0",
    }
    with pytest.raises(TwinVerificationError):
        arithmetic_twins.build_pair_items(TEMPLATES, bad_records, seed=SEED)


def test_main_build_items_covers_arithmetic_twins_branch(monkeypatch, tmp_path):
    from bitcliff_pipeline.__main__ import build_items

    monkeypatch.setattr(
        arithmetic_twins,
        "load_pair_items",
        lambda seed: arithmetic_twins.build_pair_items(TEMPLATES, GSM8K_FIRST60, seed),
    )
    cfg = LadderConfig(
        model_id="test-model",
        hf_repo="fake/repo",
        f16_path=tmp_path / "f16.gguf",
        quants=(QuantFile("Q4_K_M", "m.gguf", "bartowski", True),),
        generation=GenSettings(42, 0.0, 1, 640, 4096),
        suites={"arithmetic_twins": {"seed": SEED}},
    )
    items = build_items(cfg, base_dir=tmp_path)
    assert len(items) == 94
    assert all(isinstance(i, EvalItem) and i.suite == "arithmetic_twins" for i in items)
