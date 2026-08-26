from bitcliff_pipeline.config import GenSettings
from bitcliff_pipeline.generate import OutputRecord
from bitcliff_pipeline.grading import (
    GradeResult,
    detect_loop,
    grade_record,
    read_grades,
    write_grades,
)
from bitcliff_pipeline.items import EvalItem
import dataclasses

GEN = dataclasses.asdict(GenSettings(42, 0.0, 1, 640, 4096))


def record(item, text, finish_reason="stop", quant="Q4_K_M"):
    return OutputRecord(
        item_id=item.id, suite=item.suite, quant_label=quant,
        model_sha256="abc", prompt=item.prompt, text=text,
        finish_reason=finish_reason, gen_settings=GEN, machine="test",
    )


def test_detect_loop_on_repeated_tail():
    assert detect_loop("Setup text. " + "the answer is the answer is " * 6)
    assert detect_loop("na na na " * 20, min_len=5)


def test_detect_loop_false_on_normal_text():
    assert not detect_loop("The two codes are 1234 and 5678.")
    assert not detect_loop("")


def test_grade_record_longctx_retrieval_correct():
    item = EvalItem("longctx_retrieval-1-000", "longctx_retrieval", "codes?", ("1234", "5678"))
    g = grade_record({item.id: item}, record(item, "They are 1234 and 5678."))
    assert g == GradeResult(
        "longctx_retrieval-1-000", "longctx_retrieval", "Q4_K_M", "correct", False, False
    )


def test_grade_record_truncation_is_separate_flag():
    item = EvalItem("arithmetic-1-000", "arithmetic", "2+2?", ("4",))
    g = grade_record({item.id: item}, record(item, "Let me think step by", finish_reason="length"))
    assert g.state == "wrong"       # never arrived => wrong (spec §6)
    assert g.truncated is True      # ...but tracked separately


def test_grade_record_spectacle_unscored():
    item = EvalItem("spec-001", "spectacle", "haiku", None)
    g = grade_record({item.id: item}, record(item, "five seven five"))
    assert g.state == "unscored"


def test_grades_jsonl_roundtrip(tmp_path):
    grades = [
        GradeResult("a", "retrieval", "Q8_0", "correct", False, False),
        GradeResult("b", "arithmetic", "Q2_K", "wrong", True, True),
    ]
    path = tmp_path / "grades.jsonl"
    write_grades(grades, path)
    assert read_grades(path) == grades
