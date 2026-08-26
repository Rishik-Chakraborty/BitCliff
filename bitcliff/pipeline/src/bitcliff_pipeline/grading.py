import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from .generate import OutputRecord
from .items import EvalItem
from .suites import arithmetic, factual_qa, longctx_retrieval, spectacle

GRADERS = {
    "longctx_retrieval": longctx_retrieval.grade,
    "arithmetic": arithmetic.grade,
    # PREREG §3.3: twins are graded by the identical §3.2 rule — same
    # function object, deliberately not a copy.
    "arithmetic_twins": arithmetic.grade,
    "spectacle": spectacle.grade,
    "factual_qa": factual_qa.grade,
}


@dataclass(frozen=True)
class GradeResult:
    item_id: str
    suite: str
    quant_label: str
    state: str
    truncated: bool
    loop: bool


def detect_loop(text: str, min_len: int = 10, min_repeats: int = 3) -> bool:
    tail = text[-2000:].rstrip()
    max_size = len(tail) // min_repeats
    for size in range(min_len, max_size + 1):
        chunk = tail[-size:]
        if chunk * min_repeats == tail[-size * min_repeats :]:
            return True
    return False


def grade_record(items_by_id: dict[str, EvalItem], record: OutputRecord) -> GradeResult:
    item = items_by_id[record.item_id]
    state = GRADERS[item.suite](item, record.text)
    return GradeResult(
        item_id=item.id,
        suite=item.suite,
        quant_label=record.quant_label,
        state=state,
        truncated=record.finish_reason == "length",
        loop=detect_loop(record.text),
    )


def write_grades(grades: list[GradeResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for g in grades:
            f.write(json.dumps(dataclasses.asdict(g)) + "\n")


def read_grades(path: Path) -> list[GradeResult]:
    field_names = {f.name for f in dataclasses.fields(GradeResult)}
    out = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        out.append(GradeResult(**{k: v for k, v in d.items() if k in field_names}))
    return out
