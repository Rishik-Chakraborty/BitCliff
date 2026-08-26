from pathlib import Path

import yaml

from ..items import EvalItem


def load_items(path: str | Path) -> list[EvalItem]:
    raw = yaml.safe_load(Path(path).read_text())
    return [
        EvalItem(id=p["id"], suite="spectacle", prompt=p["prompt"], expected=None)
        for p in raw["prompts"]
    ]


def grade(item: EvalItem, text: str) -> str:
    return "unscored"
