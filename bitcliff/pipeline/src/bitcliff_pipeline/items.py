from dataclasses import dataclass


@dataclass(frozen=True)
class EvalItem:
    """One evaluation prompt. expected is None for unscored (spectacle) items."""

    id: str
    suite: str
    prompt: str
    expected: tuple[str, ...] | None
