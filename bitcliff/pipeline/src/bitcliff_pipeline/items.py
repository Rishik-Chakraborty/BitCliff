from dataclasses import dataclass


@dataclass(frozen=True)
class EvalItem:
    """One evaluation prompt. expected is None for unscored (spectacle) items.

    prompt_tokens holds the full token-id prompt for suites (e.g.
    longctx_retrieval) that assemble the prompt in token space rather than
    text space; prompt then holds only a short human-readable descriptor
    (never the underlying document). It is None for suites that generate
    from item.prompt directly.
    """

    id: str
    suite: str
    prompt: str
    expected: tuple[str, ...] | None
    prompt_tokens: tuple[int, ...] | None = None
