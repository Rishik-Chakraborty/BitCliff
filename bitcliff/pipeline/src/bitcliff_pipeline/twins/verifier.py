import re
from dataclasses import dataclass

from ..suites.arithmetic import extract_final_number


class TwinVerificationError(Exception):
    pass


@dataclass(frozen=True)
class TwinTemplate:
    """A GSM8K contamination-control twin template.

    ``solve_src`` is Python source defining ``def solve(**params)`` that
    computes the numeric answer from the template's parameters.
    ``constraints_src`` is Python source defining ``def valid(**params)``
    that returns True iff a set of (possibly resampled) parameter values
    keeps the problem sane (positivity, integer intermediates, ordering
    relations the text asserts, etc).
    """

    gsm8k_index: int
    text_template: str
    param_names: tuple[str, ...]
    solve_src: str
    constraints_src: str
    original_values: dict
    original_answer: str


def _normalize_number(value) -> str:
    f = float(value)
    return str(int(f)) if f == int(f) else str(f)


def load_solve(solve_src: str):
    ns: dict = {}
    exec(solve_src, ns)  # noqa: S102 - trusted, repo-authored template source
    return ns["solve"]


def load_valid(constraints_src: str):
    ns: dict = {}
    exec(constraints_src, ns)  # noqa: S102 - trusted, repo-authored template source
    return ns["valid"]


def verify_template(t: TwinTemplate, gsm8k_question: str, gsm8k_answer: str) -> None:
    """Raise TwinVerificationError unless t round-trips against the real
    GSM8K record (gsm8k_question, gsm8k_answer).

    Checks, in order:
      (a) t.text_template.format(**t.original_values) == gsm8k_question, exactly.
      (b) solve(**original numeric params) equals the #### answer extracted
          from gsm8k_answer.
      (c) every param's original value appears in the formatted text.
    """
    try:
        formatted = t.text_template.format(**t.original_values)
    except (KeyError, IndexError) as exc:
        raise TwinVerificationError(
            f"template idx {t.gsm8k_index}: text_template.format failed: {exc!r}"
        ) from exc

    if formatted != gsm8k_question:
        raise TwinVerificationError(
            f"template idx {t.gsm8k_index}: formatted text does not match GSM8K "
            f"question exactly.\n  formatted: {formatted!r}\n  expected:  {gsm8k_question!r}"
        )

    gold = extract_final_number(gsm8k_answer)
    if gold is None:
        raise TwinVerificationError(
            f"template idx {t.gsm8k_index}: could not extract #### answer from gsm8k_answer"
        )

    solve = load_solve(t.solve_src)
    try:
        computed = solve(**t.original_values)
    except Exception as exc:  # noqa: BLE001 - surface any solve() failure as a verification failure
        raise TwinVerificationError(
            f"template idx {t.gsm8k_index}: solve(**original_values) raised: {exc!r}"
        ) from exc

    got = _normalize_number(computed)
    if got != gold:
        raise TwinVerificationError(
            f"template idx {t.gsm8k_index}: solve(**original_values) = {got} "
            f"but GSM8K #### answer = {gold}"
        )

    if t.original_answer != gold:
        raise TwinVerificationError(
            f"template idx {t.gsm8k_index}: stored original_answer {t.original_answer!r} "
            f"does not match GSM8K #### answer {gold!r}"
        )

    for name in t.param_names:
        value = t.original_values[name]
        expected_repr = _formatted_repr_for_param(t.text_template, name, value)
        if expected_repr is None:
            raise TwinVerificationError(
                f"template idx {t.gsm8k_index}: param {name!r} does not appear as a "
                f"placeholder in text_template"
            )
        if expected_repr not in formatted:
            raise TwinVerificationError(
                f"template idx {t.gsm8k_index}: param {name!r}'s formatted value "
                f"{expected_repr!r} does not appear in the formatted text"
            )


def _formatted_repr_for_param(text_template: str, name: str, value) -> str | None:
    """Render `value` the same way text_template's {name} (or {name:spec})
    placeholder would render it, so callers can check it shows up in the
    formatted text -- honoring any format spec (e.g. ",", ".2f")."""
    m = re.search(r"\{" + re.escape(name) + r"(:([^}]*))?\}", text_template)
    if m is None:
        return None
    spec = m.group(2) or ""
    return format(value, spec)
