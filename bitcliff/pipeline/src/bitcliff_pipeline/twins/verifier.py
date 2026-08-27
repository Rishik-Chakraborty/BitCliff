import re
from dataclasses import dataclass

from ..suites.arithmetic import _normalize as _normalize_answer_string
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

    ``derived_src`` (optional) is Python source defining
    ``def derive(**base_params) -> dict`` returning extra placeholder values
    the ORIGINAL text states as literals but that are functions of the base
    params (e.g. "the remaining 24 liters" where 24 = vol1 + vol2 - 1).
    Derived names appear as placeholders in ``text_template`` and their
    ORIGINAL values are stored in ``original_values`` (so the exact-text
    round trip against the real GSM8K record still holds), but they are
    never resampled: the builder recomputes them from the resampled base
    params, so the twin text stays internally consistent at any seed.
    ``verify_template`` additionally asserts derive(**original base params)
    reproduces the stored original derived values exactly.
    """

    gsm8k_index: int
    text_template: str
    param_names: tuple[str, ...]
    solve_src: str
    constraints_src: str
    original_values: dict
    original_answer: str
    derived_src: str | None = None


def _normalize_number(value) -> str:
    """Normalize a solve() result to the exact answer string the arithmetic
    grader would accept from a correct model.

    The pre-rounding step lives here (rather than in every solve()) because
    this is the single choke point through which both verify_template and
    build_twin format answers: round to 6 decimals first, killing binary
    float artifacts (882.9999999999999 -> 883, 0.9999999999999997 -> 1,
    671.5999999999999 -> 671.6) so a registered twin answer can never be one
    that exact-match grading would wrongly fail. Formatting then reuses the
    arithmetic suite's own normalizer (integer-valued -> integer string,
    else decimal string) — the same function the grader applies to model
    output — deliberately imported, not duplicated.
    """
    rounded = round(float(value), 6)
    return _normalize_answer_string(repr(rounded))


def load_solve(solve_src: str):
    ns: dict = {}
    exec(solve_src, ns)  # noqa: S102 - trusted, repo-authored template source
    return ns["solve"]


def load_valid(constraints_src: str):
    ns: dict = {}
    exec(constraints_src, ns)  # noqa: S102 - trusted, repo-authored template source
    return ns["valid"]


def load_derive(derived_src: str):
    ns: dict = {}
    exec(derived_src, ns)  # noqa: S102 - trusted, repo-authored template source
    return ns["derive"]


def verify_template(t: TwinTemplate, gsm8k_question: str, gsm8k_answer: str) -> None:
    """Raise TwinVerificationError unless t round-trips against the real
    GSM8K record (gsm8k_question, gsm8k_answer).

    Checks, in order:
      (a) t.text_template.format(**t.original_values) == gsm8k_question, exactly.
      (b) solve(**original numeric params) equals the #### answer extracted
          from gsm8k_answer.
      (c) if t.derived_src is set, derive(**original base params) reproduces
          the stored original derived values exactly (names must not collide
          with base params).
      (d) every param's (and derived placeholder's) original value appears in
          the formatted text.
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

    derived_names: tuple[str, ...] = ()
    if t.derived_src is not None:
        derive = load_derive(t.derived_src)
        base = {name: t.original_values[name] for name in t.param_names}
        try:
            derived = derive(**base)
        except Exception as exc:  # noqa: BLE001 - surface any derive() failure
            raise TwinVerificationError(
                f"template idx {t.gsm8k_index}: derive(**original base params) raised: {exc!r}"
            ) from exc
        derived_names = tuple(derived)
        for name, value in derived.items():
            if name in t.param_names:
                raise TwinVerificationError(
                    f"template idx {t.gsm8k_index}: derived param {name!r} "
                    f"collides with a base param name"
                )
            if name not in t.original_values or t.original_values[name] != value:
                raise TwinVerificationError(
                    f"template idx {t.gsm8k_index}: derived param {name!r}: "
                    f"derive(**original base params) = {value!r} but "
                    f"original_values stores {t.original_values.get(name)!r}"
                )

    for name in t.param_names + derived_names:
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
