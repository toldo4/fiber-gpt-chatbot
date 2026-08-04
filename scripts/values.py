"""Value coercion shared by the extractors and the traversal engine.

Both an LLM and a human typing at a prompt produce messy answers — "~6",
"6 g/100 g", "5-7", "Yes.", "true". Coercion lives here so that a value means
the same thing however it entered the system.
"""

from __future__ import annotations

import re

UNKNOWN = "unknown"

_NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
# Captures both ends so "5-7" is read as a range rather than as 5 followed by
# the negative number -7.
_RANGE = re.compile(
    r"^\s*(?:about\s+|approx\.?\s+|~\s*)?"
    r"(\d+(?:[.,]\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:[.,]\d+)?)"
)

_TRUE = {"yes", "y", "true", "t", "1", "affirmative"}
_FALSE = {"no", "n", "false", "f", "0", "negative"}
_NULLISH = {"", "null", "none", "unknown", "unsure", "n/a", "na", "nan", "?", "idk"}


def coerce_boolean(value: object) -> str:
    """Normalise to exactly "yes", "no", or "unknown"."""
    if value is None:
        return UNKNOWN
    if isinstance(value, bool):
        return "yes" if value else "no"
    token = str(value).strip().strip(".!").lower()
    if token in _TRUE:
        return "yes"
    if token in _FALSE:
        return "no"
    return UNKNOWN


def coerce_number(value: object) -> float | None:
    """Pull a single float out of a messy answer, or None if there isn't one.

    A range ("5-7", "5 to 7") collapses to its midpoint, which is the least
    misleading single value to compare against a threshold.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().lower().replace("−", "-")
    if text in _NULLISH:
        return None

    span = _RANGE.match(text)
    if span:
        lo, hi = _to_float(span.group(1)), _to_float(span.group(2))
        if lo is not None and hi is not None:
            return (lo + hi) / 2

    match = _NUMBER.search(text)
    return _to_float(match.group()) if match else None


def coerce_categorical(value: object, options: list[str]) -> str:
    """Snap to one of `options`, case-insensitively, else "unknown"."""
    if value is None:
        return UNKNOWN
    token = str(value).strip().lower()
    for option in options:
        if token == option.lower():
            return option
    return UNKNOWN


def _to_float(token: str) -> float | None:
    try:
        return float(token.replace(",", "."))
    except ValueError:
        return None
