"""Argument matchers for expected tool calls (BFCL-style, but forgiving where the server is).

A matcher is one of:

- a string: equal after trimming, ignoring case (`bench-1` matches `BENCH-1`)
- a number: numerically equal; a numeric string counts, because the server accepts it
- a boolean: the same boolean, or the strings "true"/"false"
- `null`: the argument is absent (or null)
- a list: a list of the same length whose items match in any order
- a mapping of operators, all of which must hold:
  - `$contains: text`: substring of a string (ignoring case), or an item of a list
  - `$regex: pattern`: `re.search` on a string, ignoring case
  - `$any_of: [matcher, ...]`: at least one matches
  - `$all: [matcher, ...]`: every one matches
  - `$present: true|false`: the argument is given / absent

Every argument a matcher names must be present unless the matcher allows absence
(`null`, `$present: false`, or such an alternative in `$any_of`). Arguments the task
does not name are not checked.
"""

import re
from typing import Any

OPERATORS = ("$contains", "$regex", "$any_of", "$all", "$present")


class _Missing:
    def __repr__(self) -> str:
        return "MISSING"


MISSING: Any = _Missing()
"""Value of an argument the model did not give."""


def check_matcher(matcher: Any) -> None:
    """Raises ValueError if `matcher` is not valid matcher syntax."""
    if matcher is None or isinstance(matcher, str | int | float | bool):
        return
    if isinstance(matcher, list):
        for item in matcher:
            check_matcher(item)
        return
    if not isinstance(matcher, dict):
        raise ValueError(f"unsupported matcher {matcher!r}")
    if not matcher:
        raise ValueError("empty operator mapping")
    unknown = [k for k in matcher if k not in OPERATORS]
    if unknown:
        raise ValueError(f"unknown operators {unknown} (known: {', '.join(OPERATORS)})")
    for op, arg in matcher.items():
        if op in ("$contains", "$regex") and not isinstance(arg, str):
            raise ValueError(f"{op} takes a string")
        if op == "$regex":
            try:
                re.compile(arg)
            except re.error as exc:
                raise ValueError(f"bad $regex: {exc}") from None
        if op in ("$any_of", "$all"):
            if not isinstance(arg, list) or not arg:
                raise ValueError(f"{op} takes a non-empty list")
            for item in arg:
                check_matcher(item)
        if op == "$present" and not isinstance(arg, bool):
            raise ValueError("$present takes true or false")


def fill(matcher: Any, placeholder: str, value: str) -> Any:
    """`matcher` with `placeholder` replaced by `value` in every string."""
    if isinstance(matcher, str):
        return matcher.replace(placeholder, value)
    if isinstance(matcher, list):
        return [fill(m, placeholder, value) for m in matcher]
    if isinstance(matcher, dict):
        return {k: fill(m, placeholder, value) for k, m in matcher.items()}
    return matcher


def args_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Whether the call's `actual` arguments satisfy every named matcher."""
    return all(matches(m, actual.get(name, MISSING)) for name, m in expected.items())


def matches(matcher: Any, value: Any) -> bool:
    if value is None:
        value = MISSING
    if matcher is None:
        return value is MISSING
    if isinstance(matcher, dict):
        return all(_operator(op, arg, value) for op, arg in matcher.items())
    if value is MISSING:
        return False
    if isinstance(matcher, bool):  # before numbers: bool is an int subclass
        if isinstance(value, str):
            return value.strip().lower() == str(matcher).lower()
        return isinstance(value, bool) and value == matcher
    if isinstance(matcher, int | float):
        if isinstance(value, bool):
            return False
        if isinstance(value, str):
            try:
                value = float(value.strip())
            except ValueError:
                return False
        return isinstance(value, int | float) and value == matcher
    if isinstance(matcher, str):
        return isinstance(value, str) and value.strip().casefold() == matcher.strip().casefold()
    if isinstance(matcher, list):
        return isinstance(value, list) and len(value) == len(matcher) and _unordered(matcher, value)
    raise ValueError(f"unsupported matcher {matcher!r}")


def _operator(op: str, arg: Any, value: Any) -> bool:
    if op == "$present":
        return (value is not MISSING) == arg
    if op == "$any_of":
        return any(matches(m, value) for m in arg)
    if op == "$all":
        return all(matches(m, value) for m in arg)
    if op == "$contains":
        if isinstance(value, str):
            return arg.casefold() in value.casefold()
        if isinstance(value, list):
            return any(matches(arg, item) for item in value)
        return False
    if op == "$regex":
        return isinstance(value, str) and re.search(arg, value, re.IGNORECASE) is not None
    raise ValueError(f"unknown operator {op}")


def _unordered(matchers: list[Any], values: list[Any]) -> bool:
    """Each matcher gets its own value (backtracking; the lists are a few items long)."""
    if not matchers:
        return True
    first, rest = matchers[0], matchers[1:]
    for i, value in enumerate(values):
        if matches(first, value) and _unordered(rest, values[:i] + values[i + 1 :]):
            return True
    return False
