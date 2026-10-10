import math

import pytest

from bench_evals.scorer import Score
from bench_evals.stats import Rate, compare, rate, task_outcomes, wilson

Z = 1.959963984540054  # two-sided 95 % normal quantile


def wilson_by_hand(k: int, n: int) -> tuple[float, float]:
    """Wilson (1927) score interval, written out independently of statsmodels."""
    p = k / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return centre - half, centre + half


def bounds(r: Rate) -> tuple[float, float]:
    assert r.low is not None
    assert r.high is not None
    return r.low, r.high


def run(task: str, solved: bool | None, *, right_tool: bool | None = None) -> Score:
    valid = solved is not None
    return Score(task, valid, solved, solved if right_tool is None else right_tool, 1)


# --- Wilson ---


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [  # textbook values, 4 decimals (e.g. 7/10 -> 0.3968-0.8922)
        (7, 10, 0.3968, 0.8922),
        (0, 10, 0.0, 0.2775),
        (10, 10, 0.7225, 1.0),
        (1, 2, 0.0945, 0.9055),
    ],
)
def test_wilson_reference_values(k, n, low, high):
    r = wilson(k, n)
    assert (r.successes, r.n) == (k, n)
    assert r.low == pytest.approx(low, abs=1e-4)
    assert r.high == pytest.approx(high, abs=1e-4)


@pytest.mark.parametrize(("k", "n"), [(1, 24), (12, 24), (23, 24), (3, 7), (40, 45)])
def test_wilson_matches_formula(k, n):
    low, high = bounds(wilson(k, n))
    assert (low, high) == pytest.approx(wilson_by_hand(k, n), abs=1e-12)
    assert low <= k / n <= high


def test_wilson_confidence_level_widens():
    (l99, h99), (l95, h95), (l80, h80) = (bounds(wilson(7, 10, c)) for c in (0.99, 0.95, 0.8))
    assert l99 < l95 < l80 < h80 < h95 < h99


def test_wilson_empty_and_invalid():
    assert wilson(0, 0) == Rate(0, 0, None, None)
    assert wilson(0, 0).value is None
    with pytest.raises(ValueError, match="successes"):
        wilson(3, 2)
    with pytest.raises(ValueError, match="successes"):
        wilson(-1, 2)


# --- per-task outcomes over k runs ---


def test_outcome_is_strict_majority_of_valid_runs():
    scores = [
        *[run("two-of-three", s) for s in (True, True, False)],
        *[run("one-of-three", s) for s in (True, False, False)],
        *[run("one-of-two", s) for s in (True, False)],  # tie is not solved
        *[run("one-of-one", s) for s in (True,)],
        *[run("invalid-ignored", s) for s in (True, None, None)],  # 1 of 1 valid
        *[run("all-invalid", s) for s in (None, None)],
        run("single-fail", False),
    ]
    assert task_outcomes(scores) == {
        "two-of-three": True,
        "one-of-three": False,
        "one-of-two": False,
        "one-of-one": True,
        "invalid-ignored": True,
        "all-invalid": None,
        "single-fail": False,
    }


def test_outcome_runs_need_not_be_adjacent():
    scores = [run("a", True), run("b", False), run("a", False), run("b", True), run("a", True)]
    assert task_outcomes(scores) == {"a": True, "b": False}


def test_outcome_other_metric():
    scores = [run("t", False, right_tool=True), run("t", False, right_tool=True)]
    assert task_outcomes(scores) == {"t": False}
    assert task_outcomes(scores, "right_tool") == {"t": True}


def test_outcome_rejects_non_bool_metric():
    with pytest.raises(TypeError, match="calls"):
        task_outcomes([run("t", True)], "calls")


def test_rate_leaves_out_tasks_without_outcome():
    r = rate({"a": True, "b": False, "c": None, "d": True})
    assert (r.successes, r.n, r.value) == (2, 3, pytest.approx(2 / 3))
    assert (r.low, r.high) == pytest.approx(wilson_by_hand(2, 3))
    assert rate({"a": None}) == Rate(0, 0, None, None)


# --- exact McNemar ---


def outcomes(pattern: str) -> dict[str, bool | None]:
    """'1' solved, '0' not solved, '-' no outcome; task ids are positions."""
    return {f"t{i}": {"1": True, "0": False, "-": None}[c] for i, c in enumerate(pattern)}


def test_mcnemar_counts_and_exact_p_value():
    # 3 both, 1 only a, 6 only b, 2 neither: p = 2 * P(X <= 1 | n=7, 1/2) = 16/128
    a = outcomes("111" + "1" + "000000" + "00")
    b = outcomes("111" + "0" + "111111" + "00")
    p = compare(a, b)
    assert (p.n, p.only_a, p.only_b, p.unpaired) == (12, 1, 6, 0)
    assert p.p_value == pytest.approx(0.125)
    assert (p.a.successes, p.b.successes) == (4, 9)
    assert p.difference == pytest.approx(5 / 12)
    assert compare(b, a).p_value == pytest.approx(0.125)
    assert compare(b, a).difference == pytest.approx(-5 / 12)


@pytest.mark.parametrize(
    ("only_a", "only_b", "expected"),
    [  # two-sided exact binomial on the discordant pairs, by hand
        (0, 5, 2 / 32),
        (0, 6, 2 / 64),
        (2, 8, 2 * (1 + 10 + 45) / 1024),
        (5, 5, 1.0),
        (0, 1, 1.0),
    ],
)
def test_mcnemar_p_value_only_depends_on_discordant(only_a, only_b, expected):
    a = outcomes("1" * only_a + "0" * only_b + "1100")
    b = outcomes("0" * only_a + "1" * only_b + "1100")
    assert compare(a, b).p_value == pytest.approx(expected)


def test_mcnemar_no_discordant_pair():
    p = compare(outcomes("1100"), outcomes("1100"))
    assert (p.only_a, p.only_b, p.p_value, p.difference) == (0, 0, 1.0, 0.0)


def test_mcnemar_pairs_only_tasks_with_outcome_in_both():
    a = outcomes("1-01") | {"only-a": True}
    b = outcomes("0110")
    p = compare(a, b)
    assert p.n == 3  # t0, t2, t3; t1 has no outcome in a, only-a is not in b
    assert p.unpaired == 2
    assert (p.only_a, p.only_b) == (2, 1)
    assert (p.a.n, p.b.n) == (3, 3)


def test_mcnemar_empty():
    p = compare({}, outcomes("--"))
    assert (p.n, p.p_value, p.difference, p.a.value) == (0, 1.0, None, None)


def test_end_to_end_from_scores():
    # k=3 per task and variant, one invalid run; v2 fixes t1 and t2, breaks nothing
    v1 = [
        run(t, s)
        for t, runs in {"t0": "110", "t1": "100", "t2": "000"}.items()
        for s in map(bool, map(int, runs))
    ]
    v2 = [
        run(t, s)
        for t, runs in {"t0": "111", "t1": "110", "t2": "11"}.items()
        for s in map(bool, map(int, runs))
    ]
    v2.append(run("t2", None))
    p = compare(task_outcomes(v1), task_outcomes(v2))
    assert (p.n, p.only_a, p.only_b) == (3, 0, 2)
    assert p.p_value == pytest.approx(0.5)
    assert (p.a.value, p.b.value) == (pytest.approx(1 / 3), 1.0)
