import pytest

from bench_evals.matchers import MISSING, args_match, check_matcher, fill, matches


@pytest.mark.parametrize(
    ("matcher", "value", "expected"),
    [
        ("BENCH-1", "bench-1 ", True),
        ("BENCH-1", "BENCH-12", False),
        ("5", 5, False),  # a string matcher wants a string
        (5, 5, True),
        (5, 5.0, True),
        (5, "5", True),  # the server coerces it, so the call works
        (5, "five", False),
        (5, True, False),  # bool is an int subclass, but not a number here
        (True, True, True),
        (True, 1, False),
        (False, "false", True),
        (True, "yes", False),
        (None, MISSING, True),
        (None, None, True),
        (None, "x", False),
        ("x", MISSING, False),
        (["bug", "ui"], ["UI", "bug"], True),
        (["bug", "ui"], ["bug"], False),
        (["bug", "bug"], ["bug", "ui"], False),  # each item matches once
        (["bug"], "bug", False),
        ({"$contains": "status"}, "project = BENCH AND Status = Done", True),
        ({"$contains": "bug"}, ["ui", "Bug"], True),
        ({"$contains": "bug"}, MISSING, False),
        ({"$regex": r"status\s*=\s*\"?done"}, 'status = "Done"', True),
        ({"$regex": "^done$"}, "not done", False),
        ({"$any_of": ["open", "all"]}, "ALL", True),
        ({"$any_of": [None, "concise"]}, MISSING, True),
        ({"$any_of": [None, "concise"]}, "detailed", False),
        (
            {"$all": [{"$contains": "BENCH"}, {"$contains": "bug"}]},
            "project=BENCH AND type=Bug",
            True,
        ),
        ({"$all": [{"$contains": "BENCH"}, {"$contains": "bug"}]}, "project=BENCH", False),
        ({"$present": True}, "abc", True),
        ({"$present": True}, MISSING, False),
        ({"$present": True}, None, False),
        ({"$present": False}, MISSING, True),
        ({"$present": True, "$contains": "a"}, "b", False),  # operators are ANDed
    ],
)
def test_matches(matcher, value, expected):
    check_matcher(matcher)
    assert matches(matcher, value) is expected


def test_args_match_checks_only_named_arguments_and_requires_them():
    actual = {"jql": "project = BENCH", "page_size": 5, "response_format": "concise"}
    assert args_match({"jql": {"$contains": "bench"}}, actual)
    assert args_match({"page_size": 5, "cursor": None}, actual)
    assert not args_match({"cursor": {"$present": True}}, actual)
    assert not args_match({"labels": ["x"]}, actual)
    assert not args_match({"jql": "x"}, {"_unparsed": "{jql: x"})


@pytest.mark.parametrize(
    "matcher",
    [
        {},
        {"$contain": "x"},
        {"$contains": 1},
        {"$regex": "("},
        {"$any_of": []},
        {"$all": "x"},
        {"$present": "yes"},
        {"$any_of": [{"$nope": 1}]},
        [{"$regex": "["}],
        {"nested": "object"},
    ],
)
def test_check_matcher_rejects(matcher):
    with pytest.raises(ValueError, match=r"matcher|operator|takes|regex"):
        check_matcher(matcher)


def test_fill_replaces_the_placeholder_everywhere():
    matcher = {"$all": [{"$contains": "{run_id}"}, "x {run_id}"]}
    assert fill(matcher, "{run_id}", "r1") == {"$all": [{"$contains": "r1"}, "x r1"]}
    assert fill(5, "{run_id}", "r1") == 5
