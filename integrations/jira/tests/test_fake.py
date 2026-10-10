"""The fake's JQL subset and Jira-like behaviour, directly against the fake."""

import re

import httpx2
import pytest

from bench_core.testing import FakeClock

from bench_jira.client import BASE_URL
from bench_jira.fake import JiraFake, seeded_fake
from bench_jira.fake.jql import JqlError, parse


@pytest.fixture
def fake() -> JiraFake:
    return seeded_fake(FakeClock(), require_auth=False)


def keys(fake: JiraFake, query: str) -> list[str]:
    parsed = parse(query, fake.context())
    hits = parsed.sort([i for i in fake.issues.values() if parsed.matches(i)])
    return [i["key"] for i in hits]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("key = bench-3", ["BENCH-3"]),
        ("issuekey in (BENCH-1, BENCH-2) ORDER BY key", ["BENCH-1", "BENCH-2"]),
        (
            "project = 'Bench Sandbox' AND status = done ORDER BY key",
            ["BENCH-5", "BENCH-6", "BENCH-9"],
        ),
        ("status not in ('To Do', Done) ORDER BY key", ["BENCH-3", "BENCH-8", "BENCH-12"]),
        ("statusCategory = indeterminate ORDER BY key", ["BENCH-3", "BENCH-8", "BENCH-12"]),
        ("assignee = 'Bench Bot' AND NOT status = Done ORDER BY key", ["BENCH-8", "BENCH-10"]),
        ("labels = crash AND (priority = High OR issuetype = Story) ORDER BY key", ["BENCH-7"]),
        ("summary ~ 'csv' ORDER BY key", ["BENCH-3", "BENCH-11"]),
        ('summary ~ "export*" ORDER BY key', ["BENCH-3", "BENCH-9"]),
        ("comment ~ streaming", ["BENCH-3"]),
        (
            "description !~ crash AND labels = crash ORDER BY key",
            ["BENCH-1", "BENCH-7", "BENCH-12"],
        ),
        (
            "created >= '2026-09-03' ORDER BY created ASC",
            ["BENCH-9", "BENCH-10", "BENCH-11", "BENCH-12"],
        ),
        ("created < '2026/09/01 02:00' OR created = '2026-09-01 09:00'", ["BENCH-1"]),
    ],
)
def test_queries(fake: JiraFake, query: str, expected: list[str]) -> None:
    assert keys(fake, query) == expected


def test_order_by_several_fields(fake: JiraFake) -> None:
    ordered = keys(fake, "project = BENCH ORDER BY priority DESC, key DESC")
    assert ordered[:3] == ["BENCH-12", "BENCH-1", "BENCH-7"]


def test_negations_skip_empty_fields(fake: JiraFake) -> None:
    assert keys(fake, "project = BENCH AND assignee is EMPTY")
    assert keys(fake, "assignee is not EMPTY") == keys(fake, "assignee = 'Bench Bot'")
    assert keys(fake, "assignee != 'Bench Bot'") == []


def test_relative_dates_use_the_fake_clock(fake: JiraFake) -> None:
    assert keys(fake, "created >= -1d") == []
    fake.add_issue("Fresh")
    assert keys(fake, "created >= -1d") == ["BENCH-13"]
    assert keys(fake, "updated > -2w AND summary ~ fresh") == ["BENCH-13"]


def test_default_order_is_newest_first(fake: JiraFake) -> None:
    assert keys(fake, "project = BENCH")[:2] == ["BENCH-12", "BENCH-11"]


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ("", "Unbounded JQL queries are not allowed"),
        ("ORDER BY key", "Unbounded JQL queries are not allowed"),
        ("status = Done AND", "Expecting a field name at the end of the query."),
        ("summary ~ ", "Error in JQL Query: Expecting either a value, list or function"),
        ("status Done", "Expecting operator but got 'Done'"),
        ("status = Done project = BENCH", "Expecting either 'OR' or 'AND'"),
        ("(status = Done", "Expecting ')' before the end of the query."),
        ("labels is foo", "Your query couldn't be processed."),
    ],
)
def test_syntax_errors_read_like_jira(fake: JiraFake, query: str, message: str) -> None:
    with pytest.raises(JqlError, match=re.escape(message)):
        parse(query, fake.context())


@pytest.mark.parametrize(
    "query",
    [
        "sprint = 1",
        "status = Closed",
        "project = NOPE",
        "summary = crash",
        "status ~ done",
        "priority > Lowish",
        "assignee = me()",
        "created > yesterday",
        "sprint is EMPTY",
    ],
)
def test_clauses_that_cannot_match_find_nothing(fake: JiraFake, query: str) -> None:
    """Live /search/jql answers these with 200 and no issues, not with an error."""
    assert keys(fake, query) == []


def test_unknown_sort_field_is_ignored(fake: JiraFake) -> None:
    assert keys(fake, "status = Done ORDER BY rank") == keys(fake, "status = Done")


def test_priority_compares_by_rank(fake: JiraFake) -> None:
    high = set(keys(fake, "priority > Medium"))
    assert high
    assert high == set(keys(fake, "priority in (High, Highest)"))


@pytest.mark.anyio
async def test_search_returns_only_requested_fields(fake: JiraFake) -> None:
    async with httpx2.AsyncClient(transport=fake.transport()) as http:
        response = await http.get(f"{BASE_URL}/search/jql", params={"jql": "key = BENCH-1"})
    issue = response.json()["issues"][0]
    assert (issue["key"], issue["fields"]) == ("BENCH-1", {})
    assert response.json()["isLast"] is True
