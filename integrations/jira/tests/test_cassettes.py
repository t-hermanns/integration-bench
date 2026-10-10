"""Client against cassettes: synthetic now, recorded from the sandbox later.

Assertions only use facts the sandbox seed guarantees, so they hold for both kinds.
"""

from collections.abc import AsyncIterator

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from vcr.errors import CannotOverwriteExistingCassetteException

from bench_core.permissions import NotFoundOrNoAccess, PermissionDenied
from bench_core.testing import read_manifest, stale_cassettes

from bench_jira.client import JiraClient
from bench_jira.scenarios import CRASHES, SCENARIOS, SUITE

pytestmark = pytest.mark.anyio


@pytest.fixture
async def client() -> AsyncIterator[JiraClient]:
    async with SUITE.replay_api() as api:
        yield JiraClient(api)


def test_manifest_covers_every_scenario() -> None:
    manifest = read_manifest(SUITE.cassette_dir)
    assert set(manifest) == set(SCENARIOS), "run `bench synth jira`"
    assert set(manifest.values()) <= {"synthetic", "recorded"}
    for name in SCENARIOS:
        assert SUITE.path(name).exists(), name


async def test_synthetic_cassettes_are_up_to_date(tmp_path) -> None:
    assert await stale_cassettes(SUITE, tmp_path) == [], "run `bench synth jira`"


async def test_search_issues(client: JiraClient) -> None:
    with SUITE.replay("search_issues") as cassette:
        first = await client.search_issues(CRASHES, page_size=2)
        second = await client.search_issues(CRASHES, page_size=2, cursor=first.next_cursor)
    assert cassette.all_played
    assert len(first.items) == 2
    assert first.next_cursor is not None
    assert second.next_cursor is None
    issues = first.items + second.items
    assert [i.key for i in issues] == ["BENCH-1", "BENCH-7", "BENCH-12"]
    assert all("crash" in i.summary.lower() for i in issues)
    first_hit = issues[0]
    assert (first_hit.issue_type, first_hit.priority) == ("Bug", "Highest")
    assert first_hit.labels == ["checkout", "crash"]


async def test_search_no_results(client: JiraClient) -> None:
    with SUITE.replay("search_no_results"):
        page = await client.search_issues("project = BENCH AND summary ~ kubernetes")
    assert (page.items, page.next_cursor) == ([], None)


async def test_search_invalid_jql(client: JiraClient) -> None:
    with (
        SUITE.replay("search_invalid_jql"),
        pytest.raises(ToolError, match=r"sprint_name.*Fix the arguments"),
    ):
        await client.search_issues("project = BENCH AND sprint_name = 4")


async def test_get_issue(client: JiraClient) -> None:
    with SUITE.replay("get_issue"):
        issue = await client.get_issue("BENCH-1")
    assert issue.summary == "Checkout crashes when the cart is empty"
    assert (issue.status, issue.status_category) == ("To Do", "new")
    assert issue.project == "BENCH"
    assert issue.assignee == "Alex Doe"
    assert issue.description is not None
    assert "1. Open the shop" in issue.description
    assert "KeyError: 'total'" in issue.description
    assert issue.comment_total == len(issue.comments) == 2
    assert issue.comments[0].author == "Sam Lee"


async def test_get_issue_not_found(client: JiraClient) -> None:
    with SUITE.replay("get_issue_not_found"), pytest.raises(NotFoundOrNoAccess):
        await client.get_issue("BENCH-999")


async def test_create_issue(client: JiraClient) -> None:
    with SUITE.replay("create_issue"):
        created = await client.create_issue(
            "BENCH",
            "Export button does nothing in Firefox",
            issue_type="Bug",
            description="Chrome works fine.\n\n- Firefox 131\n- Windows 11",
            labels=["export", "firefox"],
        )
    assert created.key.startswith("BENCH-")


async def test_create_issue_invalid(client: JiraClient) -> None:
    with (
        SUITE.replay("create_issue_invalid"),
        pytest.raises(ToolError, match=r"issuetype: .*Fix the arguments"),
    ):
        await client.create_issue("BENCH", "Wrong type", issue_type="Bugfix")


async def test_create_issue_forbidden(client: JiraClient) -> None:
    with (
        SUITE.replay("create_issue_forbidden") as cassette,
        pytest.raises(PermissionDenied, match="OAuth scope"),
    ):
        await client.create_issue("BENCH", "Should not be created")
    assert cassette.play_count == 1, "a scope mismatch is not retried with a new token"


async def test_add_comment(client: JiraClient) -> None:
    with SUITE.replay("add_comment"):
        comment = await client.add_comment("BENCH-2", "Design draft is ready for review.")
    assert comment.body == "Design draft is ready for review."
    assert comment.id


async def test_transition_issue(client: JiraClient) -> None:
    with SUITE.replay("transition_issue") as cassette:
        for target in ("In Progress", "To Do"):
            available = await client.transitions("BENCH-4")
            assert target in {t.to_status for t in available}
            chosen = next(t for t in available if t.to_status == target)
            await client.transition_issue("BENCH-4", chosen.id)
    assert cassette.all_played


async def test_transition_invalid(client: JiraClient) -> None:
    with (
        SUITE.replay("transition_invalid"),
        pytest.raises(ToolError, match=r"not valid.*Fix the arguments"),
    ):
        await client.transition_issue("BENCH-4", "999")


async def test_unrecorded_request_fails_instead_of_going_out(client: JiraClient) -> None:
    with (
        SUITE.replay("get_issue"),
        pytest.raises(CannotOverwriteExistingCassetteException),
    ):
        await client.get_issue("BENCH-2")
