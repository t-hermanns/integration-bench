"""GitHub client against cassettes (synthetic now, recorded from the sandbox later).

Assertions only use facts the sandbox seed guarantees, so they hold for both kinds.
"""

from collections.abc import AsyncIterator

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from vcr.errors import CannotOverwriteExistingCassetteException

from bench_core.permissions import NotFoundOrNoAccess, PermissionDenied
from bench_core.testing import read_manifest, stale_cassettes
from bench_github.client import GitHubClient
from bench_github.fake import OWNER, REPO
from bench_github.models import DirectoryListing, FileContent
from bench_github.scenarios import OPEN_BUGS, SCENARIOS, SUITE

pytestmark = pytest.mark.anyio


@pytest.fixture
async def gh() -> AsyncIterator[GitHubClient]:
    async with SUITE.replay_api() as api:
        yield GitHubClient(api)


def test_manifest_covers_every_scenario() -> None:
    manifest = read_manifest(SUITE.cassette_dir)
    assert set(manifest) == set(SCENARIOS)
    assert set(manifest.values()) <= {"synthetic", "recorded"}
    for name in SCENARIOS:
        assert SUITE.path(name).exists(), name


async def test_synthetic_cassettes_are_up_to_date(tmp_path) -> None:
    assert await stale_cassettes(SUITE, tmp_path) == [], "run `bench synth github`"


async def test_search_issues(gh: GitHubClient) -> None:
    with SUITE.replay("search_issues") as cassette:
        first = await gh.search_issues(OPEN_BUGS, per_page=3)
        second = await gh.search_issues(OPEN_BUGS, per_page=3, cursor=first.next_cursor)
    assert cassette.all_played
    assert first.total_count == len(first.items) + len(second.items)
    assert len(first.items) == 3
    assert second.next_cursor is None
    hits = first.items + second.items
    assert len({i.number for i in hits}) == len(hits)
    for issue in hits:
        assert issue.repository == f"{OWNER}/{REPO}"
        assert issue.state == "open"
        assert "bug" in issue.labels
        assert not issue.is_pull_request
    assert any(i.title == "Crash when config file is empty" for i in hits)


async def test_search_no_results(gh: GitHubClient) -> None:
    with SUITE.replay("search_no_results"):
        page = await gh.search_issues(f"repo:{OWNER}/{REPO} kubernetes")
    assert (page.items, page.total_count, page.next_cursor) == ([], 0, None)


async def test_search_unknown_user(gh: GitHubClient) -> None:
    with (
        SUITE.replay("search_unknown_user"),
        pytest.raises(ToolError, match=r"cannot be searched.*Fix the arguments"),
    ):
        await gh.search_issues(f"repo:{OWNER}/{REPO} author:no-such-user-4711")


async def test_get_issue(gh: GitHubClient) -> None:
    with SUITE.replay("get_issue"):
        issue = await gh.get_issue(OWNER, REPO, 1)
    assert issue.title == "Crash when config file is empty"
    assert issue.labels == ["bug"]
    assert issue.state == "open"
    assert issue.comments == 3
    assert issue.body is not None
    assert "KeyError" in issue.body
    assert issue.assignees
    assert issue.milestone is not None


async def test_get_issue_not_found(gh: GitHubClient) -> None:
    with SUITE.replay("get_issue_not_found"), pytest.raises(NotFoundOrNoAccess):
        await gh.get_issue(OWNER, REPO, 999)


async def test_list_issue_comments(gh: GitHubClient) -> None:
    with SUITE.replay("list_issue_comments"):
        first = await gh.list_issue_comments(OWNER, REPO, 1, per_page=2)
        second = await gh.list_issue_comments(OWNER, REPO, 1, per_page=2, cursor=first.next_cursor)
    comments = first.items + second.items
    assert [len(first.items), len(second.items)] == [2, 1]
    assert second.next_cursor is None
    assert comments[0].body == "I can reproduce this on 2.3.1."
    assert [c.created_at for c in comments] == sorted(c.created_at for c in comments)


async def test_create_issue(gh: GitHubClient) -> None:
    with SUITE.replay("create_issue"):
        issue = await gh.create_issue(
            OWNER,
            REPO,
            "Export button does nothing in Firefox",
            body="Clicking *Export* in Firefox 131 has no effect; Chrome works.",
            labels=["bug"],
        )
    assert issue.title == "Export button does nothing in Firefox"
    assert issue.state == "open"
    assert issue.labels == ["bug"]
    assert issue.number > 0


async def test_create_issue_forbidden(gh: GitHubClient) -> None:
    with (
        SUITE.replay("create_issue_forbidden"),
        pytest.raises(PermissionDenied, match="issues:write") as exc,
    ):
        await gh.create_issue(OWNER, REPO, "Should not be created")
    assert exc.value.missing == frozenset({"issues:write"})


async def test_add_issue_comment(gh: GitHubClient) -> None:
    with SUITE.replay("add_issue_comment"):
        comment = await gh.add_issue_comment(
            OWNER, REPO, 1, "Confirmed on the latest release as well."
        )
    assert comment.body == "Confirmed on the latest release as well."
    assert comment.html_url.endswith(f"#issuecomment-{comment.id}")


async def test_list_pull_requests(gh: GitHubClient) -> None:
    with SUITE.replay("list_pull_requests"):
        first = await gh.list_pull_requests(OWNER, REPO, state="all", per_page=3)
        second = await gh.list_pull_requests(
            OWNER, REPO, state="all", per_page=3, cursor=first.next_cursor
        )
    pulls = first.items + second.items
    assert len(first.items) == 3
    assert second.next_cursor is None
    assert {p.state for p in pulls} == {"open", "closed"}
    assert any(p.merged_at is not None for p in pulls)
    assert any(p.draft for p in pulls)
    updated = [p.updated_at for p in pulls]
    assert updated == sorted(updated, reverse=True)
    assert all(p.base == "main" for p in pulls)


async def test_get_file(gh: GitHubClient) -> None:
    with SUITE.replay("get_file"):
        result = await gh.get_file_contents(OWNER, REPO, "README.md")
    assert isinstance(result, FileContent)
    assert result.text is not None
    assert result.text.startswith("# widgets")
    assert result.size == len(result.text.encode())


async def test_get_file_with_space(gh: GitHubClient) -> None:
    with SUITE.replay("get_file_with_space"):
        result = await gh.get_file_contents(OWNER, REPO, "docs/user guide.md", ref="main")
    assert isinstance(result, FileContent)
    assert result.path == "docs/user guide.md"
    assert result.text is not None
    assert result.text.startswith("# User guide")


async def test_get_directory(gh: GitHubClient) -> None:
    with SUITE.replay("get_directory"):
        root = await gh.get_file_contents(OWNER, REPO, "")
        nested = await gh.get_file_contents(OWNER, REPO, "src/widgets")
    assert isinstance(root, DirectoryListing)
    assert {"README.md", "src", "docs"} <= {e.name for e in root.entries}
    assert {e.name: e.type for e in root.entries}["src"] == "dir"
    assert isinstance(nested, DirectoryListing)
    assert {e.path for e in nested.entries} == {"src/widgets/__init__.py", "src/widgets/core.py"}


async def test_get_binary_file(gh: GitHubClient) -> None:
    with SUITE.replay("get_binary_file"):
        result = await gh.get_file_contents(OWNER, REPO, "assets/logo.png")
    assert isinstance(result, FileContent)
    assert result.text is None
    assert result.note is not None
    assert "Binary" in result.note


async def test_get_large_file(gh: GitHubClient) -> None:
    with SUITE.replay("get_large_file"):
        result = await gh.get_file_contents(OWNER, REPO, "data/large.csv")
    assert isinstance(result, FileContent)
    assert result.text is None
    assert result.size > 1_000_000
    assert result.note is not None
    assert "too large" in result.note


async def test_get_file_not_found(gh: GitHubClient) -> None:
    with (
        SUITE.replay("get_file_not_found"),
        pytest.raises(NotFoundOrNoAccess, match=r"docs/missing\.md"),
    ):
        await gh.get_file_contents(OWNER, REPO, "docs/missing.md")


async def test_unrecorded_request_fails_instead_of_going_out(gh: GitHubClient) -> None:
    with SUITE.replay("get_issue"), pytest.raises(CannotOverwriteExistingCassetteException):
        await gh.get_issue(OWNER, REPO, 2)
