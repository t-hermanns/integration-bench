"""The GitHub fake behaves like GitHub where the client and the tools depend on it."""

from collections.abc import AsyncIterator

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from bench_core.auth import StaticToken
from bench_core.http import RateLimited
from bench_core.permissions import NotFoundOrNoAccess, PermissionDenied
from bench_core.testing import FakeClock, Fault
from bench_github.client import GitHubClient, make_api
from bench_github.fake import OWNER, REPO, GitHubFake, SearchError, parse_search, seeded_fake

pytestmark = pytest.mark.anyio

TOKEN = "fake-token"


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def fake(clock: FakeClock) -> GitHubFake:
    fake = seeded_fake(clock)
    fake.issue_token(TOKEN, ["issues:write", "pull_requests:read", "contents:read"])
    return fake


@pytest.fixture
async def gh(fake: GitHubFake, clock: FakeClock) -> AsyncIterator[GitHubClient]:
    credential = StaticToken("github", TOKEN)
    async with make_api(credential=credential, transport=fake.transport(), clock=clock) as api:
        yield GitHubClient(api)


async def test_write_permission_implies_read(gh: GitHubClient) -> None:
    assert (await gh.get_issue(OWNER, REPO, 1)).number == 1


async def test_unknown_token_gets_bad_credentials(fake: GitHubFake, clock: FakeClock) -> None:
    credential = StaticToken("github", "other")
    async with make_api(credential=credential, transport=fake.transport(), clock=clock) as api:
        with pytest.raises(ToolError, match=r"\(401\)"):
            await GitHubClient(api).get_issue(OWNER, REPO, 1)


async def test_created_issue_and_comment_are_visible(gh: GitHubClient, clock: FakeClock) -> None:
    created = await gh.create_issue(OWNER, REPO, "New bug", labels=["bug"])
    assert created.author == "bench-bot[bot]"
    assert created.created_at.timestamp() == clock.time()
    await gh.add_issue_comment(OWNER, REPO, created.number, "First!")
    issue = await gh.get_issue(OWNER, REPO, created.number)
    assert issue.comments == 1
    hits = await gh.search_issues(f"repo:{OWNER}/{REPO} New bug", per_page=100)
    assert [h.number for h in hits.items] == [created.number]


async def test_create_issue_rejects_unknown_assignee(gh: GitHubClient) -> None:
    with pytest.raises(ToolError, match="assignees invalid"):
        await gh.create_issue(OWNER, REPO, "x", assignees=["stranger"])


async def test_unknown_repo_is_404(gh: GitHubClient) -> None:
    with pytest.raises(NotFoundOrNoAccess):
        await gh.list_pull_requests(OWNER, "private-repo")


async def test_search_separates_issues_and_pull_requests(gh: GitHubClient) -> None:
    issues = await gh.search_issues(f"repo:{OWNER}/{REPO}", per_page=100)
    pulls = await gh.search_issues(f"repo:{OWNER}/{REPO} is:pr", per_page=100)
    assert issues.items
    assert pulls.items
    assert not any(i.is_pull_request for i in issues.items)
    assert all(p.is_pull_request for p in pulls.items)


async def test_search_qualifiers_and_negation(gh: GitHubClient) -> None:
    closed = await gh.search_issues(f"repo:{OWNER}/{REPO} is:closed", per_page=100)
    assert closed.items
    assert {i.state for i in closed.items} == {"closed"}
    not_bugs = await gh.search_issues(f"repo:{OWNER}/{REPO} -label:bug", per_page=100)
    assert not_bugs.items
    assert all("bug" not in i.labels for i in not_bugs.items)
    quoted = await gh.search_issues(f'repo:{OWNER}/{REPO} "unicode paths"', per_page=100)
    assert [i.title for i in quoted.items] == ["Crash on startup with Unicode paths"]


def test_parse_search_rejects_unknown_users() -> None:
    with pytest.raises(SearchError, match="cannot be searched"):
        parse_search("author:ghost", {"octocat"})
    parse_search("author:octocat", {"octocat"})


async def test_missing_permission_names_it(fake: GitHubFake, gh: GitHubClient) -> None:
    fake.issue_token(TOKEN, ["issues:read"])
    with pytest.raises(PermissionDenied) as exc:
        await gh.list_pull_requests(OWNER, REPO)
    assert exc.value.missing == frozenset({"pull_requests:read"})


async def test_primary_rate_limit_beyond_budget(fake: GitHubFake, gh: GitHubClient) -> None:
    # Reset is 60 s away, the default wait budget 30 s: fail fast with an actionable error.
    fake.inject(Fault.rate_limit(), path="/search/issues")
    with pytest.raises(RateLimited, match="retry after 61 s"):
        await gh.search_issues("bug")


async def test_secondary_rate_limit_is_retried(
    fake: GitHubFake, gh: GitHubClient, clock: FakeClock
) -> None:
    fake.inject(Fault.rate_limit(retry_after=5), path="/search/issues")
    page = await gh.search_issues(f"repo:{OWNER}/{REPO}")
    assert page.items
    assert clock.sleeps == [5]


async def test_server_error_shape(fake: GitHubFake, gh: GitHubClient) -> None:
    fake.inject(Fault.server_error(502))
    with pytest.raises(ToolError, match=r"server error \(502\)"):
        await gh.get_issue(OWNER, REPO, 1)
