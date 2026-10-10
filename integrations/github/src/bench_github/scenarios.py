"""Cassette scenarios for the GitHub client (`bench synth github`, later `bench record`).

Each scenario drives `GitHubClient` against the sandbox repository. Expected tool errors
are caught, so the cassette holds the error response the tests then replay. Scenarios
with `prepare` shape the fake and can only be synthesized.
"""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx2
from mcp.server.mcpserver.exceptions import ToolError

from bench_core.auth import StaticToken
from bench_core.http import ApiClient
from bench_core.testing import CassetteSuite, FakeBackend, FakeClock, Scenario
from bench_github.client import GitHubClient, make_api
from bench_github.fake import OWNER, REPO, GitHubFake, seeded_fake

CASSETTE_DIR = Path(__file__).resolve().parents[2] / "tests" / "cassettes"
SYNTHETIC_TOKEN = "synthetic-token"
FULL_ACCESS = ["issues:write", "pull_requests:read", "contents:read"]


def _fake(clock: FakeClock) -> FakeBackend:
    fake = seeded_fake(clock)
    fake.issue_token(SYNTHETIC_TOKEN, FULL_ACCESS)
    return fake


def _api(transport: httpx2.AsyncBaseTransport | None, clock: FakeClock | None) -> ApiClient:
    # The Authorization header is redacted in cassettes, so any token replays.
    return make_api(
        credential=StaticToken("github", SYNTHETIC_TOKEN), transport=transport, clock=clock
    )


async def _expect_error(call: Awaitable[Any]) -> None:
    try:
        await call
    except ToolError:
        return
    raise AssertionError("scenario expected a tool error")


Run = Callable[[GitHubClient], Awaitable[Any]]


def _scenario(run: Run, description: str, prepare: Callable[[GitHubFake], None] | None = None):
    async def drive(api: ApiClient) -> None:
        await run(GitHubClient(api))

    def shape(fake: FakeBackend) -> None:
        assert prepare is not None
        assert isinstance(fake, GitHubFake)
        prepare(fake)

    return Scenario(drive, shape if prepare else None, description)


# --- Scenarios ----------------------------------------------------------------------------

OPEN_BUGS = f"repo:{OWNER}/{REPO} is:open label:bug"


async def search_issues(gh: GitHubClient) -> None:
    first = await gh.search_issues(OPEN_BUGS, per_page=3)
    await gh.search_issues(OPEN_BUGS, per_page=3, cursor=first.next_cursor)


async def search_no_results(gh: GitHubClient) -> None:
    await gh.search_issues(f"repo:{OWNER}/{REPO} kubernetes")


async def search_unknown_user(gh: GitHubClient) -> None:
    await _expect_error(gh.search_issues(f"repo:{OWNER}/{REPO} author:no-such-user-4711"))


async def get_issue(gh: GitHubClient) -> None:
    await gh.get_issue(OWNER, REPO, 1)


async def get_issue_not_found(gh: GitHubClient) -> None:
    await _expect_error(gh.get_issue(OWNER, REPO, 999))


async def list_issue_comments(gh: GitHubClient) -> None:
    first = await gh.list_issue_comments(OWNER, REPO, 1, per_page=2)
    await gh.list_issue_comments(OWNER, REPO, 1, per_page=2, cursor=first.next_cursor)


async def create_issue(gh: GitHubClient) -> None:
    await gh.create_issue(
        OWNER,
        REPO,
        "Export button does nothing in Firefox",
        body="Clicking *Export* in Firefox 131 has no effect; Chrome works.",
        labels=["bug"],
    )


async def create_issue_forbidden(gh: GitHubClient) -> None:
    await _expect_error(gh.create_issue(OWNER, REPO, "Should not be created"))


def _read_only_token(fake: GitHubFake) -> None:
    fake.issue_token(SYNTHETIC_TOKEN, ["issues:read", "pull_requests:read", "contents:read"])


async def add_issue_comment(gh: GitHubClient) -> None:
    await gh.add_issue_comment(OWNER, REPO, 1, "Confirmed on the latest release as well.")


async def list_pull_requests(gh: GitHubClient) -> None:
    first = await gh.list_pull_requests(OWNER, REPO, state="all", per_page=3)
    await gh.list_pull_requests(OWNER, REPO, state="all", per_page=3, cursor=first.next_cursor)


async def get_file(gh: GitHubClient) -> None:
    await gh.get_file_contents(OWNER, REPO, "README.md")


async def get_file_with_space(gh: GitHubClient) -> None:
    await gh.get_file_contents(OWNER, REPO, "docs/user guide.md", ref="main")


async def get_directory(gh: GitHubClient) -> None:
    await gh.get_file_contents(OWNER, REPO, "")
    await gh.get_file_contents(OWNER, REPO, "src/widgets")


async def get_binary_file(gh: GitHubClient) -> None:
    await gh.get_file_contents(OWNER, REPO, "assets/logo.png")


async def get_large_file(gh: GitHubClient) -> None:
    await gh.get_file_contents(OWNER, REPO, "data/large.csv")


async def get_file_not_found(gh: GitHubClient) -> None:
    await _expect_error(gh.get_file_contents(OWNER, REPO, "docs/missing.md"))


SCENARIOS: dict[str, Scenario] = {
    "search_issues": _scenario(search_issues, "Open bugs, two pages"),
    "search_no_results": _scenario(search_no_results, "Search without hits"),
    "search_unknown_user": _scenario(search_unknown_user, "422 for an unknown author"),
    "get_issue": _scenario(get_issue, "Issue with labels, assignee and milestone"),
    "get_issue_not_found": _scenario(get_issue_not_found, "404 for a missing issue"),
    "list_issue_comments": _scenario(list_issue_comments, "Three comments, two pages"),
    "create_issue": _scenario(create_issue, "New issue with a label"),
    "create_issue_forbidden": _scenario(
        create_issue_forbidden, "403 without issues:write", prepare=_read_only_token
    ),
    "add_issue_comment": _scenario(add_issue_comment, "Comment on issue 1"),
    "list_pull_requests": _scenario(list_pull_requests, "All pull requests, two pages"),
    "get_file": _scenario(get_file, "README.md as text"),
    "get_file_with_space": _scenario(get_file_with_space, "Quoted path and ref"),
    "get_directory": _scenario(get_directory, "Root and nested directory listing"),
    "get_binary_file": _scenario(get_binary_file, "PNG file"),
    "get_large_file": _scenario(get_large_file, "File over 1 MB without content"),
    "get_file_not_found": _scenario(get_file_not_found, "404 for a missing path"),
}

SUITE = CassetteSuite(
    integration="github",
    cassette_dir=CASSETTE_DIR,
    scenarios=SCENARIOS,
    fake=_fake,
    api=_api,
)
