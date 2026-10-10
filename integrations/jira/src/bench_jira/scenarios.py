"""Cassette scenarios: `bench synth jira` now, `bench record jira` later.

Each scenario drives `JiraClient` against the sandbox. Expected tool errors are caught,
so the cassette holds the error response the tests then replay. Scenarios with `prepare`
shape the fake and can only be synthesized.
"""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx2

from bench_core.http import ApiClient
from bench_core.testing import CassetteSuite, FakeBackend, FakeClock, Scenario, expect_tool_error

from bench_jira.client import JiraClient, make_api
from bench_jira.fake import JiraFake, credential, seeded_fake
from bench_jira.scopes import READ, WRITE

CASSETTE_DIR = Path(__file__).resolve().parents[2] / "tests" / "cassettes"
SYNTHETIC_TOKEN = "synthetic-token"
FULL_ACCESS = [READ, WRITE]


def _fake(clock: FakeClock) -> FakeBackend:
    fake = seeded_fake(clock)
    fake.issue_token(SYNTHETIC_TOKEN, FULL_ACCESS)
    return fake


def _api(transport: httpx2.AsyncBaseTransport | None, clock: FakeClock | None) -> ApiClient:
    # The Authorization header is redacted in cassettes, so any token replays.
    return make_api(credential=credential(SYNTHETIC_TOKEN), transport=transport, clock=clock)


Run = Callable[[JiraClient], Awaitable[Any]]


def _scenario(
    run: Run, description: str, prepare: Callable[[JiraFake], None] | None = None
) -> Scenario:
    async def drive(api: ApiClient) -> None:
        await run(JiraClient(api))

    def shape(fake: FakeBackend) -> None:
        assert prepare is not None
        assert isinstance(fake, JiraFake)
        prepare(fake)

    return Scenario(drive, shape if prepare else None, description)


# --- Scenarios ----------------------------------------------------------------------------

CRASHES = "project = BENCH AND summary ~ crash ORDER BY key ASC"
"""Matches three seeded issues (BENCH-1, -7, -12), so a page size of 2 gives two pages."""


async def search_issues(client: JiraClient) -> None:
    first = await client.search_issues(CRASHES, page_size=2)
    await client.search_issues(CRASHES, page_size=2, cursor=first.next_cursor)


async def search_no_results(client: JiraClient) -> None:
    await client.search_issues("project = BENCH AND summary ~ kubernetes")


async def search_invalid_jql(client: JiraClient) -> None:
    await expect_tool_error(client.search_issues("project = BENCH AND sprint_name = 4"))


async def get_issue(client: JiraClient) -> None:
    await client.get_issue("BENCH-1")


async def get_issue_not_found(client: JiraClient) -> None:
    await expect_tool_error(client.get_issue("BENCH-999"))


async def create_issue(client: JiraClient) -> None:
    await client.create_issue(
        "BENCH",
        "Export button does nothing in Firefox",
        issue_type="Bug",
        description="Chrome works fine.\n\n- Firefox 131\n- Windows 11",
        labels=["export", "firefox"],
    )


async def create_issue_invalid(client: JiraClient) -> None:
    await expect_tool_error(client.create_issue("BENCH", "Wrong type", issue_type="Bugfix"))


async def create_issue_forbidden(client: JiraClient) -> None:
    await expect_tool_error(client.create_issue("BENCH", "Should not be created"))


async def add_comment(client: JiraClient) -> None:
    await client.add_comment("BENCH-2", "Design draft is ready for review.")


async def transition_issue(client: JiraClient) -> None:
    """BENCH-4 to In Progress and back, so a live run leaves the sandbox as it was."""
    for target in ("In Progress", "To Do"):
        available = await client.transitions("BENCH-4")
        chosen = next(t for t in available if t.to_status == target)
        await client.transition_issue("BENCH-4", chosen.id)


async def transition_invalid(client: JiraClient) -> None:
    await expect_tool_error(client.transition_issue("BENCH-4", "999"))


def _read_only_token(fake: JiraFake) -> None:
    fake.issue_token(SYNTHETIC_TOKEN, [READ])


SCENARIOS: dict[str, Scenario] = {
    "search_issues": _scenario(search_issues, "Summary contains 'crash', two pages"),
    "search_no_results": _scenario(search_no_results, "Search without hits"),
    "search_invalid_jql": _scenario(search_invalid_jql, "JQL with an unknown field"),
    "get_issue": _scenario(get_issue, "BENCH-1 with description and comments"),
    "get_issue_not_found": _scenario(get_issue_not_found, "Missing issue"),
    "create_issue": _scenario(create_issue, "New bug with description and labels"),
    "create_issue_invalid": _scenario(create_issue_invalid, "Unknown issue type"),
    "create_issue_forbidden": _scenario(
        create_issue_forbidden, "OAuth token without write:jira-work", prepare=_read_only_token
    ),
    "add_comment": _scenario(add_comment, "Plain-text comment on BENCH-2"),
    "transition_issue": _scenario(transition_issue, "BENCH-4 to In Progress and back"),
    "transition_invalid": _scenario(transition_invalid, "Transition id that does not exist"),
}

SUITE = CassetteSuite(
    integration="jira",
    cassette_dir=CASSETTE_DIR,
    scenarios=SCENARIOS,
    fake=_fake,
    api=_api,
)
