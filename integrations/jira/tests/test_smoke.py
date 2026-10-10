"""`bench smoke jira` offline: the smoke calls through the MCP server, against the fake."""

from collections.abc import AsyncIterator

import pytest

from bench_core.http import ApiClient
from bench_core.sandbox import SmokeCall, smoke
from bench_core.testing import FakeClock

from bench_jira.client import make_api
from bench_jira.fake import credential, seeded_fake
from bench_jira.sandbox import SMOKE
from bench_jira.scopes import READ, WRITE
from bench_jira.server import create_server

pytestmark = pytest.mark.anyio

TOKEN = "smoke-token"


@pytest.fixture
async def api() -> AsyncIterator[ApiClient]:
    clock = FakeClock()
    fake = seeded_fake(clock)
    fake.issue_token(TOKEN, [READ, WRITE])
    async with make_api(
        credential=credential(TOKEN), transport=fake.transport(), clock=clock
    ) as api:
        yield api


async def run(api: ApiClient, calls: list[SmokeCall]) -> tuple[str | None, list[str]]:
    lines: list[str] = []
    failure = await smoke(create_server(api=api, read_only=True, variant="v1"), calls, lines.append)
    return failure, lines


async def test_the_smoke_calls_pass_against_the_seeded_fake(api: ApiClient) -> None:
    failure, lines = await run(api, SMOKE)
    assert failure is None
    assert lines == ["2 tools listed", "ok  jira_search_issues", "ok  jira_get_issue"]


async def test_the_caller_owned_client_outlives_the_session(api: ApiClient) -> None:
    assert (await run(api, SMOKE))[0] is None
    assert (await run(api, SMOKE))[0] is None


async def test_write_tools_are_filtered_out(api: ApiClient) -> None:
    failure, _ = await run(api, [SmokeCall("jira_create_issue", {})])
    assert failure == "jira_create_issue: not listed (read-only filter or descriptions?)"


async def test_a_tool_error_stops_the_run(api: ApiClient) -> None:
    failure, lines = await run(
        api, [SmokeCall("jira_get_issue", {"issue_key": "BENCH-999"}), *SMOKE]
    )
    assert failure is not None
    assert failure.startswith("jira_get_issue: ")
    assert lines == ["2 tools listed"]


async def test_args_from_an_empty_result(api: ApiClient) -> None:
    calls = [
        SmokeCall("jira_search_issues", {"jql": "project = BENCH AND labels = none"}),
        SMOKE[1],
    ]
    failure, _ = await run(api, calls)
    assert failure == "jira_get_issue: previous result lacks IndexError('list index out of range')"
