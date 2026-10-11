"""Auth and retry scenarios against the in-memory fake.

Runs over `httpx2.MockTransport` and never inside a cassette (vcrpy patches MockTransport
too). Clocks are fake, so waits are recorded instead of slept.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from mcp.client import Client
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent

from bench_core.auth import Credential
from bench_core.http import RateLimited
from bench_core.testing import FakeClock, Fault

from bench_todoist.auth import granted_scopes
from bench_todoist.client import PROVIDER, TodoistClient, make_api
from bench_todoist.fake import FIRST_ID, TodoistFake, credential, seeded_fake
from bench_todoist.scopes import READ, WRITE
from bench_todoist.server import create_server
from bench_todoist.tools import TOOL_WAIT_BUDGET

pytestmark = pytest.mark.anyio

TOKEN = "retry-token"
ALL = [READ, WRITE]
GET = "todoist_get_task"
CREATE = "todoist_create_task"


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def fake(clock: FakeClock) -> TodoistFake:
    fake = seeded_fake(clock)
    fake.issue_token(TOKEN, ALL)
    return fake


@pytest.fixture
async def client_for(fake: TodoistFake, clock: FakeClock) -> AsyncIterator[Any]:
    apis = []

    def make(cred: Credential) -> TodoistClient:
        api = make_api(credential=cred, transport=fake.transport(), clock=clock)
        apis.append(api)
        return TodoistClient(api)

    yield make
    for api in apis:
        await api.aclose()


@pytest.fixture
def account(request: pytest.FixtureRequest) -> str:
    # Refresh locks are per account and process-wide; keep tests independent.
    return f"todoist-{request.node.name}"


def text(result: CallToolResult) -> str:
    return "\n".join(c.text for c in result.content if isinstance(c, TextContent))


# --- Static credentials -------------------------------------------------------------------


async def test_revoked_token_is_reported(fake: TodoistFake, client_for: Any) -> None:
    fake.expire_token(TOKEN)
    with pytest.raises(ToolError) as exc:
        await client_for(credential(TOKEN)).get_task(FIRST_ID)
    assert "rejected" in str(exc.value)


async def test_static_tokens_report_no_scopes() -> None:
    assert await granted_scopes(credential(TOKEN)) is None


# --- Rate limits and network errors -------------------------------------------------------


async def test_rate_limit_honours_retry_after(
    fake: TodoistFake, clock: FakeClock, client_for: Any
) -> None:
    fake.inject(Fault.rate_limit(retry_after=7), method="GET")
    assert (await client_for(credential(TOKEN)).get_task(FIRST_ID)).id == FIRST_ID
    assert clock.sleeps == [7]


async def test_budget_exceeded_on_the_client(
    fake: TodoistFake, clock: FakeClock, client_for: Any
) -> None:
    fake.inject(Fault.rate_limit(retry_after=45), method="GET")
    with pytest.raises(RateLimited) as exc:
        await client_for(credential(TOKEN)).get_task(FIRST_ID)
    assert exc.value.retry_after == 45
    assert clock.sleeps == []


async def test_wait_budget_is_per_tool_call(fake: TodoistFake, clock: FakeClock) -> None:
    server = create_server(
        credential=credential(TOKEN),
        transport=fake.transport(),
        clock=clock,
        read_only=False,
        variant="v1",
    )
    fake.inject(Fault.rate_limit(retry_after=TOOL_WAIT_BUDGET + 1), method="GET")
    async with Client(server) as client:
        result = await client.call_tool(GET, {"task_id": FIRST_ID})
        assert result.is_error
        assert f"{PROVIDER} rate limited" in text(result)

        # The next tool call starts with a fresh budget.
        fake.inject(Fault.rate_limit(retry_after=TOOL_WAIT_BUDGET - 1), method="GET")
        assert not (await client.call_tool(GET, {"task_id": FIRST_ID})).is_error
    assert clock.sleeps == [TOOL_WAIT_BUDGET - 1]


async def test_network_errors_on_writes(fake: TodoistFake, clock: FakeClock) -> None:
    server = create_server(
        credential=credential(TOKEN),
        transport=fake.transport(),
        clock=clock,
        read_only=False,
        variant="v1",
    )
    before = len(fake.tasks)
    fake.inject(Fault.network(), method="POST")
    async with Client(server) as client:
        result = await client.call_tool(CREATE, {"title": "x"})
    assert result.is_error
    assert f"Could not reach {PROVIDER} (ConnectError)" in text(result)
    assert len(fake.tasks) == before
