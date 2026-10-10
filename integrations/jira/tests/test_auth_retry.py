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

from bench_core.auth import AuthError, Credential, MemoryTokenStore, TokenSet
from bench_core.http import RateLimited
from bench_core.permissions import PermissionDenied
from bench_core.testing import FakeClock, Fault

from bench_jira.auth import api_token, granted_scopes, user_token
from bench_jira.client import PROVIDER, JiraClient, make_api
from bench_jira.fake import JiraFake, credential, seeded_fake
from bench_jira.fake.backend import TOKEN_PATH
from bench_jira.scopes import READ, WRITE
from bench_jira.server import create_server
from bench_jira.tools import TOOL_WAIT_BUDGET

pytestmark = pytest.mark.anyio

TOKEN = "retry-token"
ALL = [READ, WRITE]
GET = "jira_get_issue"
CREATE = "jira_create_issue"
CLIENT_ID = "fake-client"
CLIENT_SECRET = "fake-client-secret"


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def fake(clock: FakeClock) -> JiraFake:
    fake = seeded_fake(clock)
    fake.issue_token(TOKEN, ALL)
    fake.oauth_clients[CLIENT_ID] = CLIENT_SECRET
    return fake


@pytest.fixture
async def client_for(fake: JiraFake, clock: FakeClock) -> AsyncIterator[Any]:
    apis = []

    def make(cred: Credential) -> JiraClient:
        api = make_api(credential=cred, transport=fake.transport(), clock=clock)
        apis.append(api)
        return JiraClient(api)

    yield make
    for api in apis:
        await api.aclose()


@pytest.fixture
def account(request: pytest.FixtureRequest) -> str:
    # Refresh locks are per account and process-wide; keep tests independent.
    return f"jira-{request.node.name}"


def text(result: CallToolResult) -> str:
    return "\n".join(c.text for c in result.content if isinstance(c, TextContent))


# --- User tokens (rotating refresh) -------------------------------------------------------


def login(
    fake: JiraFake, clock: FakeClock, account: str, *, expires_in: float = 3600
) -> MemoryTokenStore:
    """State right after `bench auth login jira`."""
    access, refresh = fake.user_tokens(ALL)
    tokens = TokenSet(access, refresh, expires_at=clock.time() + expires_in, scopes=frozenset(ALL))
    return MemoryTokenStore({account: tokens})


def user_credential(
    fake: JiraFake, clock: FakeClock, account: str, store: MemoryTokenStore
) -> Credential:
    return user_token(
        account,
        store,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        transport=fake.transport(),
        clock=clock,
    )


def refreshes(fake: JiraFake) -> int:
    return len(fake.calls("POST", TOKEN_PATH))


async def test_expiring_token_is_refreshed_before_the_call(
    fake: JiraFake, clock: FakeClock, account: str, client_for: Any
) -> None:
    store = login(fake, clock, account, expires_in=30)  # inside the 60 s skew
    old = store.load(account)
    client = client_for(user_credential(fake, clock, account, store))
    assert (await client.get_issue("BENCH-1")).key == "BENCH-1"
    assert refreshes(fake) == 1
    new = store.load(account)
    assert new is not None
    assert old is not None
    assert new.refresh_token != old.refresh_token, "rotated refresh token is stored"


async def test_401_refreshes_and_retries_once(
    fake: JiraFake, clock: FakeClock, account: str, client_for: Any
) -> None:
    store = login(fake, clock, account)
    tokens = store.load(account)
    assert tokens is not None
    assert tokens.access_token is not None
    fake.expire_token(tokens.access_token)
    client = client_for(user_credential(fake, clock, account, store))
    assert (await client.get_issue("BENCH-1")).key == "BENCH-1"
    assert refreshes(fake) == 1


async def test_used_refresh_token_asks_for_login(
    fake: JiraFake, clock: FakeClock, account: str, client_for: Any
) -> None:
    store = login(fake, clock, account, expires_in=0)
    fake.refresh_tokens.clear()  # e.g. another process already rotated it
    client = client_for(user_credential(fake, clock, account, store))
    with pytest.raises(AuthError, match=r"bench auth login"):
        await client.get_issue("BENCH-1")


async def test_granted_scopes_come_from_the_token(
    fake: JiraFake, clock: FakeClock, account: str
) -> None:
    store = login(fake, clock, account, expires_in=0)
    assert await granted_scopes(user_credential(fake, clock, account, store)) == frozenset(ALL)


async def test_missing_scope_is_final_without_a_refresh(
    fake: JiraFake, clock: FakeClock, account: str, client_for: Any
) -> None:
    access, refresh = fake.user_tokens([READ])
    tokens = TokenSet(access, refresh, expires_at=clock.time() + 3600, scopes=frozenset([READ]))
    store = MemoryTokenStore({account: tokens})
    client = client_for(user_credential(fake, clock, account, store))
    with pytest.raises(PermissionDenied, match="OAuth scope"):
        await client.create_issue("BENCH", "Not allowed")
    assert refreshes(fake) == 0, "a new token would have the same scopes"
    assert store.load(account) == tokens


# --- API tokens (CI) ----------------------------------------------------------------------


async def test_api_token_uses_basic_auth(fake: JiraFake, client_for: Any) -> None:
    fake.issue_token(fake.basic_token("bot@example.com", "api-token"), ALL)
    cred = api_token("jira-ci", email="bot@example.com", token="api-token")
    assert (await client_for(cred).get_issue("BENCH-1")).key == "BENCH-1"
    assert fake.requests[-1].headers["authorization"].startswith("Basic ")
    assert await granted_scopes(cred) is None


async def test_revoked_api_token_asks_for_a_new_one(fake: JiraFake, client_for: Any) -> None:
    cred = api_token("jira-ci", email="bot@example.com", token="revoked")
    with pytest.raises(AuthError, match="static token"):
        await client_for(cred).get_issue("BENCH-1")


# --- Rate limits and network errors -------------------------------------------------------


async def test_rate_limit_honours_retry_after(
    fake: JiraFake, clock: FakeClock, client_for: Any
) -> None:
    fake.inject(Fault.rate_limit(retry_after=7), method="GET")
    assert (await client_for(credential(TOKEN)).get_issue("BENCH-1")).key == "BENCH-1"
    assert clock.sleeps == [7]


async def test_budget_exceeded_on_the_client(
    fake: JiraFake, clock: FakeClock, client_for: Any
) -> None:
    fake.inject(Fault.rate_limit(retry_after=45), method="GET")
    with pytest.raises(RateLimited) as exc:
        await client_for(credential(TOKEN)).get_issue("BENCH-1")
    assert exc.value.retry_after == 45
    assert clock.sleeps == []


async def test_wait_budget_is_per_tool_call(fake: JiraFake, clock: FakeClock) -> None:
    server = create_server(
        credential=credential(TOKEN),
        transport=fake.transport(),
        clock=clock,
        read_only=False,
        variant="v1",
    )
    fake.inject(Fault.rate_limit(retry_after=TOOL_WAIT_BUDGET + 1), method="GET")
    async with Client(server) as client:
        result = await client.call_tool(GET, {"issue_key": "BENCH-1"})
        assert result.is_error
        assert f"{PROVIDER} rate limited" in text(result)

        # The next tool call starts with a fresh budget.
        fake.inject(Fault.rate_limit(retry_after=TOOL_WAIT_BUDGET - 1), method="GET")
        assert not (await client.call_tool(GET, {"issue_key": "BENCH-1"})).is_error
    assert clock.sleeps == [TOOL_WAIT_BUDGET - 1]


async def test_network_errors_on_writes(fake: JiraFake, clock: FakeClock) -> None:
    server = create_server(
        credential=credential(TOKEN),
        transport=fake.transport(),
        clock=clock,
        read_only=False,
        variant="v1",
    )
    before = len(fake.issues)
    fake.inject(Fault.network(), method="POST")
    async with Client(server) as client:
        result = await client.call_tool(CREATE, {"project_key": "BENCH", "summary": "x"})
    assert result.is_error
    assert f"Could not reach {PROVIDER} (ConnectError)" in text(result)
    assert len(fake.issues) == before


async def test_writes_are_not_retried_on_rate_limits(
    fake: JiraFake, clock: FakeClock, client_for: Any
) -> None:
    fake.inject(Fault.rate_limit(retry_after=3), method="POST")
    with pytest.raises(ToolError, match="429"):
        await client_for(credential(TOKEN)).create_issue("BENCH", "Once")
    assert clock.sleeps == []
    assert len(fake.calls("POST")) == 1


async def test_server_errors_on_reads_back_off_and_retry(
    fake: JiraFake, clock: FakeClock, client_for: Any
) -> None:
    fake.inject(Fault.server_error(503), method="GET", times=2)
    assert (await client_for(credential(TOKEN)).get_issue("BENCH-1")).key == "BENCH-1"
    assert len(clock.sleeps) == 2
    assert 2 * 0.7 <= clock.sleeps[0] <= 5 * 1.3
    assert clock.sleeps[1] > clock.sleeps[0]
