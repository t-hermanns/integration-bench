"""Auth and retry scenarios against the in-memory fake.

Runs over `httpx2.MockTransport` and never inside a cassette (vcrpy patches MockTransport
too). Clocks are fake, so waits are recorded instead of slept.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from mcp.client import Client
from mcp.types import CallToolResult, TextContent

from bench_core.auth import AuthError, Credential, MemoryTokenStore, TokenSet
from bench_core.http import RateLimited
from bench_core.testing import FakeClock, Fault

from bench_gdrive.auth import granted_scopes, user_token
from bench_gdrive.client import PROVIDER, GdriveClient, make_api
from bench_gdrive.fake import GdriveFake, credential, seeded_fake
from bench_gdrive.fake.backend import TOKEN_PATH
from bench_gdrive.scopes import DRIVE_FILE
from bench_gdrive.server import create_server
from bench_gdrive.tools import TOOL_WAIT_BUDGET

pytestmark = pytest.mark.anyio

TOKEN = "retry-token"
ALL = [DRIVE_FILE]
META = "gdrive_get_file_metadata"
CREATE = "gdrive_create_document"
CLIENT_ID = "fake-client"
CLIENT_SECRET = "fake-client-secret"


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def fake(clock: FakeClock) -> GdriveFake:
    fake = seeded_fake(clock)
    fake.issue_token(TOKEN, ALL)
    fake.oauth_clients[CLIENT_ID] = CLIENT_SECRET
    return fake


@pytest.fixture
async def client_for(fake: GdriveFake, clock: FakeClock) -> AsyncIterator[Any]:
    apis = []

    def make(cred: Credential) -> GdriveClient:
        api = make_api(credential=cred, transport=fake.transport(), clock=clock)
        apis.append(api)
        return GdriveClient(api)

    yield make
    for api in apis:
        await api.aclose()


@pytest.fixture
def account(request: pytest.FixtureRequest) -> str:
    # Refresh locks are per account and process-wide; keep tests independent.
    return f"gdrive-{request.node.name}"


def text(result: CallToolResult) -> str:
    return "\n".join(c.text for c in result.content if isinstance(c, TextContent))


# --- User tokens (Google: refresh tokens do not rotate) ----------------------------------


def login(
    fake: GdriveFake, clock: FakeClock, account: str, *, expires_in: float = 3600
) -> MemoryTokenStore:
    """State right after `bench auth login gdrive`."""
    access, refresh = fake.user_tokens(ALL)
    tokens = TokenSet(access, refresh, expires_at=clock.time() + expires_in, scopes=frozenset(ALL))
    return MemoryTokenStore({account: tokens})


def user_credential(
    fake: GdriveFake, clock: FakeClock, account: str, store: MemoryTokenStore
) -> Credential:
    return user_token(
        account,
        store,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        transport=fake.transport(),
        clock=clock,
    )


def any_id(fake: GdriveFake) -> str:
    return next(iter(fake.files))


def refreshes(fake: GdriveFake) -> int:
    return len(fake.calls("POST", TOKEN_PATH))


async def test_expiring_token_is_refreshed_before_the_call(
    fake: GdriveFake, clock: FakeClock, account: str, client_for: Any
) -> None:
    store = login(fake, clock, account, expires_in=30)  # inside the 60 s skew
    old = store.load(account)
    client = client_for(user_credential(fake, clock, account, store))
    fid = any_id(fake)
    assert (await client.get_file(fid)).id == fid
    assert refreshes(fake) == 1
    new = store.load(account)
    assert new is not None
    assert old is not None
    assert new.access_token != old.access_token
    assert new.refresh_token == old.refresh_token, "Google omits refresh_token; keep ours"


async def test_401_refreshes_and_retries_once(
    fake: GdriveFake, clock: FakeClock, account: str, client_for: Any
) -> None:
    store = login(fake, clock, account)
    tokens = store.load(account)
    assert tokens is not None
    assert tokens.access_token is not None
    fake.expire_token(tokens.access_token)
    client = client_for(user_credential(fake, clock, account, store))
    await client.get_file(any_id(fake))
    assert refreshes(fake) == 1


async def test_refresh_token_works_again(
    fake: GdriveFake, clock: FakeClock, account: str, client_for: Any
) -> None:
    store = login(fake, clock, account, expires_in=0)
    client = client_for(user_credential(fake, clock, account, store))
    await client.get_file(any_id(fake))
    clock.advance(3600)
    await client.get_file(any_id(fake))
    assert refreshes(fake) == 2


async def test_revoked_refresh_token_asks_for_login(
    fake: GdriveFake, clock: FakeClock, account: str, client_for: Any
) -> None:
    store = login(fake, clock, account, expires_in=0)
    fake.refresh_tokens.clear()  # e.g. the user removed the app's access
    client = client_for(user_credential(fake, clock, account, store))
    with pytest.raises(AuthError, match=r"bench auth login"):
        await client.get_file(any_id(fake))


async def test_granted_scopes_come_from_the_token(
    fake: GdriveFake, clock: FakeClock, account: str
) -> None:
    store = login(fake, clock, account, expires_in=0)
    assert await granted_scopes(user_credential(fake, clock, account, store)) == frozenset(ALL)


# --- Rate limits and network errors -------------------------------------------------------


async def test_403_rate_limit_backs_off_exponentially(
    fake: GdriveFake, clock: FakeClock, client_for: Any
) -> None:
    fake.inject(Fault.rate_limit(), method="GET", times=3)
    fid = any_id(fake)
    assert (await client_for(credential(TOKEN)).get_file(fid)).id == fid
    assert clock.sleeps == [1.5, 2.5, 4.5]  # 2^n + jitter (FakeClock: 0.5)


async def test_server_errors_are_retried(
    fake: GdriveFake, clock: FakeClock, client_for: Any
) -> None:
    fake.inject(Fault.server_error(500), method="GET")
    await client_for(credential(TOKEN)).get_file(any_id(fake))
    assert clock.sleeps == [1.5]


async def test_persistent_rate_limit_reaches_the_model(fake: GdriveFake, clock: FakeClock) -> None:
    server = create_server(
        credential=credential(TOKEN),
        transport=fake.transport(),
        clock=clock,
        read_only=False,
        variant="v1",
    )
    fake.inject(Fault.rate_limit(), method="GET", times=None)
    async with Client(server) as client:
        result = await client.call_tool(META, {"file_id": any_id(fake)})
    assert result.is_error
    assert f"{PROVIDER} rate limited" in text(result)
    assert sum(clock.sleeps) <= TOOL_WAIT_BUDGET


async def test_rate_limit_error_is_not_a_permission_error(
    fake: GdriveFake, clock: FakeClock, client_for: Any
) -> None:
    fake.inject(Fault.rate_limit(), method="GET", times=None)
    with pytest.raises(RateLimited):
        await client_for(credential(TOKEN)).get_file(any_id(fake))


async def test_network_errors_on_writes(fake: GdriveFake, clock: FakeClock) -> None:
    server = create_server(
        credential=credential(TOKEN),
        transport=fake.transport(),
        clock=clock,
        read_only=False,
        variant="v1",
    )
    before = len(fake.files)
    fake.inject(Fault.network(), method="POST")
    async with Client(server) as client:
        result = await client.call_tool(CREATE, {"name": "x", "content": "y"})
    assert result.is_error
    assert f"Could not reach {PROVIDER} (ConnectError)" in text(result)
    assert len(fake.files) == before
