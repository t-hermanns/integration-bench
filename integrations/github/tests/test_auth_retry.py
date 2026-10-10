"""GitHub auth and retry scenarios against the in-memory fake.

Runs over `httpx2.MockTransport` and never inside a cassette (vcrpy patches MockTransport
too). Clocks are fake, so waits are recorded instead of slept.
"""

from collections.abc import AsyncIterator
from typing import Any

import anyio
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from mcp.client import Client
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent

from bench_core.auth import AuthError, Credential, MemoryTokenStore, StaticToken, TokenSet
from bench_core.http import RateLimited
from bench_core.testing import FakeClock, Fault
from bench_github.auth import TOKEN_URL, granted_permissions, installation, user_token
from bench_github.client import GitHubClient, make_api
from bench_github.fake import OWNER, REPO, GitHubFake, seeded_fake
from bench_github.server import create_server
from bench_github.tools import TOOL_WAIT_BUDGET

pytestmark = pytest.mark.anyio

CLIENT_ID = "Iv23-fake-client"
CLIENT_SECRET = "fake-client-secret"
APP_ID = "424242"
INSTALLATION_ID = 7
ALL = ["issues:write", "pull_requests:read", "contents:read"]


@pytest.fixture(scope="module")
def app_key() -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public = (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    return private, public


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def fake(clock: FakeClock, app_key: tuple[str, str]) -> GitHubFake:
    fake = seeded_fake(clock)
    fake.oauth_clients[CLIENT_ID] = CLIENT_SECRET
    fake.register_app(
        APP_ID,
        app_key[1],
        {INSTALLATION_ID: {"issues": "read", "pull_requests": "read", "metadata": "read"}},
    )
    return fake


@pytest.fixture
def account(request: pytest.FixtureRequest) -> str:
    # Refresh locks are per account and process-wide; keep tests independent.
    return f"github-{request.node.name}"


def login(
    fake: GitHubFake, clock: FakeClock, account: str, *, expires_in: float = 28_800
) -> tuple[MemoryTokenStore, str]:
    """State right after `bench auth login github`."""
    access, refresh = fake.user_tokens(ALL)
    store = MemoryTokenStore(
        {account: TokenSet(access, refresh, expires_at=clock.time() + expires_in)}
    )
    return store, refresh


def user_credential(
    fake: GitHubFake, clock: FakeClock, account: str, store: MemoryTokenStore
) -> Credential:
    return user_token(
        account,
        store,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        transport=fake.transport(),
        clock=clock,
    )


@pytest.fixture
async def gh_for(fake: GitHubFake, clock: FakeClock) -> AsyncIterator[Any]:
    apis = []

    def make(credential: Credential) -> GitHubClient:
        api = make_api(credential=credential, transport=fake.transport(), clock=clock)
        apis.append(api)
        return GitHubClient(api)

    yield make
    for api in apis:
        await api.aclose()


def text(result: CallToolResult) -> str:
    return "\n".join(c.text for c in result.content if isinstance(c, TextContent))


def refreshes(fake: GitHubFake) -> int:
    return len(fake.calls("POST", "/login/oauth/access_token"))


# --- User tokens (rotating refresh) -------------------------------------------------------


async def test_expiring_token_is_refreshed_before_the_call(
    fake: GitHubFake, clock: FakeClock, account: str, gh_for: Any
) -> None:
    store, old_refresh = login(fake, clock, account, expires_in=30)  # inside the 60 s skew
    gh = gh_for(user_credential(fake, clock, account, store))

    issue = await gh.get_issue(OWNER, REPO, 1)

    assert issue.number == 1
    assert refreshes(fake) == 1
    saved = store.tokens[account]
    assert saved.refresh_token != old_refresh, "GitHub rotates refresh tokens"
    assert saved.expires_at == clock.time() + 28_800
    form = fake.calls("POST", "/login/oauth/access_token")[0].content.decode()
    assert "grant_type=refresh_token" in form
    assert f"client_secret={CLIENT_SECRET}" in form
    # Only one request to the API itself: the refresh happened up front.
    assert len(fake.calls("GET", "/repos/{owner}/{repo}/issues/{number}")) == 1


async def test_401_refreshes_and_retries_once(
    fake: GitHubFake, clock: FakeClock, account: str, gh_for: Any
) -> None:
    store, _ = login(fake, clock, account)
    revoked = store.tokens[account].access_token or ""
    fake.expire_token(revoked)  # revoked server-side
    gh = gh_for(user_credential(fake, clock, account, store))

    assert (await gh.get_issue(OWNER, REPO, 1)).number == 1
    assert len(fake.calls("GET", "/repos/{owner}/{repo}/issues/{number}")) == 2
    assert store.tokens[account].access_token not in (None, revoked)
    assert refreshes(fake) == 1
    assert store.saves == 1, "renewed tokens are stored before the retry"


async def test_reused_refresh_token_asks_for_login(
    fake: GitHubFake, clock: FakeClock, account: str, gh_for: Any
) -> None:
    store, old_refresh = login(fake, clock, account, expires_in=0)
    gh = gh_for(user_credential(fake, clock, account, store))
    await gh.get_issue(OWNER, REPO, 1)  # rotates `old_refresh` away

    # Another machine still holds the old token set (e.g. a stale copy of the store).
    stale = MemoryTokenStore({account: TokenSet("gone", old_refresh, expires_at=0)})
    stale_gh = gh_for(user_credential(fake, clock, account, stale))
    with pytest.raises(AuthError, match="expired, revoked or was already used") as exc:
        await stale_gh.get_issue(OWNER, REPO, 1)
    assert f"bench auth login {account}" in str(exc.value)


async def test_wrong_client_secret_is_reported(
    fake: GitHubFake, clock: FakeClock, account: str, gh_for: Any
) -> None:
    store, _ = login(fake, clock, account, expires_in=0)
    credential = user_token(
        account,
        store,
        client_id=CLIENT_ID,
        client_secret="wrong",
        transport=fake.transport(),
        clock=clock,
    )
    with pytest.raises(AuthError, match="incorrect_client_credentials"):
        await gh_for(credential).get_issue(OWNER, REPO, 1)


async def test_concurrent_calls_share_one_refresh(
    fake: GitHubFake, clock: FakeClock, account: str, gh_for: Any
) -> None:
    store, _ = login(fake, clock, account, expires_in=0)
    gh = gh_for(user_credential(fake, clock, account, store))
    numbers: list[int] = []

    async def fetch(n: int) -> None:
        numbers.append((await gh.get_issue(OWNER, REPO, n)).number)

    async with anyio.create_task_group() as tg:
        for n in range(1, 6):
            tg.start_soon(fetch, n)

    assert sorted(numbers) == [1, 2, 3, 4, 5]
    assert refreshes(fake) == 1, "a single-use refresh token must not be spent twice"


async def test_auth_error_reaches_the_model_through_the_server(
    fake: GitHubFake, clock: FakeClock, account: str
) -> None:
    store = MemoryTokenStore({account: TokenSet("gone", "fake-refresh-unknown", expires_at=0)})
    server = create_server(
        credential=user_credential(fake, clock, account, store),
        transport=fake.transport(),
        clock=clock,
        read_only=False,
        variant="v1",
    )
    async with Client(server) as client:
        result = await client.call_tool(
            "github_get_issue", {"owner": OWNER, "repo": REPO, "number": 1}
        )
    assert result.is_error
    assert f"Run `bench auth login {account}`" in text(result)


def test_token_url_is_githubs() -> None:
    assert TOKEN_URL == "https://github.com/login/oauth/access_token"


# --- Installation tokens (CI) -------------------------------------------------------------


def app_credential(
    fake: GitHubFake, clock: FakeClock, account: str, key: str, installation_id: int = 7
) -> Credential:
    return installation(
        account,
        app_id=APP_ID,
        private_key_pem=key,
        installation_id=installation_id,
        transport=fake.transport(),
        clock=clock,
    )


async def test_installation_token_reports_permissions_and_filters_tools(
    fake: GitHubFake, clock: FakeClock, account: str, app_key: tuple[str, str]
) -> None:
    credential = app_credential(fake, clock, account, app_key[0])
    granted = await granted_permissions(credential)
    assert granted == frozenset({"issues:read", "pull_requests:read", "metadata:read"})

    server = create_server(
        credential=credential,
        transport=fake.transport(),
        clock=clock,
        granted=granted,
        read_only=False,
        variant="v1",
    )
    async with Client(server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        result = await client.call_tool(
            "github_get_issue", {"owner": OWNER, "repo": REPO, "number": 1}
        )
    assert names == {"github_search_issues", "github_get_issue", "github_list_pull_requests"}
    assert not result.is_error
    assert len(fake.calls("POST", "/app/installations/{id}/access_tokens")) == 1


async def test_installation_token_is_renewed_before_expiry(
    fake: GitHubFake, clock: FakeClock, account: str, app_key: tuple[str, str], gh_for: Any
) -> None:
    gh = gh_for(app_credential(fake, clock, account, app_key[0]))
    await gh.get_issue(OWNER, REPO, 1)
    clock.advance(3600 - 299)  # inside the 5 minute skew
    await gh.get_issue(OWNER, REPO, 1)
    assert len(fake.calls("POST", "/app/installations/{id}/access_tokens")) == 2


async def test_installation_401_mints_a_new_token(
    fake: GitHubFake, clock: FakeClock, account: str, app_key: tuple[str, str], gh_for: Any
) -> None:
    credential = app_credential(fake, clock, account, app_key[0])
    gh = gh_for(credential)
    await gh.get_issue(OWNER, REPO, 1)
    first = credential.tokens()  # type: ignore[attr-defined]
    fake.expire_token(first.access_token)
    assert (await gh.get_issue(OWNER, REPO, 2)).number == 2
    assert credential.tokens().access_token != first.access_token  # type: ignore[attr-defined]


async def test_wrong_app_key_or_installation(
    fake: GitHubFake, clock: FakeClock, account: str, app_key: tuple[str, str], gh_for: Any
) -> None:
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other_pem = other.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    with pytest.raises(AuthError, match=r"HTTP 401.*private key"):
        await gh_for(app_credential(fake, clock, account, other_pem)).get_issue(OWNER, REPO, 1)
    with pytest.raises(AuthError, match="HTTP 404"):
        await gh_for(app_credential(fake, clock, f"{account}-2", app_key[0], 99)).get_issue(
            OWNER, REPO, 1
        )


async def test_user_tokens_report_no_permissions(
    fake: GitHubFake, clock: FakeClock, account: str
) -> None:
    store, _ = login(fake, clock, account)
    assert await granted_permissions(user_credential(fake, clock, account, store)) is None
    assert await granted_permissions(StaticToken(account, "x")) is None


# --- Rate limits and network errors -------------------------------------------------------


@pytest.fixture
def token(fake: GitHubFake) -> StaticToken:
    fake.issue_token("static", ALL)
    return StaticToken("github", "static")


async def test_primary_limit_waits_until_reset(
    fake: GitHubFake, clock: FakeClock, token: StaticToken, gh_for: Any
) -> None:
    fake.inject(Fault.rate_limit(), path="/repos/{owner}/{repo}/issues/{number}")
    api = make_api(credential=token, transport=fake.transport(), clock=clock)
    async with api:
        with api.wait_budget(120):
            issue = await GitHubClient(api).get_issue(OWNER, REPO, 1)
    assert issue.number == 1
    assert clock.sleeps == [61.0], "reset (+60 s) plus one second of slack"


@pytest.mark.parametrize("status_fault", [Fault.rate_limit(retry_after=7)])
async def test_secondary_limit_honours_retry_after(
    fake: GitHubFake, clock: FakeClock, token: StaticToken, gh_for: Any, status_fault: Fault
) -> None:
    fake.inject(status_fault, path="/search/issues")
    page = await gh_for(token).search_issues(f"repo:{OWNER}/{REPO}")
    assert page.items
    assert clock.sleeps == [7]


async def test_create_issue_is_retried_on_rate_limit_without_duplicates(
    fake: GitHubFake, clock: FakeClock, token: StaticToken, gh_for: Any
) -> None:
    repo = fake.repos[f"{OWNER}/{REPO}"]
    before = len(repo.issues)
    fake.inject(Fault.rate_limit(retry_after=3), method="POST")
    created = await gh_for(token).create_issue(OWNER, REPO, "Only once")
    assert len(repo.issues) == before + 1
    assert repo.issues[created.number]["title"] == "Only once"
    assert clock.sleeps == [3]


async def test_server_errors_are_not_retried(
    fake: GitHubFake, clock: FakeClock, token: StaticToken, gh_for: Any
) -> None:
    fake.inject(Fault.server_error(503))
    with pytest.raises(ToolError, match=r"server error \(503\)"):
        await gh_for(token).get_issue(OWNER, REPO, 1)
    assert clock.sleeps == []


async def test_wait_budget_is_shared_by_one_tool_call(
    fake: GitHubFake, clock: FakeClock, token: StaticToken
) -> None:
    # get_issue (detailed) makes two requests; 15 s + 15 s exceeds the 20 s budget.
    assert TOOL_WAIT_BUDGET == 20
    fake.inject(Fault.rate_limit(retry_after=15), path="/repos/{owner}/{repo}/issues/{number}")
    fake.inject(
        Fault.rate_limit(retry_after=15), path="/repos/{owner}/{repo}/issues/{number}/comments"
    )
    server = create_server(
        credential=token, transport=fake.transport(), clock=clock, read_only=False, variant="v1"
    )
    async with Client(server) as client:
        result = await client.call_tool(
            "github_get_issue",
            {"owner": OWNER, "repo": REPO, "number": 1, "response_format": "detailed"},
        )
        assert result.is_error
        assert "rate limited, retry after 15 s" in text(result)
        assert clock.sleeps == [15]

        # The next tool call starts with a fresh budget.
        fake.inject(Fault.rate_limit(retry_after=15), path="/search/issues")
        assert not (await client.call_tool("github_search_issues", {"query": "crash"})).is_error
    assert clock.sleeps == [15, 15]


async def test_budget_exceeded_on_the_client(
    fake: GitHubFake, clock: FakeClock, token: StaticToken, gh_for: Any
) -> None:
    fake.inject(Fault.rate_limit(retry_after=45), path="/search/issues")
    with pytest.raises(RateLimited) as exc:
        await gh_for(token).search_issues("crash")
    assert exc.value.retry_after == 45
    assert clock.sleeps == []


async def test_network_errors_retry_reads_but_not_writes(
    fake: GitHubFake, clock: FakeClock, token: StaticToken
) -> None:
    server = create_server(
        credential=token, transport=fake.transport(), clock=clock, read_only=False, variant="v1"
    )
    repo = fake.repos[f"{OWNER}/{REPO}"]
    async with Client(server) as client:
        fake.inject(Fault.network(), method="GET")
        read = await client.call_tool(
            "github_get_issue", {"owner": OWNER, "repo": REPO, "number": 1}
        )
        assert not read.is_error

        before = len(repo.issues)
        fake.inject(Fault.network(), method="POST")
        write = await client.call_tool(
            "github_create_issue", {"owner": OWNER, "repo": REPO, "title": "x"}
        )
    assert write.is_error
    assert "Could not reach GitHub (ConnectError)" in text(write)
    assert "check whether it was created" in text(write)
    assert len(fake.calls("POST", "/repos/{owner}/{repo}/issues")) == 1
    assert len(repo.issues) == before
