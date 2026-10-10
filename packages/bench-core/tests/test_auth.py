import json
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import parse_qs

import anyio
import httpx2
import jwt
import keyring
import keyring.backend
import keyring.errors
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from mcp.server.mcpserver.exceptions import ToolError

from bench_core.auth import (
    AppInstallation,
    AuthError,
    ClientCredentials,
    CredentialAuth,
    EnvTokenStore,
    FileTokenStore,
    KeyringTokenStore,
    MemoryTokenStore,
    OAuthError,
    ReadOnlyStoreError,
    RotatingRefresh,
    StaticToken,
    TokenEndpoint,
    TokenSet,
    parse_scopes,
)
from bench_core.http import ApiClient, RetryPolicy
from bench_core.testing import FakeClock

TOKEN_URL = "https://auth.example.invalid/oauth/token"


class FakeOAuthProvider:
    """Token endpoint + API in one: rotates refresh tokens, rejects reused ones."""

    def __init__(self, clock: FakeClock, *, rotate: bool = True, expires_in: int = 3600) -> None:
        self.clock = clock
        self.rotate = rotate
        self.expires_in = expires_in
        self.valid_access: set[str] = {"at-0"}
        self.valid_refresh: set[str] = {"rt-0"}
        self.issued = 0
        self.refresh_calls = 0
        self.api_auth_headers: list[str | None] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if str(request.url) == TOKEN_URL:
            return self._token(request)
        header = request.headers.get("Authorization")
        self.api_auth_headers.append(header)
        token = (header or "").removeprefix("Bearer ")
        if token not in self.valid_access:
            return httpx2.Response(401, json={"message": "Unauthorized"})
        return httpx2.Response(200, json={"ok": True})

    def _token(self, request: httpx2.Request) -> httpx2.Response:
        self.refresh_calls += 1
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        assert form["grant_type"] == "refresh_token"
        assert request.headers["Accept"] == "application/json"
        rt = form["refresh_token"]
        if rt not in self.valid_refresh:
            return httpx2.Response(400, json={"error": "invalid_grant", "error_description": "bad"})
        self.issued += 1
        self.valid_access = {f"at-{self.issued}"}
        body = {
            "access_token": f"at-{self.issued}",
            "expires_in": self.expires_in,
            "scope": "read:jira-work write:jira-work",
        }
        if self.rotate:
            self.valid_refresh.discard(rt)
            self.valid_refresh.add(f"rt-{self.issued}")
            body["refresh_token"] = f"rt-{self.issued}"
        return httpx2.Response(200, json=body)


def setup_rotating(
    *, rotate: bool = True, expires_at: float | None = None, store: MemoryTokenStore | None = None
) -> tuple[RotatingRefresh, MemoryTokenStore, FakeOAuthProvider, ApiClient, FakeClock]:
    clock = FakeClock()
    provider = FakeOAuthProvider(clock, rotate=rotate)
    transport = httpx2.MockTransport(provider)
    store = store or MemoryTokenStore(
        {
            "jira": TokenSet(
                access_token="at-0",
                refresh_token="rt-0",
                expires_at=expires_at if expires_at is not None else clock.now + 3600,
            )
        }
    )
    endpoint = TokenEndpoint(TOKEN_URL, client_id="cid", client_secret="sec", transport=transport)
    cred = RotatingRefresh("jira", store, endpoint, clock=clock)
    api = ApiClient(
        "https://api.example.invalid",
        policy=RetryPolicy(),
        clock=clock,
        transport=transport,
        auth=CredentialAuth(cred),
    )
    return cred, store, provider, api, clock


# --- RotatingRefresh ----------------------------------------------------------------------


@pytest.mark.anyio
async def test_valid_token_is_used_without_refresh() -> None:
    _, store, provider, api, _ = setup_rotating()
    assert (await api.get("/myself")).status_code == 200
    assert provider.refresh_calls == 0
    assert store.saves == 0


@pytest.mark.anyio
async def test_401_triggers_refresh_and_single_retry() -> None:
    _, store, provider, api, clock = setup_rotating()
    provider.valid_access = {"other"}  # server revoked at-0 early

    assert (await api.get("/myself")).status_code == 200
    assert provider.api_auth_headers == ["Bearer at-0", "Bearer at-1"]
    saved = store.tokens["jira"]
    assert saved.access_token == "at-1"
    assert saved.refresh_token == "rt-1"
    assert saved.expires_at == clock.now + 3600
    assert saved.scopes == {"read:jira-work", "write:jira-work"}


@pytest.mark.anyio
async def test_tokens_persisted_before_retry_is_sent() -> None:
    _, store, provider, api, _ = setup_rotating()
    provider.valid_access = {"other"}
    original = provider.__call__
    store_at_retry: list[TokenSet | None] = []

    def spy(request: httpx2.Request) -> httpx2.Response:
        if request.headers.get("Authorization") == "Bearer at-1":
            store_at_retry.append(store.load("jira"))
        return original(request)

    api._http._transport = httpx2.MockTransport(spy)  # pyright: ignore[reportPrivateUsage]
    await api.get("/myself")
    assert store_at_retry
    assert store_at_retry[0] is not None
    assert store_at_retry[0].refresh_token == "rt-1"


@pytest.mark.anyio
async def test_proactive_refresh_shortly_before_expiry() -> None:
    clock_start = FakeClock().now
    _, _, provider, api, _ = setup_rotating(expires_at=clock_start + 30)  # within 60 s skew
    assert (await api.get("/myself")).status_code == 200
    assert provider.refresh_calls == 1
    assert provider.api_auth_headers == ["Bearer at-1"]


@pytest.mark.anyio
async def test_still_401_after_refresh_returns_401() -> None:
    _, _, provider, api, _ = setup_rotating()
    provider.valid_access = {"other"}

    def deny(request: httpx2.Request) -> httpx2.Response:
        if str(request.url) == TOKEN_URL:
            return provider(request)
        return httpx2.Response(401)

    api._http._transport = httpx2.MockTransport(deny)  # pyright: ignore[reportPrivateUsage]
    assert (await api.get("/myself")).status_code == 401
    assert provider.refresh_calls == 1


@pytest.mark.anyio
async def test_invalid_grant_asks_for_new_login() -> None:
    cred, _, provider, _, _ = setup_rotating()
    provider.valid_refresh = set()
    with pytest.raises(
        AuthError, match=r"expired, revoked or was already used.*bench auth login jira"
    ):
        await cred.authorization(failed="Bearer at-0")


@pytest.mark.anyio
async def test_reusing_an_old_refresh_token_fails() -> None:
    # Another machine already rotated rt-0; our stale copy is rejected by the provider.
    cred, _, provider, _, _ = setup_rotating()
    provider.valid_refresh = {"rt-7"}
    with pytest.raises(AuthError, match="already used"):
        await cred.authorization(failed="Bearer at-0")


@pytest.mark.anyio
async def test_concurrent_401s_refresh_only_once() -> None:
    cred, _, provider, _, _ = setup_rotating()
    results: list[str] = []

    async def worker() -> None:
        results.append(await cred.authorization(failed="Bearer at-0"))

    async with anyio.create_task_group() as tg:
        for _ in range(5):
            tg.start_soon(worker)
    assert provider.refresh_calls == 1
    assert results == ["Bearer at-1"] * 5


@pytest.mark.anyio
async def test_non_rotating_provider_keeps_refresh_token() -> None:
    cred, store, _, _, _ = setup_rotating(rotate=False)
    await cred.authorization(failed="Bearer at-0")
    assert store.tokens["jira"].refresh_token == "rt-0"
    assert store.tokens["jira"].access_token == "at-1"


@pytest.mark.anyio
async def test_save_failure_is_reported() -> None:
    class Broken(MemoryTokenStore):
        def save(self, account: str, tokens: TokenSet) -> None:
            raise OSError("disk full")

    cred, _, _, _, _ = setup_rotating(
        store=Broken({"jira": TokenSet(access_token="at-0", refresh_token="rt-0")})
    )
    with pytest.raises(AuthError, match=r"could not persist.*disk full"):
        await cred.authorization(failed="Bearer at-0")


@pytest.mark.anyio
async def test_missing_tokens_ask_for_login() -> None:
    cred, _, _, _, _ = setup_rotating(store=MemoryTokenStore())
    with pytest.raises(AuthError, match="no stored refresh token"):
        await cred.authorization()


def test_auth_error_is_a_tool_error() -> None:
    assert issubclass(AuthError, ToolError)


# --- StaticToken --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_static_bearer_and_basic() -> None:
    assert await StaticToken("slack", "xoxb-1").authorization() == "Bearer xoxb-1"
    basic = await StaticToken.basic("jira", "me@example.invalid", "tok").authorization()
    assert basic == "Basic bWVAZXhhbXBsZS5pbnZhbGlkOnRvaw=="


@pytest.mark.anyio
async def test_static_token_401_is_actionable() -> None:
    api = ApiClient(
        "https://api.example.invalid",
        policy=RetryPolicy(),
        transport=httpx2.MockTransport(lambda _: httpx2.Response(401)),
        auth=CredentialAuth(StaticToken("slack", "xoxb-bad")),
    )
    with pytest.raises(AuthError, match="rejected the static token"):
        await api.get("/api/auth.test")


# --- OAuth endpoint quirks ----------------------------------------------------------------


@pytest.mark.anyio
async def test_slack_ok_false_on_http_200_is_an_error() -> None:
    transport = httpx2.MockTransport(
        lambda _: httpx2.Response(200, json={"ok": False, "error": "invalid_refresh_token"})
    )
    endpoint = TokenEndpoint(TOKEN_URL, client_id="c", client_secret="s", transport=transport)
    with pytest.raises(OAuthError) as info:
        await endpoint.refresh("rt", FakeClock())
    assert info.value.is_invalid_grant


@pytest.mark.anyio
async def test_basic_client_auth_and_non_json_response() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, text="access_token=x&scope=repo")

    endpoint = TokenEndpoint(
        TOKEN_URL,
        client_id="c",
        client_secret="s",
        client_auth="basic",
        transport=httpx2.MockTransport(handler),
    )
    with pytest.raises(OAuthError, match="invalid_response"):
        await endpoint.refresh("rt", FakeClock())
    assert seen[0].headers["Authorization"].startswith("Basic ")
    assert "client_secret" not in seen[0].content.decode()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a b", {"a", "b"}),
        ("chat:write,channels:read", {"chat:write", "channels:read"}),
        (None, set()),
    ],
)
def test_parse_scopes(raw: str | None, expected: set[str]) -> None:
    assert parse_scopes(raw) == expected


@pytest.mark.anyio
async def test_client_credentials_are_cached_until_expiry() -> None:
    clock = FakeClock()
    calls: list[dict[str, list[str]]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(parse_qs(request.content.decode()))
        return httpx2.Response(200, json={"access_token": f"cc-{len(calls)}", "expires_in": 600})

    endpoint = TokenEndpoint(
        TOKEN_URL, client_id="c", client_secret="s", transport=httpx2.MockTransport(handler)
    )
    cred = ClientCredentials("svc", endpoint, scopes=frozenset({"b", "a"}), clock=clock)
    assert await cred.authorization() == "Bearer cc-1"
    assert await cred.authorization() == "Bearer cc-1"
    clock.now += 600
    assert await cred.authorization() == "Bearer cc-2"
    assert calls[0]["grant_type"] == ["client_credentials"]
    assert calls[0]["scope"] == ["a b"]


# --- GitHub App installation --------------------------------------------------------------


@pytest.fixture(scope="module")
def rsa_key() -> tuple[str, str]:
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


@pytest.mark.anyio
async def test_app_installation_mints_and_caches_token(rsa_key: tuple[str, str]) -> None:
    private, public = rsa_key
    clock = FakeClock()
    minted: list[dict[str, object]] = []

    def github(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/app/installations/42/access_tokens"
        token = request.headers["Authorization"].removeprefix("Bearer ")
        minted.append(
            jwt.decode(
                token,
                public,
                algorithms=["RS256"],
                options={"verify_exp": False, "verify_iat": False},
            )
        )
        return httpx2.Response(
            201,
            json={
                "token": f"ghs_{len(minted)}",
                "expires_at": "2027-01-15T09:00:00Z",  # clock.now + 1 h
                "permissions": {"issues": "write", "contents": "read"},
            },
        )

    cred = AppInstallation(
        "github",
        app_id="123",
        private_key_pem=private,
        installation_id=42,
        transport=httpx2.MockTransport(github),
        clock=clock,
    )
    assert await cred.authorization() == "Bearer ghs_1"
    assert await cred.authorization() == "Bearer ghs_1"
    assert minted[0]["iss"] == "123"
    assert minted[0]["iat"] == int(clock.now) - 60
    assert minted[0]["exp"] == int(clock.now) + 540
    assert cred._tokens is not None  # pyright: ignore[reportPrivateUsage]
    assert cred._tokens.scopes == {"issues:write", "contents:read"}  # pyright: ignore[reportPrivateUsage]

    clock.now += 3600 - 299  # within the 5 min skew -> renew
    assert await cred.authorization() == "Bearer ghs_2"
    assert await cred.authorization(failed="Bearer ghs_2") == "Bearer ghs_3"


@pytest.mark.anyio
async def test_app_installation_failure_is_actionable(rsa_key: tuple[str, str]) -> None:
    cred = AppInstallation(
        "github",
        app_id="123",
        private_key_pem=rsa_key[0],
        installation_id=1,
        transport=httpx2.MockTransport(lambda _: httpx2.Response(404)),
        clock=FakeClock(),
    )
    with pytest.raises(AuthError, match=r"HTTP 404.*installation id"):
        await cred.authorization()


# --- Stores -------------------------------------------------------------------------------

SAMPLE = TokenSet(
    access_token="a" * 2500,  # longer than a Windows credential blob
    refresh_token="r",
    expires_at=1.5,
    scopes=frozenset({"x", "y"}),
)


def test_token_set_json_round_trip() -> None:
    assert TokenSet.from_json(SAMPLE.to_json()) == SAMPLE


def test_file_store_round_trip_and_atomic_write(tmp_path: Path) -> None:
    store = FileTokenStore(tmp_path / "tokens")
    assert store.load("jira") is None
    store.save("jira", SAMPLE)
    store.save("jira", TokenSet(access_token="new"))
    assert store.load("jira") == TokenSet(access_token="new")
    assert [p.name for p in (tmp_path / "tokens").iterdir()] == ["jira.json"]


def test_file_store_rejects_path_tricks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid account"):
        FileTokenStore(tmp_path).save("../evil", SAMPLE)


class DictKeyring(keyring.backend.KeyringBackend):
    priority = 1  # pyright: ignore[reportAssignmentType]

    def __init__(self) -> None:
        super().__init__()
        self.data: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.data.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        assert len(password) <= 1280, "Windows Credential Manager limit"
        self.data[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        if self.data.pop((service, username), None) is None:
            raise keyring.errors.PasswordDeleteError(username)


@pytest.fixture
def fake_keyring() -> Iterator[DictKeyring]:
    previous = keyring.get_keyring()
    backend = DictKeyring()
    keyring.set_keyring(backend)
    try:
        yield backend
    finally:
        keyring.set_keyring(previous)


def test_keyring_store_chunks_and_cleans_up(fake_keyring: DictKeyring) -> None:
    store = KeyringTokenStore()
    assert store.load("jira") is None
    store.save("jira", SAMPLE)
    assert store.load("jira") == SAMPLE
    assert len(fake_keyring.data) == 4  # pointer + 3 chunks

    small = TokenSet(access_token="s")
    store.save("jira", small)
    assert store.load("jira") == small
    assert len(fake_keyring.data) == 2  # pointer + 1 chunk, old generation deleted


def test_keyring_store_survives_crash_before_pointer_flip(fake_keyring: DictKeyring) -> None:
    store = KeyringTokenStore()
    store.save("jira", SAMPLE)
    original_set = fake_keyring.set_password

    def crash_on_pointer(service: str, username: str, password: str) -> None:
        if username == "jira":
            raise RuntimeError("power loss")
        original_set(service, username, password)

    fake_keyring.set_password = crash_on_pointer  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        store.save("jira", TokenSet(access_token="b" * 3000))
    assert store.load("jira") == SAMPLE


def test_env_store_reads_and_caches_but_never_rotates() -> None:
    env = {"DRIVE_RT": "rt-ci", "DRIVE_SCOPES": "https://www.googleapis.com/auth/drive.file"}
    store = EnvTokenStore(
        {"gdrive": {"refresh_token": "DRIVE_RT", "scopes": "DRIVE_SCOPES"}}, environ=env
    )
    loaded = store.load("gdrive")
    assert loaded == TokenSet(
        refresh_token="rt-ci", scopes=frozenset({"https://www.googleapis.com/auth/drive.file"})
    )
    store.save("gdrive", TokenSet(access_token="fresh", refresh_token="rt-ci"))
    cached = store.load("gdrive")
    assert cached is not None
    assert cached.access_token == "fresh"

    with pytest.raises(ReadOnlyStoreError, match="rotated the refresh token"):
        store.save("gdrive", TokenSet(access_token="x", refresh_token="rt-new"))
    with pytest.raises(ReadOnlyStoreError, match="no env credentials"):
        store.save("slack", TokenSet(access_token="x"))
    assert store.load("slack") is None
    assert (
        EnvTokenStore({"gdrive": {"refresh_token": "DRIVE_RT"}}, environ={}).load("gdrive") is None
    )


@pytest.mark.anyio
async def test_env_store_with_non_rotating_refresh_in_ci() -> None:
    clock = FakeClock()
    provider = FakeOAuthProvider(clock, rotate=False)
    provider.valid_refresh = {"rt-ci"}
    store = EnvTokenStore({"gdrive": {"refresh_token": "RT"}}, environ={"RT": "rt-ci"})
    endpoint = TokenEndpoint(
        TOKEN_URL, client_id="c", client_secret="s", transport=httpx2.MockTransport(provider)
    )
    cred = RotatingRefresh("gdrive", store, endpoint, clock=clock)
    assert await cred.authorization() == "Bearer at-1"
    assert await cred.authorization() == "Bearer at-1"
    assert provider.refresh_calls == 1


def test_sample_is_json_serializable() -> None:
    assert json.loads(SAMPLE.to_json())["scopes"] == ["x", "y"]
