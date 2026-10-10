"""Credential kinds and the httpx2 auth flow that applies them.

- `StaticToken`        Slack `xoxb`, Jira API token (Basic), CI installation token
- `RotatingRefresh`    OAuth with refresh tokens (Jira 3LO, GitHub App user tokens, Slack
                       dev app, Drive). The new token set is persisted *before* it is used,
                       because rotating providers invalidate the old refresh token at once.
- `ClientCredentials`  client_credentials grant, token cached in memory
- `AppInstallation`    GitHub App: app JWT -> 1 h installation token, cached in memory

Every kind renews proactively shortly before expiry and reactively when a request comes
back 401 (`CredentialAuth`). Renewal is serialized per account; a task that waited for
the lock reuses a token another task just obtained instead of spending the refresh token
a second time.
"""

import base64
from dataclasses import replace
from datetime import datetime
from typing import Any, ClassVar, Protocol, Self

import anyio
import httpx2
import jwt
from mcp.server.mcpserver.exceptions import ToolError

from bench_core.auth.oauth import OAuthError, TokenEndpoint
from bench_core.auth.store import TokenSet, TokenStore
from bench_core.clock import Clock, SystemClock


class AuthError(ToolError):
    def __init__(self, account: str, reason: str, hint: str | None = None) -> None:
        self.account = account
        self.reason = reason
        hint = hint or f"Run `bench auth login {account}` to sign in again."
        super().__init__(f"Authentication for '{account}' failed: {reason}. {hint}")


class Credential(Protocol):
    account: str

    async def authorization(self, *, failed: str | None = None) -> str:
        """Value for the Authorization header.

        `failed` is a header value the server just rejected with 401; the credential must
        then return a different (renewed) value or raise `AuthError`.
        """
        ...


_locks: dict[str, anyio.Lock] = {}


def _lock(account: str) -> anyio.Lock:
    lock = _locks.get(account)
    if lock is None:
        lock = _locks[account] = anyio.Lock()
    return lock


def _bearer(token: str) -> str:
    return f"Bearer {token}"


class StaticToken:
    def __init__(self, account: str, token: str, *, scheme: str = "Bearer") -> None:
        self.account = account
        self._header = f"{scheme} {token}"

    @classmethod
    def basic(cls, account: str, username: str, token: str) -> Self:
        """Jira Cloud API tokens: HTTP Basic with the account e-mail."""
        encoded = base64.b64encode(f"{username}:{token}".encode()).decode("ascii")
        return cls(account, encoded, scheme="Basic")

    async def authorization(self, *, failed: str | None = None) -> str:
        if failed is not None:
            raise AuthError(
                self.account,
                "the server rejected the static token (401)",
                "Check that the token is valid and not revoked.",
            )
        return self._header


class _CachedRenewal:
    """Shared proactive/reactive renewal logic for kinds that can mint a fresh token."""

    skew: ClassVar[float] = 60.0

    def __init__(self, account: str, clock: Clock | None) -> None:
        self.account = account
        self.clock: Clock = clock or SystemClock()

    def _current(self) -> TokenSet | None:
        raise NotImplementedError

    def tokens(self) -> TokenSet | None:
        """The tokens in use right now (None before the first renewal)."""
        return self._current()

    async def _renew(self, current: TokenSet | None) -> TokenSet:
        raise NotImplementedError

    def _usable(self, tokens: TokenSet | None, failed: str | None) -> str | None:
        if tokens is None or not tokens.access_token:
            return None
        if tokens.expires_within(self.clock.time(), self.skew):
            return None
        header = _bearer(tokens.access_token)
        return None if header == failed else header

    async def authorization(self, *, failed: str | None = None) -> str:
        header = self._usable(self._current(), failed)
        if header is not None:
            return header
        async with _lock(self.account):
            current = self._current()  # another task may have renewed while we waited
            header = self._usable(current, failed)
            if header is not None:
                return header
            renewed = await self._renew(current)
            assert renewed.access_token is not None
            return _bearer(renewed.access_token)


class RotatingRefresh(_CachedRenewal):
    def __init__(
        self,
        account: str,
        store: TokenStore,
        endpoint: TokenEndpoint,
        *,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(account, clock)
        self.store = store
        self.endpoint = endpoint

    def _current(self) -> TokenSet | None:
        return self.store.load(self.account)

    async def _renew(self, current: TokenSet | None) -> TokenSet:
        if current is None or not current.refresh_token:
            raise AuthError(self.account, "no stored refresh token")
        try:
            fresh = await self.endpoint.refresh(current.refresh_token, self.clock)
        except OAuthError as e:
            if e.is_invalid_grant:
                raise AuthError(
                    self.account, "the refresh token is expired, revoked or was already used"
                ) from e
            raise AuthError(self.account, f"token refresh failed ({e})") from e
        merged = replace(
            fresh,
            # Non-rotating providers omit refresh_token; keep the one we have.
            refresh_token=fresh.refresh_token or current.refresh_token,
            scopes=fresh.scopes or current.scopes,
        )
        try:
            self.store.save(self.account, merged)
        except Exception as e:
            raise AuthError(
                self.account,
                f"could not persist the renewed tokens ({e}); a rotated refresh token is lost",
            ) from e
        return merged


class ClientCredentials(_CachedRenewal):
    def __init__(
        self,
        account: str,
        endpoint: TokenEndpoint,
        *,
        scopes: frozenset[str] = frozenset(),
        clock: Clock | None = None,
    ) -> None:
        super().__init__(account, clock)
        self.endpoint = endpoint
        self.scopes = scopes
        self._tokens: TokenSet | None = None

    def _current(self) -> TokenSet | None:
        return self._tokens

    async def _renew(self, current: TokenSet | None) -> TokenSet:
        try:
            self._tokens = await self.endpoint.client_credentials(self.clock, self.scopes)
        except OAuthError as e:
            raise AuthError(
                self.account, f"client credentials grant failed ({e})", "Check client id/secret."
            ) from e
        return self._tokens


class AppInstallation(_CachedRenewal):
    """GitHub App installation token. Scopes are reported as `<permission>:<level>`."""

    skew: ClassVar[float] = 300.0

    def __init__(
        self,
        account: str,
        *,
        app_id: str,
        private_key_pem: str,
        installation_id: int,
        api_base: str = "https://api.github.com",
        transport: httpx2.AsyncBaseTransport | None = None,
        clock: Clock | None = None,
    ) -> None:
        super().__init__(account, clock)
        self.app_id = app_id
        self.private_key_pem = private_key_pem
        self.installation_id = installation_id
        self.api_base = api_base.rstrip("/")
        self.transport = transport
        self._tokens: TokenSet | None = None

    def _current(self) -> TokenSet | None:
        return self._tokens

    def app_jwt(self) -> str:
        now = int(self.clock.time())
        # Backdate iat for clock drift; GitHub allows at most 10 minutes of validity.
        claims = {"iat": now - 60, "exp": now + 540, "iss": self.app_id}
        return jwt.encode(claims, self.private_key_pem, algorithm="RS256")

    async def _renew(self, current: TokenSet | None) -> TokenSet:
        url = f"{self.api_base}/app/installations/{self.installation_id}/access_tokens"
        headers = {
            "Authorization": _bearer(self.app_jwt()),
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        async with httpx2.AsyncClient(transport=self.transport, timeout=30.0) as http:
            response = await http.post(url, headers=headers)
        if response.status_code != 201:
            raise AuthError(
                self.account,
                f"could not mint an installation token (HTTP {response.status_code})",
                "Check the app id, private key and installation id.",
            )
        body: dict[str, Any] = response.json()
        expires = datetime.fromisoformat(body["expires_at"].replace("Z", "+00:00"))
        permissions: dict[str, str] = body.get("permissions") or {}
        self._tokens = TokenSet(
            access_token=body["token"],
            expires_at=expires.timestamp(),
            scopes=frozenset(f"{name}:{level}" for name, level in permissions.items()),
        )
        return self._tokens


class CredentialAuth(httpx2.Auth):
    """Applies a credential to every request and retries once after a 401."""

    def __init__(self, credential: Credential) -> None:
        self.credential = credential

    async def async_auth_flow(self, request: httpx2.Request):  # type: ignore[override]
        header = await self.credential.authorization()
        request.headers["Authorization"] = header
        response = yield request
        if response.status_code == 401:
            request.headers["Authorization"] = await self.credential.authorization(failed=header)
            yield request

    def sync_auth_flow(self, request: httpx2.Request):  # type: ignore[override]
        raise RuntimeError("CredentialAuth is async-only; use httpx2.AsyncClient")
