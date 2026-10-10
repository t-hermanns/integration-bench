"""Minimal OAuth 2 token endpoint client (authorization_code, refresh_token and
client_credentials grants).

Handles provider quirks seen in our four APIs: GitHub answers form-encoded unless asked
for JSON, Slack reports errors as HTTP 200 with `ok: false` and lists scopes comma-
separated, and providers that do not rotate refresh tokens omit `refresh_token`.
"""

import re
from typing import Any, Literal

import httpx2

from bench_core.auth.store import TokenSet
from bench_core.clock import Clock


class OAuthError(Exception):
    def __init__(self, error: str, description: str | None = None, status: int | None = None):
        self.error = error
        self.description = description
        self.status = status
        super().__init__(f"{error}: {description}" if description else error)

    @property
    def is_invalid_grant(self) -> bool:
        # Slack calls an unusable refresh token `invalid_refresh_token`, GitHub
        # `bad_refresh_token` (sent with HTTP 200).
        return self.error in ("invalid_grant", "invalid_refresh_token", "bad_refresh_token")


def parse_scopes(raw: str | None) -> frozenset[str]:
    return frozenset(s for s in re.split(r"[\s,]+", raw or "") if s)


class TokenEndpoint:
    def __init__(
        self,
        url: str,
        *,
        client_id: str,
        client_secret: str | None = None,
        client_auth: Literal["body", "basic"] = "body",
        transport: httpx2.AsyncBaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.url = url
        self.client_id = client_id
        self.client_secret = client_secret
        self.client_auth = client_auth
        self.transport = transport
        self.timeout = timeout

    async def authorization_code(
        self, code: str, *, redirect_uri: str, code_verifier: str | None, clock: Clock
    ) -> TokenSet:
        form = {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri}
        if code_verifier is not None:
            form["code_verifier"] = code_verifier
        return await self._grant(form, clock)

    async def refresh(self, refresh_token: str, clock: Clock) -> TokenSet:
        return await self._grant(
            {"grant_type": "refresh_token", "refresh_token": refresh_token}, clock
        )

    async def client_credentials(
        self, clock: Clock, scopes: frozenset[str] = frozenset()
    ) -> TokenSet:
        form = {"grant_type": "client_credentials"}
        if scopes:
            form["scope"] = " ".join(sorted(scopes))
        return await self._grant(form, clock)

    async def _grant(self, form: dict[str, str], clock: Clock) -> TokenSet:
        extra: dict[str, Any] = {}
        if self.client_auth == "basic" and self.client_secret is not None:
            extra["auth"] = httpx2.BasicAuth(self.client_id, self.client_secret)
        else:
            form = {**form, "client_id": self.client_id}
            if self.client_secret is not None:
                form["client_secret"] = self.client_secret
        async with httpx2.AsyncClient(transport=self.transport, timeout=self.timeout) as http:
            response = await http.post(
                self.url, data=form, headers={"Accept": "application/json"}, **extra
            )
        try:
            body: dict[str, Any] = response.json()
        except ValueError:
            raise OAuthError(
                "invalid_response", response.text[:200], response.status_code
            ) from None
        if not response.is_success or body.get("ok") is False or "error" in body:
            raise OAuthError(
                str(body.get("error", f"http_{response.status_code}")),
                body.get("error_description"),
                response.status_code,
            )
        user = body.get("authed_user")
        if "access_token" not in body and isinstance(user, dict) and "access_token" in user:
            # Slack's user-token installs (no bot scopes) nest the tokens in `authed_user`.
            body = user
        access = body.get("access_token")
        if not isinstance(access, str) or not access:
            raise OAuthError(
                "invalid_response", "no access_token in response", response.status_code
            )
        expires_in = body.get("expires_in")
        return TokenSet(
            access_token=access,
            refresh_token=body.get("refresh_token"),
            expires_at=clock.time() + float(expires_in) if expires_in is not None else None,
            scopes=parse_scopes(body.get("scope")),
        )
