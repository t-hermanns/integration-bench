"""GitHub credentials: App user tokens (local) and App installation tokens (CI).

- **User token** (`bench auth login github`, browser + PKCE): expires after 8 hours, the
  refresh token rotates on every use. GitHub's token endpoint answers errors with HTTP 200
  (`bad_refresh_token`), which `TokenEndpoint` turns into an `AuthError`.
- **Installation token** (CI): minted from the App's private key, valid for one hour, and
  it reports the installation's permissions, so the server can hide tools up front.
  User tokens do not report permissions; then every tool stays registered.
"""

import httpx2

from bench_core.auth import AppInstallation, Credential, RotatingRefresh, TokenEndpoint, TokenStore
from bench_core.clock import Clock
from bench_core.permissions import Granted
from bench_github.client import BASE_URL

TOKEN_URL = "https://github.com/login/oauth/access_token"


def user_token(
    account: str,
    store: TokenStore,
    *,
    client_id: str,
    client_secret: str | None,
    token_url: str = TOKEN_URL,
    transport: httpx2.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
) -> RotatingRefresh:
    endpoint = TokenEndpoint(
        token_url, client_id=client_id, client_secret=client_secret, transport=transport
    )
    return RotatingRefresh(account, store, endpoint, clock=clock)


def installation(
    account: str,
    *,
    app_id: str,
    private_key_pem: str,
    installation_id: int,
    api_base: str = BASE_URL,
    transport: httpx2.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
) -> AppInstallation:
    return AppInstallation(
        account,
        app_id=app_id,
        private_key_pem=private_key_pem,
        installation_id=installation_id,
        api_base=api_base,
        transport=transport,
        clock=clock,
    )


async def granted_permissions(credential: Credential) -> Granted:
    """App permissions like `issues:write` for installation tokens, else None (unknown)."""
    if not isinstance(credential, AppInstallation):
        return None
    await credential.authorization()
    tokens = credential.tokens()
    return tokens.scopes if tokens is not None else None
