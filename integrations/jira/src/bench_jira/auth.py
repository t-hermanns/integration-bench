"""Jira credentials.

**User token** (`bench auth login jira`, browser sign-in): the access token expires
and the refresh token rotates on every use; `RotatingRefresh` stores the new pair before
the call is retried. Token responses report the granted scopes, so the server can hide
tools the user did not grant.
"""

import httpx2

from bench_core.auth import Credential, RotatingRefresh, TokenEndpoint, TokenStore
from bench_core.clock import Clock
from bench_core.permissions import Granted

TOKEN_URL = "https://auth.atlassian.com/oauth/token"


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


async def granted_scopes(credential: Credential) -> Granted:
    """Scopes of the current access token, or None when unknown (keep every tool)."""
    if not isinstance(credential, RotatingRefresh):
        return None
    await credential.authorization()
    tokens = credential.tokens()
    return tokens.scopes if tokens is not None and tokens.scopes else None
