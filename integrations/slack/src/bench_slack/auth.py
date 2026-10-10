"""Slack credentials: a bot token for CI, a rotating user token for development.

**Bot token** (`SLACK_BOT_TOKEN`, app `bench-ci`): an internal app's `xoxb-` token that
never expires; CI and `bench record`/`bench sandbox seed` use it.

**User token** (`bench auth login slack`, app `bench-dev`): Slack only allows a plain
`http://localhost` redirect for apps with PKCE, and such "desktop" sign-ins can only ask
for user scopes (`user_scope`, not `scope`) and use no client secret. The access token
expires after 12 hours, the refresh token rotates on every use and expires after 30
days; `RotatingRefresh` stores the new pair before the call is retried. Token responses
report the granted scopes, so the server can hide tools the user did not grant.
"""

import httpx2

from bench_core.auth import (
    Credential,
    LoginSpec,
    RotatingRefresh,
    StaticToken,
    TokenEndpoint,
    TokenStore,
)
from bench_core.clock import Clock
from bench_core.permissions import Granted

from bench_slack.scopes import ALL

TOKEN_URL = "https://slack.com/api/oauth.v2.access"
AUTHORIZE_URL = "https://slack.com/oauth/v2/authorize"

LOGIN = LoginSpec(
    account="slack",
    authorize_url=AUTHORIZE_URL,
    token_url=TOKEN_URL,
    client_id_var="SLACK_CLIENT_ID",
    params={"user_scope": ",".join(ALL)},
)
"""`bench auth login slack` (app `bench-dev`, PKCE, no client secret)."""


def user_token(
    account: str,
    store: TokenStore,
    *,
    client_id: str,
    client_secret: str | None = None,
    token_url: str = TOKEN_URL,
    transport: httpx2.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
) -> RotatingRefresh:
    endpoint = TokenEndpoint(
        token_url, client_id=client_id, client_secret=client_secret, transport=transport
    )
    return RotatingRefresh(account, store, endpoint, clock=clock)


def bot_token(token: str) -> StaticToken:
    """The `bench-ci` bot token (`SLACK_BOT_TOKEN`)."""
    return StaticToken("slack-bot", token)


async def granted_scopes(credential: Credential) -> Granted:
    """Scopes of the current access token, or None when unknown (keep every tool)."""
    if not isinstance(credential, RotatingRefresh):
        return None
    await credential.authorization()
    tokens = credential.tokens()
    return tokens.scopes if tokens is not None and tokens.scopes else None
