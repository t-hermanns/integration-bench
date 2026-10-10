"""Google Drive credentials.

**User token** (`bench auth login gdrive`, browser sign-in with PKCE against the
"Desktop app" OAuth client): the access token expires after an hour. Google does not
rotate refresh tokens (the response omits `refresh_token`, `RotatingRefresh` keeps the
stored one), and with the consent screen "In production" they do not expire after seven
days either, so one sign-in also serves CI. `access_type=offline` plus `prompt=consent`
make Google return a refresh token on every sign-in. Token responses report the granted
scopes, so the server can hide tools the user did not grant.
"""

import httpx2

from bench_core.auth import Credential, LoginSpec, RotatingRefresh, TokenEndpoint, TokenStore
from bench_core.clock import Clock
from bench_core.permissions import Granted

from bench_gdrive.scopes import DRIVE_FILE

TOKEN_URL = "https://oauth2.googleapis.com/token"
AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"

LOGIN = LoginSpec(
    account="gdrive",
    authorize_url=AUTHORIZE_URL,
    token_url=TOKEN_URL,
    client_id_var="GDRIVE_CLIENT_ID",
    client_secret_var="GDRIVE_CLIENT_SECRET",
    scopes=(DRIVE_FILE,),
    params={"access_type": "offline", "prompt": "consent"},
)
"""`bench auth login gdrive`."""


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
