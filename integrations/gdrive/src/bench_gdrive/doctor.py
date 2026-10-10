"""`bench doctor gdrive`: the Desktop OAuth client and the stored login.

Settings: `GDRIVE_CLIENT_ID`/`_SECRET`; `bench auth login gdrive` keeps the tokens in the
token store (`BENCH_TOKEN_DIR` or the OS keyring). Checking refreshes the access token
once it expired; Google keeps the refresh token, so that is harmless. The check also
looks for the sandbox folder that `bench sandbox seed gdrive` creates.
"""

from collections.abc import Mapping

import httpx2

from bench_core.auth import TokenSet, TokenStore, default_store
from bench_core.clock import Clock
from bench_core.doctor import (
    Check,
    fail,
    http_failure,
    missing,
    not_configured,
    ok,
    probe,
    stored_login,
)

from bench_gdrive.auth import LOGIN as LOGIN_SPEC
from bench_gdrive.auth import user_token
from bench_gdrive.client import GdriveClient, make_api
from bench_gdrive.fake.seed import SANDBOX
from bench_gdrive.sandbox import find_sandbox

LOGIN = "gdrive login"


async def checks(
    env: Mapping[str, str],
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
    store: TokenStore | None = None,
) -> list[Check]:
    store = store or default_store(env)
    return [await probe(LOGIN, _login(env, store, transport, clock))]


async def _login(
    env: Mapping[str, str],
    store: TokenStore,
    transport: httpx2.AsyncBaseTransport | None,
    clock: Clock | None,
) -> Check:
    absent = missing(env, "GDRIVE_CLIENT_ID", "GDRIVE_CLIENT_SECRET")
    if absent:
        return not_configured(LOGIN, absent)
    stored = stored_login(LOGIN, store, LOGIN_SPEC.account)
    if not isinstance(stored, TokenSet):
        return stored
    credential = user_token(
        LOGIN_SPEC.account,
        store,
        client_id=env["GDRIVE_CLIENT_ID"],
        client_secret=env["GDRIVE_CLIENT_SECRET"],
        transport=transport,
        clock=clock,
    )
    async with make_api(credential=credential, transport=transport, clock=clock) as api:
        about = await api.get("/drive/v3/about", params={"fields": "user(displayName)"})
        if about.status_code != 200:
            return http_failure(LOGIN, about, "/drive/v3/about")
        sandbox = await find_sandbox(GdriveClient(api))
    tokens = credential.tokens()
    granted = ", ".join(sorted(tokens.scopes)) if tokens is not None else "?"
    name = about.json().get("user", {}).get("displayName", "?")
    if sandbox is None:
        return fail(
            LOGIN, f"signed in as {name}, but no {SANDBOX} folder; run `bench sandbox seed gdrive`"
        )
    return ok(LOGIN, f"signed in as {name}; {SANDBOX} folder present; {granted}")
