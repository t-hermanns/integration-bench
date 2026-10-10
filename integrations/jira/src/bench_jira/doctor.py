"""`bench doctor jira`: the API token (CI) and the OAuth 2.0 (3LO) app (local login).

Settings: `JIRA_SITE` (e.g. "bench-sandbox"), `JIRA_EMAIL`, `JIRA_API_TOKEN`,
`JIRA_PROJECT` (default "BENCH"); `JIRA_CLIENT_ID`/`_SECRET` for the login, whose tokens
`bench auth login jira` keeps in the token store (`BENCH_TOKEN_DIR` or the OS keyring).
Checking the login rotates the refresh token once the access token expired; harmless.
"""

from collections.abc import Mapping

import httpx2

from bench_core.auth import TokenSet, TokenStore, default_store
from bench_core.clock import Clock
from bench_core.doctor import (
    Check,
    http_failure,
    missing,
    not_configured,
    ok,
    probe,
    stored_login,
)

from bench_jira.auth import LOGIN as LOGIN_SPEC
from bench_jira.auth import api_token, cloud_id, user_token
from bench_jira.client import gateway_api_url, make_api, project_key, site_api_url

API_TOKEN = "jira api token"
LOGIN = "jira login"
DEFAULT_PROJECT = "BENCH"


async def checks(
    env: Mapping[str, str],
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
    store: TokenStore | None = None,
) -> list[Check]:
    store = store or default_store(env)
    return [
        await probe(API_TOKEN, _api_token(env, transport, clock)),
        await probe(LOGIN, _login(env, store, transport, clock)),
    ]


async def _api_token(
    env: Mapping[str, str], transport: httpx2.AsyncBaseTransport | None, clock: Clock | None
) -> Check:
    absent = missing(env, "JIRA_SITE", "JIRA_EMAIL", "JIRA_API_TOKEN")
    if absent:
        return not_configured(API_TOKEN, absent)
    site = env["JIRA_SITE"]
    key = project_key(env.get("JIRA_PROJECT") or DEFAULT_PROJECT)
    credential = api_token("jira-doctor", email=env["JIRA_EMAIL"], token=env["JIRA_API_TOKEN"])
    async with make_api(
        credential=credential, base_url=site_api_url(site), transport=transport, clock=clock
    ) as api:
        me = await api.get("/myself")
        if me.status_code != 200:
            return http_failure(API_TOKEN, me, "/myself")
        project = await api.get(f"/project/{key}")
        if project.status_code != 200:
            return http_failure(API_TOKEN, project, f"project {key}")
    name = me.json().get("displayName", "?")
    return ok(API_TOKEN, f"{site}.atlassian.net as {name}; project {key} visible")


async def _login(
    env: Mapping[str, str],
    store: TokenStore,
    transport: httpx2.AsyncBaseTransport | None,
    clock: Clock | None,
) -> Check:
    absent = missing(env, "JIRA_SITE", "JIRA_CLIENT_ID", "JIRA_CLIENT_SECRET")
    if absent:
        return not_configured(LOGIN, absent)
    stored = stored_login(LOGIN, store, LOGIN_SPEC.account)
    if not isinstance(stored, TokenSet):
        return stored
    credential = user_token(
        LOGIN_SPEC.account,
        store,
        client_id=env["JIRA_CLIENT_ID"],
        client_secret=env["JIRA_CLIENT_SECRET"],
        transport=transport,
        clock=clock,
    )
    site = env["JIRA_SITE"]
    key = project_key(env.get("JIRA_PROJECT") or DEFAULT_PROJECT)
    cloud = await cloud_id(credential, site, transport=transport)
    async with make_api(
        credential=credential, base_url=gateway_api_url(cloud), transport=transport, clock=clock
    ) as api:
        me = await api.get("/myself")
        if me.status_code != 200:
            return http_failure(LOGIN, me, "/myself")
        project = await api.get(f"/project/{key}")
        if project.status_code != 200:
            return http_failure(LOGIN, project, f"project {key}")
    scopes = credential.tokens()
    granted = ", ".join(sorted(scopes.scopes)) if scopes is not None else "?"
    name = me.json().get("displayName", "?")
    return ok(LOGIN, f"signed in as {name}; cloud id {cloud}; project {key} visible; {granted}")
