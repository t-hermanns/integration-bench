"""`bench doctor github`: the App installation (CI) and the App's OAuth client (local login).

Settings: `GITHUB_APP_ID`, `GITHUB_INSTALLATION_ID` and the private key, either as
`GITHUB_PRIVATE_KEY` (PEM text, CI secret) or `GITHUB_PRIVATE_KEY_PATH` (local file);
`GITHUB_ORG`/`GITHUB_REPO` name the sandbox repo; `GITHUB_CLIENT_ID`/`_SECRET` the login,
whose tokens `bench auth login github` keeps in the token store (`BENCH_TOKEN_DIR` or the
OS keyring). Checking the login may rotate them, which is harmless.
"""

from collections.abc import Mapping
from pathlib import Path

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
from bench_core.permissions import select_tools

from bench_github import scopes
from bench_github.auth import LOGIN as LOGIN_SPEC
from bench_github.auth import granted_permissions, installation, user_token
from bench_github.client import make_api, repo_path
from bench_github.tools import TOOLS

APP = "github app"
LOGIN = "github login"
DEFAULT_ORG = "bench-sandbox"
DEFAULT_REPO = "widgets"


def private_key(env: Mapping[str, str]) -> str | None:
    if env.get("GITHUB_PRIVATE_KEY"):
        return env["GITHUB_PRIVATE_KEY"]
    path = env.get("GITHUB_PRIVATE_KEY_PATH")
    return Path(path).read_text(encoding="utf-8") if path else None


async def checks(
    env: Mapping[str, str],
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
    store: TokenStore | None = None,
) -> list[Check]:
    store = store or default_store(env)
    return [
        await probe(APP, _app(env, transport, clock)),
        await probe(LOGIN, _login(env, store, transport, clock)),
    ]


def _repo(env: Mapping[str, str]) -> tuple[str, str]:
    full_name = f"{env.get('GITHUB_ORG') or DEFAULT_ORG}/{env.get('GITHUB_REPO') or DEFAULT_REPO}"
    owner, repo = full_name.split("/", 1)
    return owner, repo


async def _app(
    env: Mapping[str, str], transport: httpx2.AsyncBaseTransport | None, clock: Clock | None
) -> Check:
    absent = missing(env, "GITHUB_APP_ID", "GITHUB_INSTALLATION_ID")
    if not env.get("GITHUB_PRIVATE_KEY") and not env.get("GITHUB_PRIVATE_KEY_PATH"):
        absent.append("GITHUB_PRIVATE_KEY or GITHUB_PRIVATE_KEY_PATH")
    if absent:
        return not_configured(APP, absent)
    try:
        key = private_key(env)
    except OSError as exc:
        return fail(APP, f"cannot read the private key: {exc}")
    assert key is not None
    credential = installation(
        "github-doctor",
        app_id=env["GITHUB_APP_ID"],
        private_key_pem=key,
        installation_id=int(env["GITHUB_INSTALLATION_ID"]),
        transport=transport,
        clock=clock,
    )
    granted = await granted_permissions(credential)
    selection = select_tools([t.access for t in TOOLS], granted, expand=scopes.EXPAND)
    owner, repo = _repo(env)
    full_name = f"{owner}/{repo}"
    async with make_api(credential=credential, transport=transport, clock=clock) as api:
        response = await api.get(repo_path(owner, repo))
    if response.status_code != 200:
        return http_failure(APP, response, f"repo {full_name}")
    detail = (
        f"installation {env['GITHUB_INSTALLATION_ID']} sees {full_name}; "
        f"permissions {', '.join(sorted(granted or ()))}; "
        f"{len(selection.enabled)}/{len(TOOLS)} tools"
    )
    if selection.disabled:
        hidden = "; ".join(f"{name}: {why}" for name, why in selection.disabled.items())
        return fail(APP, f"{detail}; hidden: {hidden}")
    return ok(APP, detail)


async def _login(
    env: Mapping[str, str],
    store: TokenStore,
    transport: httpx2.AsyncBaseTransport | None,
    clock: Clock | None,
) -> Check:
    absent = missing(env, "GITHUB_CLIENT_ID", "GITHUB_CLIENT_SECRET")
    if absent:
        return not_configured(LOGIN, absent)
    stored = stored_login(LOGIN, store, LOGIN_SPEC.account)
    if not isinstance(stored, TokenSet):
        return stored
    credential = user_token(
        LOGIN_SPEC.account,
        store,
        client_id=env["GITHUB_CLIENT_ID"],
        client_secret=env["GITHUB_CLIENT_SECRET"],
        transport=transport,
        clock=clock,
    )
    owner, repo = _repo(env)
    async with make_api(credential=credential, transport=transport, clock=clock) as api:
        me = await api.get("/user")
        if me.status_code != 200:
            return http_failure(LOGIN, me, "/user")
        response = await api.get(repo_path(owner, repo))
        if response.status_code != 200:
            return http_failure(LOGIN, response, f"repo {owner}/{repo}")
    return ok(LOGIN, f"signed in as {me.json().get('login', '?')}; sees {owner}/{repo}")
