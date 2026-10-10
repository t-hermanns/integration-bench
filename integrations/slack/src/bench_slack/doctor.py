"""`bench doctor slack`: the CI bot token and the developer sign-in.

`slack bot` calls `auth.test` with `SLACK_BOT_TOKEN` and checks that the bot is in the
seeded `#bench-sandbox`. `slack login` uses the user token from `bench auth login slack`
(app `bench-dev`, `SLACK_CLIENT_ID`); checking refreshes it once it expired, which
rotates the refresh token, so the check also proves the rotation path works.
"""

from collections.abc import Mapping

import httpx2

from bench_core.auth import TokenSet, TokenStore, default_store
from bench_core.clock import Clock
from bench_core.doctor import Check, fail, missing, not_configured, ok, probe, stored_login
from bench_core.sandbox import SandboxError

from bench_slack.auth import LOGIN as LOGIN_SPEC
from bench_slack.auth import user_token
from bench_slack.client import SlackClient, make_api
from bench_slack.fake.seed import MESSAGES, SANDBOX
from bench_slack.sandbox import SETTINGS, bot_api, sandbox_channel, threads

BOT = "slack bot"
LOGIN = "slack login"


async def checks(
    env: Mapping[str, str],
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
    store: TokenStore | None = None,
) -> list[Check]:
    store = store or default_store(env)
    return [
        await probe(BOT, _bot(env, transport, clock)),
        await probe(LOGIN, _login(env, store, transport, clock)),
    ]


async def _bot(
    env: Mapping[str, str], transport: httpx2.AsyncBaseTransport | None, clock: Clock | None
) -> Check:
    absent = missing(env, *SETTINGS)
    if absent:
        return not_configured(BOT, absent)
    async with bot_api(env, transport=transport, clock=clock) as api:
        client = SlackClient(api)
        who = await client.auth_test()
        try:
            channel = await sandbox_channel(client)
        except SandboxError as exc:
            return fail(BOT, f"{who.get('user')}@{who.get('team')}: {exc}")
        seeded = await threads(client, channel.id) == [(m.text, m.replies) for m in MESSAGES]
    state = "seeded" if seeded else "not seeded (run `bench sandbox seed slack`)"
    detail = f"{who.get('user')}@{who.get('team')}; #{SANDBOX} {state}"
    return ok(BOT, detail) if seeded else fail(BOT, detail)


async def _login(
    env: Mapping[str, str],
    store: TokenStore,
    transport: httpx2.AsyncBaseTransport | None,
    clock: Clock | None,
) -> Check:
    absent = missing(env, LOGIN_SPEC.client_id_var)
    if absent:
        return not_configured(LOGIN, absent)
    stored = stored_login(LOGIN, store, LOGIN_SPEC.account)
    if not isinstance(stored, TokenSet):
        return stored
    credential = user_token(
        LOGIN_SPEC.account,
        store,
        client_id=env[LOGIN_SPEC.client_id_var],
        transport=transport,
        clock=clock,
    )
    async with make_api(credential=credential, transport=transport, clock=clock) as api:
        who = await SlackClient(api).auth_test()
    tokens = credential.tokens()
    granted = ", ".join(sorted(tokens.scopes)) if tokens is not None else "?"
    return ok(LOGIN, f"signed in as {who.get('user')}@{who.get('team')}; {granted}")
