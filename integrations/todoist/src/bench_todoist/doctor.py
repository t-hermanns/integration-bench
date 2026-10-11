"""`bench doctor todoist`: does the live account answer?

Reads the settings from `.env` (see `sandbox.SETTINGS`) and calls `PROBE`, a cheap
authenticated read. Extend it to check the sandbox too, once `bench sandbox seed todoist`
exists.
"""

from collections.abc import Mapping

import httpx2

from bench_core.auth import TokenStore
from bench_core.clock import Clock
from bench_core.doctor import Check, fail, http_failure, missing, not_configured, ok, probe
from bench_core.sandbox import SandboxError

from bench_todoist.client import make_api
from bench_todoist.sandbox import SETTINGS, live_credential

TARGET = "todoist"
PROBE = "/"
"""Replace with a cheap authenticated endpoint (e.g. "current user")."""


async def checks(
    env: Mapping[str, str],
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
    store: TokenStore | None = None,
) -> list[Check]:
    return [await probe(TARGET, _check(env, transport, clock))]


async def _check(
    env: Mapping[str, str],
    transport: httpx2.AsyncBaseTransport | None,
    clock: Clock | None,
) -> Check:
    absent = missing(env, *SETTINGS)
    if absent:
        return not_configured(TARGET, absent)
    try:
        credential = live_credential(env, transport=transport, clock=clock)
    except SandboxError as exc:
        return fail(TARGET, str(exc))
    async with make_api(credential=credential, transport=transport, clock=clock) as api:
        response = await api.get(PROBE)
    if response.status_code != 200:
        return http_failure(TARGET, response, PROBE)
    return ok(TARGET, f"{PROBE} answered")
