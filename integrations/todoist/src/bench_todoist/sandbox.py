"""`bench sandbox seed todoist` and the live API for `bench record todoist`.

`seed` creates the fake's seed data in the live test account. It must be idempotent: an
already seeded sandbox is left alone, anything else that differs is an error. Until it is
written, it refuses to run.
"""

from collections.abc import Mapping

import httpx2

from bench_core.auth import Credential
from bench_core.clock import Clock
from bench_core.http import ApiClient
from bench_core.sandbox import Echo, SandboxError, SmokeCall, require

from bench_todoist.auth import static_token
from bench_todoist.client import make_api

SETTINGS = ("TODOIST_TOKEN",)


def live_credential(
    env: Mapping[str, str],
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
) -> Credential:
    """The live account from `.env`."""
    require(env, *SETTINGS)
    return static_token("todoist", env["TODOIST_TOKEN"])


async def seed(
    env: Mapping[str, str],
    *,
    echo: Echo = print,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> None:
    raise SandboxError("todoist: seeding is not written yet (see sandbox.py)")


def live_api(env: Mapping[str, str]) -> ApiClient:
    """`bench record todoist`: the live account that holds the sandbox."""
    return make_api(credential=live_credential(env))


# Read-only calls for `bench smoke todoist`; args may use the previous result.
SMOKE: list[SmokeCall] = []
