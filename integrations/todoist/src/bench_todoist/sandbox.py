"""`bench sandbox seed todoist` and the live API for `bench record todoist`.

`seed` creates the fake's seed tasks in the live test account (the Inbox). It is
idempotent: tasks whose title is already open are left alone.
"""

from collections.abc import Mapping

import httpx2

from bench_core.auth import Credential
from bench_core.clock import Clock
from bench_core.http import ApiClient
from bench_core.sandbox import Echo, SmokeCall, require

from bench_todoist.auth import static_token
from bench_todoist.client import TodoistClient, make_api
from bench_todoist.fake.seed import TASKS

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
    async with make_api(credential=live_credential(env), transport=transport) as api:
        client = TodoistClient(api)
        titles: set[str] = set()
        cursor: str | None = None
        while True:
            page = await client.list_tasks(cursor=cursor, page_size=100)
            titles |= {task.title for task in page.items}
            if not (cursor := page.next_cursor):
                break
        missing = [(t, d) for t, d in TASKS if t not in titles]
        for title, description in missing:
            await client.create_task(title, description=description)
        echo(f"todoist: created {len(missing)} task(s), {len(TASKS) - len(missing)} already there")


def live_api(env: Mapping[str, str]) -> ApiClient:
    """`bench record todoist`: the live account that holds the sandbox."""
    return make_api(credential=live_credential(env))


# Read-only calls for `bench smoke todoist`; args may use the previous result.
SMOKE: list[SmokeCall] = [
    SmokeCall("todoist_list_tasks", {"query": "crash"}),
    SmokeCall("todoist_get_task", lambda listed: {"task_id": listed["tasks"][0]["id"]}),
]
