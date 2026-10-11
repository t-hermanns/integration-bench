"""Cassette scenarios: `bench synth todoist` now, `bench record todoist` later.

Each scenario drives `TodoistClient` against the sandbox. Expected tool errors are caught,
so the cassette holds the error response the tests then replay. Scenarios with `prepare`
shape the fake and can only be synthesized; a `teardown` undoes what a live run created
(it gets the run's result, e.g. the created item).
"""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx2

from bench_core.http import ApiClient
from bench_core.testing import CassetteSuite, FakeBackend, FakeClock, Scenario, expect_tool_error

from bench_todoist.client import TodoistClient, make_api
from bench_todoist.fake import TodoistFake, credential, seeded_fake
from bench_todoist.scopes import READ, WRITE

CASSETTE_DIR = Path(__file__).resolve().parents[2] / "tests" / "cassettes"
SYNTHETIC_TOKEN = "synthetic-token"
FULL_ACCESS = [READ, WRITE]


def _fake(clock: FakeClock) -> FakeBackend:
    fake = seeded_fake(clock)
    fake.issue_token(SYNTHETIC_TOKEN, FULL_ACCESS)
    return fake


def _api(transport: httpx2.AsyncBaseTransport | None, clock: FakeClock | None) -> ApiClient:
    # The Authorization header is redacted in cassettes, so any token replays.
    return make_api(credential=credential(SYNTHETIC_TOKEN), transport=transport, clock=clock)


Run = Callable[[TodoistClient], Awaitable[Any]]
Teardown = Callable[[ApiClient, Any], Awaitable[None]]


def _scenario(
    run: Run,
    description: str,
    prepare: Callable[[TodoistFake], None] | None = None,
    teardown: Teardown | None = None,
) -> Scenario:
    async def drive(api: ApiClient) -> Any:
        return await run(TodoistClient(api))

    def shape(fake: FakeBackend) -> None:
        assert prepare is not None
        assert isinstance(fake, TodoistFake)
        prepare(fake)

    return Scenario(drive, shape if prepare else None, description, teardown)


# --- Scenarios ----------------------------------------------------------------------------

CRASHES = "crash"
"""Matches three seeded items, so a page size of 2 gives two pages."""


async def list_items(client: TodoistClient) -> None:
    first = await client.list_items(query=CRASHES, page_size=2)
    await client.list_items(query=CRASHES, page_size=2, cursor=first.next_cursor)


async def list_no_results(client: TodoistClient) -> None:
    await client.list_items(query="kubernetes")


async def get_item(client: TodoistClient) -> None:
    await client.get_item("1001")


async def get_item_not_found(client: TodoistClient) -> None:
    await expect_tool_error(client.get_item("9999"))


async def create_item(client: TodoistClient) -> None:
    await client.create_item(
        "Export button does nothing in Firefox", description="Chrome works fine."
    )


async def create_item_invalid(client: TodoistClient) -> None:
    await expect_tool_error(client.create_item("x" * 201))


async def create_item_forbidden(client: TodoistClient) -> None:
    await expect_tool_error(client.create_item("Should not be created"))


def _read_only_token(fake: TodoistFake) -> None:
    fake.issue_token(SYNTHETIC_TOKEN, [READ])


SCENARIOS: dict[str, Scenario] = {
    "list_items": _scenario(list_items, "Items matching 'crash', two pages"),
    "list_no_results": _scenario(list_no_results, "List without hits"),
    "get_item": _scenario(get_item, "First seeded item"),
    "get_item_not_found": _scenario(get_item_not_found, "Missing item"),
    "create_item": _scenario(create_item, "New item with a description"),
    "create_item_invalid": _scenario(create_item_invalid, "Title over 200 characters"),
    "create_item_forbidden": _scenario(
        create_item_forbidden, "Token without the write scope", prepare=_read_only_token
    ),
}

SUITE = CassetteSuite(
    integration="todoist",
    cassette_dir=CASSETTE_DIR,
    scenarios=SCENARIOS,
    fake=_fake,
    api=_api,
)
