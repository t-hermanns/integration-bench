"""Client against cassettes: synthetic now, recorded from the sandbox later.

Assertions only use facts the sandbox seed guarantees, so they hold for both kinds.
"""

from collections.abc import AsyncIterator

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from vcr.errors import CannotOverwriteExistingCassetteException

from bench_core.permissions import NotFoundOrNoAccess, PermissionDenied
from bench_core.testing import read_manifest, stale_cassettes

from bench_todoist.client import TodoistClient
from bench_todoist.scenarios import CRASHES, SCENARIOS, SUITE

pytestmark = pytest.mark.anyio


@pytest.fixture
async def client() -> AsyncIterator[TodoistClient]:
    async with SUITE.replay_api() as api:
        yield TodoistClient(api)


def test_manifest_covers_every_scenario() -> None:
    manifest = read_manifest(SUITE.cassette_dir)
    assert set(manifest) == set(SCENARIOS), "run `bench synth todoist`"
    assert set(manifest.values()) <= {"synthetic", "recorded"}
    for name in SCENARIOS:
        assert SUITE.path(name).exists(), name


async def test_synthetic_cassettes_are_up_to_date(tmp_path) -> None:
    assert await stale_cassettes(SUITE, tmp_path) == [], "run `bench synth todoist`"


async def test_list_items(client: TodoistClient) -> None:
    with SUITE.replay("list_items") as cassette:
        first = await client.list_items(query=CRASHES, page_size=2)
        second = await client.list_items(query=CRASHES, page_size=2, cursor=first.next_cursor)
    assert cassette.all_played
    assert len(first.items) == 2
    assert first.next_cursor is not None
    assert second.next_cursor is None
    items = first.items + second.items
    assert len({i.id for i in items}) == len(items) == 3
    assert all(CRASHES in i.title.lower() for i in items)


async def test_list_no_results(client: TodoistClient) -> None:
    with SUITE.replay("list_no_results"):
        page = await client.list_items(query="kubernetes")
    assert (page.items, page.next_cursor) == ([], None)


async def test_get_item(client: TodoistClient) -> None:
    with SUITE.replay("get_item"):
        item = await client.get_item("1001")
    assert item.title == "Crash when config file is empty"
    assert item.status == "open"
    assert item.description is not None


async def test_get_item_not_found(client: TodoistClient) -> None:
    with SUITE.replay("get_item_not_found"), pytest.raises(NotFoundOrNoAccess):
        await client.get_item("9999")


async def test_create_item(client: TodoistClient) -> None:
    with SUITE.replay("create_item"):
        item = await client.create_item(
            "Export button does nothing in Firefox", description="Chrome works fine."
        )
    assert item.title == "Export button does nothing in Firefox"
    assert item.status == "open"


async def test_create_item_invalid(client: TodoistClient) -> None:
    with (
        SUITE.replay("create_item_invalid"),
        pytest.raises(ToolError, match=r"title is too long.*Fix the arguments"),
    ):
        await client.create_item("x" * 201)


async def test_create_item_forbidden(client: TodoistClient) -> None:
    with SUITE.replay("create_item_forbidden"), pytest.raises(PermissionDenied):
        await client.create_item("Should not be created")


async def test_unrecorded_request_fails_instead_of_going_out(client: TodoistClient) -> None:
    with (
        SUITE.replay("get_item"),
        pytest.raises(CannotOverwriteExistingCassetteException),
    ):
        await client.get_item("1002")
