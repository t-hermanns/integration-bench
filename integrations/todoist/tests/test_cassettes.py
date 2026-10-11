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
from bench_todoist.scenarios import CRASHES, FIRST_TASK, MISSING_ID, SCENARIOS, SUITE

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


async def test_list_tasks(client: TodoistClient) -> None:
    with SUITE.replay("list_tasks") as cassette:
        first = await client.list_tasks(query=CRASHES, page_size=2)
        second = await client.list_tasks(query=CRASHES, page_size=2, cursor=first.next_cursor)
    assert cassette.all_played
    assert len(first.items) == 2
    assert first.next_cursor is not None
    assert second.next_cursor is None
    tasks = first.items + second.items
    assert len({i.id for i in tasks}) == len(tasks) == 3
    assert all(CRASHES in i.title.lower() for i in tasks)


async def test_list_no_results(client: TodoistClient) -> None:
    with SUITE.replay("list_no_results"):
        page = await client.list_tasks(query="kubernetes")
    assert (page.items, page.next_cursor) == ([], None)


async def test_get_task(client: TodoistClient) -> None:
    with SUITE.replay("get_task"):
        page = await client.list_tasks(query=FIRST_TASK)
        task = await client.get_task(page.items[0].id)
    assert task.title == FIRST_TASK
    assert task.status == "open"
    assert task.description is not None


async def test_get_task_not_found(client: TodoistClient) -> None:
    with SUITE.replay("get_task_not_found"), pytest.raises(NotFoundOrNoAccess):
        await client.get_task(MISSING_ID)


async def test_create_task(client: TodoistClient) -> None:
    with SUITE.replay("create_task"):
        task = await client.create_task(
            "Export button does nothing in Firefox", description="Chrome works fine."
        )
    assert task.title == "Export button does nothing in Firefox"
    assert task.status == "open"


async def test_create_task_invalid(client: TodoistClient) -> None:
    with (
        SUITE.replay("create_task_invalid"),
        pytest.raises(ToolError, match=r"content is empty.*Fix the arguments"),
    ):
        await client.create_task("   ")


async def test_create_task_forbidden(client: TodoistClient) -> None:
    with SUITE.replay("create_task_forbidden"), pytest.raises(PermissionDenied):
        await client.create_task("Should not be created")


async def test_unrecorded_request_fails_instead_of_going_out(client: TodoistClient) -> None:
    with (
        SUITE.replay("get_task"),
        pytest.raises(CannotOverwriteExistingCassetteException),
    ):
        await client.get_task("1002")
