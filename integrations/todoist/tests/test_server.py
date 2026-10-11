"""MCP server end to end: `Client(server)` -> tools -> client -> in-memory fake."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from mcp.client import Client
from mcp.types import CallToolResult, TextContent

from bench_core.testing import FakeClock

from bench_todoist.fake import TodoistFake, credential, seeded_fake
from bench_todoist.scopes import READ, WRITE
from bench_todoist.server import create_server
from bench_todoist.tools import DESCRIPTION_LIMIT

pytestmark = pytest.mark.anyio

TOKEN = "server-token"
LIST = "todoist_list_items"
GET = "todoist_get_item"
CREATE = "todoist_create_item"
TOOL_NAMES = {LIST, GET, CREATE}


@pytest.fixture
def fake() -> TodoistFake:
    fake = seeded_fake(FakeClock())
    fake.issue_token(TOKEN, [READ, WRITE])
    return fake


def server_for(fake: TodoistFake, **kwargs: Any):
    kwargs.setdefault("read_only", False)
    return create_server(
        credential=credential(TOKEN),
        transport=fake.transport(),
        clock=FakeClock(),
        variant="v1",
        **kwargs,
    )


@pytest.fixture
async def client(fake: TodoistFake) -> AsyncIterator[Client]:
    async with Client(server_for(fake)) as client:
        yield client


async def call(client: Client, tool: str, **arguments: Any) -> dict[str, Any]:
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, text(result)
    assert result.structured_content is not None
    return result.structured_content


def text(result: CallToolResult) -> str:
    return "\n".join(c.text for c in result.content if isinstance(c, TextContent))


# --- Registration -------------------------------------------------------------------------


async def test_tools_descriptions_and_annotations(client: Client) -> None:
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == TOOL_NAMES
    listing = tools[LIST]
    assert listing.description is not None
    assert listing.description.startswith("List items")
    assert "ctx" not in listing.input_schema["properties"]
    assert listing.input_schema["properties"]["page_size"]["maximum"] == 50
    assert listing.output_schema is not None
    create = tools[CREATE]
    assert create.annotations is not None
    assert create.annotations.read_only_hint is False
    assert listing.annotations is not None
    assert listing.annotations.read_only_hint is True


async def test_read_only_hides_write_tools(fake: TodoistFake) -> None:
    async with Client(server_for(fake, read_only=True)) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert names == TOOL_NAMES - {CREATE}


async def test_missing_scopes_hide_tools(fake: TodoistFake) -> None:
    async with Client(server_for(fake, granted=frozenset({READ}))) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert names == TOOL_NAMES - {CREATE}


# --- Tools --------------------------------------------------------------------------------


async def test_list_items_pages_with_cursor(client: Client) -> None:
    first = await call(client, LIST, query="crash", page_size=2)
    assert len(first["items"]) == 2
    assert set(first["items"][0]) == {"id", "title", "status"}
    second = await call(client, LIST, query="crash", page_size=2, cursor=first["next_cursor"])
    assert len(second["items"]) == 1
    assert "next_cursor" not in second


async def test_list_items_detailed(client: Client) -> None:
    page = await call(client, LIST, page_size=1, response_format="detailed")
    hit = page["items"][0]
    assert {"description", "updated_at", "url"} <= set(hit)


async def test_get_item_concise_and_detailed(client: Client) -> None:
    concise = await call(client, GET, item_id="1001")
    assert concise["title"] == "Crash when config file is empty"
    assert "url" not in concise
    detailed = await call(client, GET, item_id="1001", response_format="detailed")
    assert {"created_at", "updated_at", "url"} <= set(detailed)


async def test_long_descriptions_are_clipped(client: Client, fake: TodoistFake) -> None:
    item = fake.add_item("Long", description="x" * 5_000)
    result = await call(client, GET, item_id=item["id"])
    assert len(result["description"]) == DESCRIPTION_LIMIT["concise"]
    assert result["description_truncated"] is True


async def test_create_item_then_get_it(client: Client) -> None:
    created = await call(client, CREATE, title="New", description="Details")
    fetched = await call(client, GET, item_id=created["id"])
    assert (fetched["title"], fetched["description"]) == ("New", "Details")


async def test_errors_reach_the_model_as_tool_errors(client: Client) -> None:
    missing = await client.call_tool(GET, {"item_id": "9999"})
    assert missing.is_error
    assert "not found" in text(missing)
    invalid = await client.call_tool(GET, {"item_id": "../etc"})
    assert invalid.is_error
    assert "Invalid item id" in text(invalid)
    too_big = await client.call_tool(LIST, {"page_size": 500})
    assert too_big.is_error
    assert "page_size" in text(too_big)
