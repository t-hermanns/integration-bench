"""MCP server end to end: `Client(server)` -> tools -> client -> in-memory fake."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from mcp.client import Client
from mcp.types import CallToolResult, TextContent

from bench_core.testing import FakeClock

from bench_slack.fake import (
    GENERAL,
    MESSAGES,
    OTHER_CHANNELS,
    RELEASE,
    SANDBOX,
    SlackFake,
    credential,
    seeded_fake,
)
from bench_slack.scopes import ALL, CHANNELS_HISTORY, CHANNELS_READ
from bench_slack.server import create_server
from bench_slack.tools import TEXT_LIMIT

pytestmark = pytest.mark.anyio

TOKEN = "server-token"
LIST = "slack_list_channels"
HISTORY = "slack_read_channel_history"
THREAD = "slack_read_thread"
POST = "slack_post_message"
REPLY = "slack_reply_in_thread"
TOOL_NAMES = {LIST, HISTORY, THREAD, POST, REPLY}
MISSING_TS = "1234567890.123456"


@pytest.fixture
def fake() -> SlackFake:
    fake = seeded_fake(FakeClock())
    fake.issue_token(TOKEN, list(ALL))
    return fake


def server_for(fake: SlackFake, **kwargs: Any):
    kwargs.setdefault("read_only", False)
    return create_server(
        credential=credential(TOKEN),
        transport=fake.transport(),
        clock=FakeClock(),
        variant="v1",
        **kwargs,
    )


@pytest.fixture
async def client(fake: SlackFake) -> AsyncIterator[Client]:
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
    assert listing.description.startswith("List the workspace's public channels")
    assert "ctx" not in listing.input_schema["properties"]
    assert listing.input_schema["properties"]["page_size"]["maximum"] == 100
    assert listing.output_schema is not None
    for name in (POST, REPLY):
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.read_only_hint is False
    assert listing.annotations is not None
    assert listing.annotations.read_only_hint is True


async def test_read_only_hides_write_tools(fake: SlackFake) -> None:
    async with Client(server_for(fake, read_only=True)) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert names == TOOL_NAMES - {POST, REPLY}


async def test_missing_scopes_hide_tools(fake: SlackFake) -> None:
    async with Client(server_for(fake, granted=frozenset({CHANNELS_READ}))) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert names == {LIST}
    granted = frozenset({CHANNELS_READ, CHANNELS_HISTORY})
    async with Client(server_for(fake, granted=granted)) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert names == {LIST, HISTORY, THREAD}


# --- Tools --------------------------------------------------------------------------------


async def sandbox_id(client: Client) -> str:
    channels = (await call(client, LIST))["channels"]
    return next(c["id"] for c in channels if c["name"] == SANDBOX)


async def release_ts(client: Client, channel: str) -> str:
    messages = (await call(client, HISTORY, channel_id=channel))["messages"]
    return next(m["ts"] for m in messages if m["text"] == RELEASE)


async def test_list_channels_concise_and_detailed(client: Client) -> None:
    concise = (await call(client, LIST))["channels"]
    assert {c["name"] for c in concise} == {SANDBOX, *OTHER_CHANNELS}
    assert set(concise[0]) == {"id", "name", "is_member"}
    detailed = (await call(client, LIST, response_format="detailed"))["channels"]
    sandbox = next(c for c in detailed if c["name"] == SANDBOX)
    assert sandbox["purpose"].startswith("Test channel")
    assert sandbox["num_members"] == 2


async def test_list_channels_pages_with_cursor(client: Client) -> None:
    first = await call(client, LIST, page_size=3)
    assert len(first["channels"]) == 3
    second = await call(client, LIST, page_size=3, cursor=first["next_cursor"])
    assert len(second["channels"]) == 1
    assert "next_cursor" not in second


async def test_history_is_newest_first_with_thread_info(client: Client) -> None:
    channel = await sandbox_id(client)
    messages = (await call(client, HISTORY, channel_id=channel))["messages"]
    assert [m["text"] for m in messages] == [m.text for m in reversed(MESSAGES)]
    release = next(m for m in messages if m["text"] == RELEASE)
    assert release["reply_count"] == 2
    assert release["thread_ts"] == release["ts"]
    assert set(messages[0]) == {"ts", "user", "text"}
    detailed = await call(client, HISTORY, channel_id=channel, response_format="detailed")
    assert detailed["messages"][0]["posted_at"].startswith("2026-")


async def test_read_thread(client: Client) -> None:
    channel = await sandbox_id(client)
    parent = await release_ts(client, channel)
    thread = await call(client, THREAD, channel_id=channel, thread_ts=parent)
    assert [m["text"] for m in thread["messages"]] == [RELEASE, *MESSAGES[1].replies]


async def test_long_texts_are_clipped(client: Client, fake: SlackFake) -> None:
    channel = await sandbox_id(client)
    fake.add_message(channel, "x" * (TEXT_LIMIT["concise"] + 10))
    newest = (await call(client, HISTORY, channel_id=channel, page_size=1))["messages"][0]
    assert len(newest["text"]) <= TEXT_LIMIT["concise"]
    assert newest["text_truncated"] is True


async def test_post_then_read_it_back(client: Client) -> None:
    channel = await sandbox_id(client)
    posted = await call(client, POST, channel_id=channel, text="Deploy finished.")
    assert posted == {"channel_id": channel, "ts": posted["ts"]}
    newest = (await call(client, HISTORY, channel_id=channel, page_size=1))["messages"][0]
    assert (newest["ts"], newest["text"]) == (posted["ts"], "Deploy finished.")


async def test_reply_then_read_the_thread(client: Client) -> None:
    channel = await sandbox_id(client)
    parent = await release_ts(client, channel)
    reply = await call(client, REPLY, channel_id=channel, thread_ts=parent, text="Shipped.")
    assert reply["thread_ts"] == parent
    thread = (await call(client, THREAD, channel_id=channel, thread_ts=parent))["messages"]
    assert thread[-1]["text"] == "Shipped."
    history = (await call(client, HISTORY, channel_id=channel))["messages"]
    assert "Shipped." not in [m["text"] for m in history]


async def test_errors_reach_the_model_as_tool_errors(client: Client) -> None:
    channel = await sandbox_id(client)
    reply = {"channel_id": channel, "thread_ts": MISSING_TS, "text": "hi"}
    cases: list[tuple[str, dict[str, Any], str]] = [
        (HISTORY, {"channel_id": "bench-sandbox"}, "not the channel name"),
        (HISTORY, {"channel_id": "C0000000000"}, "not found"),
        (THREAD, {"channel_id": channel, "thread_ts": "yesterday"}, "Invalid thread_ts"),
        (THREAD, {"channel_id": channel, "thread_ts": MISSING_TS}, "not found"),
        (POST, {"channel_id": channel, "text": "  "}, "must not be empty"),
        (POST, {"channel_id": channel, "text": "x" * 4001}, "limit is 4000"),
        (REPLY, reply, "not found"),
        (LIST, {"page_size": 500}, "page_size"),
    ]
    for tool, arguments, expected in cases:
        result = await client.call_tool(tool, arguments)
        assert result.is_error, (tool, arguments)
        assert expected in text(result), (tool, text(result))


async def test_channels_the_app_has_not_joined(client: Client) -> None:
    channels = (await call(client, LIST))["channels"]
    general = next(c["id"] for c in channels if c["name"] == GENERAL)
    for tool, extra in [(HISTORY, {}), (POST, {"text": "hello"})]:
        result = await client.call_tool(tool, {"channel_id": general, **extra})
        assert result.is_error
        assert "not a member" in text(result)
