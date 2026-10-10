"""Slack MCP tools: thin wrappers that shape client results for the model.

`response_format="concise"` returns what the model needs for its next step (channel ids,
message `ts`, author ids, text); `"detailed"` adds topics, member counts and posting
times. Fields that a format leaves out are omitted from the output instead of being sent
as null. Long texts are clipped with a flag that says so. Authors are Slack user ids
(`U...`), or the bot id for bot messages without a user.

Every tool call gets one wait budget for rate-limit retries (`TOOL_WAIT_BUDGET`).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

from mcp.server.mcpserver import Context
from pydantic import BaseModel, Field

from bench_core.mcp import ResponseFormat, ToolSpec, clip, lifespan_state, omit, tool_call

from bench_slack import scopes
from bench_slack.client import PROVIDER, SlackClient
from bench_slack.models import Channel, Message

TOOL_WAIT_BUDGET = 20.0
"""Seconds one tool call may spend waiting on rate limits before it reports them."""

TEXT_LIMIT = {"concise": 1_500, "detailed": 10_000}


@dataclass
class SlackState:
    client: SlackClient


@contextmanager
def _slack(ctx: Context[SlackState]) -> Iterator[SlackClient]:
    """The client for one tool call: one wait budget for all its requests."""
    client = lifespan_state(ctx).client
    with tool_call(client.api, provider=PROVIDER, wait_budget=TOOL_WAIT_BUDGET):
        yield client


# --- Output models ------------------------------------------------------------------------


class Output(BaseModel):
    pass


class ChannelOut(Output):
    id: str
    name: str
    is_member: bool
    topic: str | None = omit()
    purpose: str | None = omit()
    num_members: int | None = omit()


class ChannelList(Output):
    channels: list[ChannelOut]
    next_cursor: str | None = omit()


class MessageOut(Output):
    ts: str
    user: str | None
    text: str
    text_truncated: bool | None = omit()
    thread_ts: str | None = omit()
    reply_count: int | None = omit()
    posted_at: datetime | None = omit()


class MessageList(Output):
    messages: list[MessageOut]
    next_cursor: str | None = omit()


class Posted(Output):
    channel_id: str
    ts: str
    thread_ts: str | None = omit()


# --- Conversions --------------------------------------------------------------------------


def _channel(channel: Channel, fmt: ResponseFormat) -> ChannelOut:
    out = ChannelOut(id=channel.id, name=channel.name, is_member=channel.is_member)
    if fmt == "detailed":
        out = out.model_copy(
            update={
                "topic": channel.topic,
                "purpose": channel.purpose,
                "num_members": channel.num_members,
            }
        )
    return out


def _message(message: Message, fmt: ResponseFormat) -> MessageOut:
    text, clipped = clip(message.text, TEXT_LIMIT[fmt])
    out = MessageOut(
        ts=message.ts,
        user=message.user or message.bot_id,
        text=text or "",
        text_truncated=clipped,
        thread_ts=message.thread_ts,
        reply_count=message.reply_count or None,
    )
    if fmt == "detailed":
        out = out.model_copy(update={"posted_at": message.posted_at})
    return out


def _messages(messages: list[Message], next_cursor: str | None, fmt: ResponseFormat) -> MessageList:
    return MessageList(messages=[_message(m, fmt) for m in messages], next_cursor=next_cursor)


# --- Tools --------------------------------------------------------------------------------

PageSize = Annotated[int, Field(ge=1, le=100)]


async def list_channels(
    ctx: Context[SlackState],
    cursor: str | None = None,
    page_size: PageSize = 50,
    response_format: ResponseFormat = "concise",
) -> ChannelList:
    with _slack(ctx) as client:
        page = await client.list_channels(cursor=cursor, page_size=page_size)
    return ChannelList(
        channels=[_channel(c, response_format) for c in page.items], next_cursor=page.next_cursor
    )


async def read_channel_history(
    channel_id: str,
    ctx: Context[SlackState],
    cursor: str | None = None,
    page_size: PageSize = 20,
    response_format: ResponseFormat = "concise",
) -> MessageList:
    with _slack(ctx) as client:
        page = await client.read_history(channel_id, cursor=cursor, page_size=page_size)
    return _messages(page.items, page.next_cursor, response_format)


async def read_thread(
    channel_id: str,
    thread_ts: str,
    ctx: Context[SlackState],
    cursor: str | None = None,
    page_size: PageSize = 50,
    response_format: ResponseFormat = "concise",
) -> MessageList:
    with _slack(ctx) as client:
        page = await client.read_thread(channel_id, thread_ts, cursor=cursor, page_size=page_size)
    return _messages(page.items, page.next_cursor, response_format)


async def post_message(channel_id: str, text: str, ctx: Context[SlackState]) -> Posted:
    with _slack(ctx) as client:
        message = await client.post_message(channel_id, text)
    return Posted(channel_id=channel_id, ts=message.ts)


async def reply_in_thread(
    channel_id: str, thread_ts: str, text: str, ctx: Context[SlackState]
) -> Posted:
    with _slack(ctx) as client:
        message = await client.reply_in_thread(channel_id, thread_ts, text)
    return Posted(channel_id=channel_id, ts=message.ts, thread_ts=message.thread_ts)


TOOLS = [
    ToolSpec(list_channels, scopes.LIST_CHANNELS),
    ToolSpec(read_channel_history, scopes.READ_CHANNEL_HISTORY),
    ToolSpec(read_thread, scopes.READ_THREAD),
    ToolSpec(post_message, scopes.POST_MESSAGE),
    ToolSpec(reply_in_thread, scopes.REPLY_IN_THREAD),
]
