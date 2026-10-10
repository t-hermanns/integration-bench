"""`bench sandbox seed slack` and the live API for `bench record slack`.

Both use the `bench-ci` bot token (`SLACK_BOT_TOKEN`). The sandbox is the public channel
`#bench-sandbox`, which a workspace member creates and invites the bot to (`/invite
@bench-ci`). Seeding posts the fake's messages and thread replies there, oldest first.
A rerun reads the channel and only checks it: Slack's join notices are ignored, any
other difference is an error (delete the extra messages in Slack, then seed again).
"""

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager

import httpx2

from bench_core.clock import Clock
from bench_core.http import ApiClient
from bench_core.sandbox import Echo, SandboxError, require

from bench_slack.auth import bot_token
from bench_slack.client import SlackClient, make_api
from bench_slack.fake.seed import MESSAGES, SANDBOX
from bench_slack.models import Channel, Message, Page

SETTINGS = ("SLACK_BOT_TOKEN",)
EXPECTED = [(m.text, m.replies) for m in MESSAGES]
"""What `threads` reads from a seeded sandbox."""


def bot_api(
    env: Mapping[str, str],
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
) -> ApiClient:
    require(env, *SETTINGS)
    return make_api(credential=bot_token(env["SLACK_BOT_TOKEN"]), transport=transport, clock=clock)


async def _all[T](fetch: Callable[[str | None], Awaitable[Page[T]]]) -> AsyncIterator[T]:
    """Every item of a cursor-paginated listing; `fetch(cursor)` returns one page."""
    cursor: str | None = None
    while True:
        page = await fetch(cursor)
        for item in page.items:
            yield item
        if page.next_cursor is None:
            return
        cursor = page.next_cursor


async def find_sandbox(client: SlackClient) -> Channel | None:
    async for channel in _all(lambda cursor: client.list_channels(cursor=cursor, page_size=100)):
        if channel.name == SANDBOX:
            return channel
    return None


async def threads(client: SlackClient, channel: str) -> list[tuple[str, tuple[str, ...]]]:
    """(text, reply texts) per top-level message, oldest first, like `MESSAGES`."""
    top: list[Message] = [
        m async for m in _all(lambda c: client.read_history(channel, cursor=c, page_size=100))
    ]
    result = []
    for message in reversed(top):
        replies: tuple[str, ...] = ()
        if message.reply_count:
            thread = await client.read_thread(channel, message.ts, page_size=100)
            replies = tuple(m.text for m in thread.items if m.ts != message.ts)
        result.append((message.text, replies))
    return result


async def sandbox_channel(client: SlackClient) -> Channel:
    channel = await find_sandbox(client)
    if channel is None:
        raise SandboxError(f"no public channel #{SANDBOX}; create it in Slack first")
    if not channel.is_member:
        raise SandboxError(f"the bot is not in #{SANDBOX}; run `/invite @bench-ci` there")
    return channel


async def seed(
    env: Mapping[str, str],
    *,
    echo: Echo = print,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> None:
    async with bot_api(env, transport=transport) as api:
        client = SlackClient(api)
        channel = await sandbox_channel(client)
        actual = await threads(client, channel.id)
        if actual == EXPECTED:
            echo(f"slack: #{SANDBOX} already holds the {len(MESSAGES)} seed messages")
            return
        if actual:
            raise SandboxError(
                f"#{SANDBOX} holds {len(actual)} other messages; delete them in Slack and "
                "seed again"
            )
        for message in MESSAGES:
            parent = await client.post_message(channel.id, message.text)
            for reply in message.replies:
                await client.reply_in_thread(channel.id, parent.ts, reply)
            echo(f"#{SANDBOX}: {message.text[:50]}")
    echo(f"slack: posted {len(MESSAGES)} seed messages to #{SANDBOX}")


@asynccontextmanager
async def live_api(env: Mapping[str, str]) -> AsyncIterator[ApiClient]:
    """`bench record slack`: the bot token, which can read and post in the sandbox."""
    async with bot_api(env) as api:
        yield api
