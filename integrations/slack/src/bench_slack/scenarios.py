"""Cassette scenarios: `bench synth slack` now, `bench record slack` later.

Each scenario drives `SlackClient` against the sandbox. Expected tool errors are caught,
so the cassette holds the error response the tests then replay. Scenarios with `prepare`
shape the fake and can only be synthesized; a `teardown` undoes what a live run created
(it gets the run's result, e.g. the created item).

Slack assigns channel ids and message timestamps, so every scenario starts from names:
it finds `#bench-sandbox` in the channel list and messages by their seed text, then uses
the ids from those responses. Replays see the same responses, so the ids match.
"""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx2

from bench_core.http import ApiClient
from bench_core.testing import CassetteSuite, FakeBackend, FakeClock, Scenario, expect_tool_error

from bench_slack.client import SlackClient, make_api
from bench_slack.fake import RELEASE, SlackFake, credential, seeded_fake
from bench_slack.models import Message
from bench_slack.sandbox import sandbox_channel
from bench_slack.scopes import ALL, CHANNELS_HISTORY, CHANNELS_READ

CASSETTE_DIR = Path(__file__).resolve().parents[2] / "tests" / "cassettes"
SYNTHETIC_TOKEN = "synthetic-token"
FULL_ACCESS = list(ALL)


def _fake(clock: FakeClock) -> FakeBackend:
    fake = seeded_fake(clock)
    fake.issue_token(SYNTHETIC_TOKEN, FULL_ACCESS)
    return fake


def _api(transport: httpx2.AsyncBaseTransport | None, clock: FakeClock | None) -> ApiClient:
    # The Authorization header is redacted in cassettes, so any token replays.
    return make_api(credential=credential(SYNTHETIC_TOKEN), transport=transport, clock=clock)


Run = Callable[[SlackClient], Awaitable[Any]]
Teardown = Callable[[ApiClient, Any], Awaitable[None]]


def _scenario(
    run: Run,
    description: str,
    prepare: Callable[[SlackFake], None] | None = None,
    teardown: Teardown | None = None,
) -> Scenario:
    async def drive(api: ApiClient) -> Any:
        return await run(SlackClient(api))

    def shape(fake: FakeBackend) -> None:
        assert prepare is not None
        assert isinstance(fake, SlackFake)
        prepare(fake)

    return Scenario(drive, shape if prepare else None, description, teardown)


# --- Scenarios ----------------------------------------------------------------------------

MISSING_CHANNEL = "C0000000000"
MISSING_TS = "1234567890.123456"
POSTED = "integration-bench check: post_message works."
REPLIED = "integration-bench check: reply_in_thread works."
HISTORY_PAGE = 3
"""Five seed messages, so the newest come in two pages (live adds Slack's join notices)."""


async def _sandbox(client: SlackClient) -> str:
    return (await sandbox_channel(client)).id


async def _release(client: SlackClient, channel: str) -> Message:
    page = await client.read_history(channel, page_size=10)
    return next(m for m in page.items if m.text == RELEASE)


async def list_channels(client: SlackClient) -> None:
    await client.list_channels(page_size=100)


async def list_channels_paged(client: SlackClient) -> None:
    first = await client.list_channels(page_size=1)
    await client.list_channels(page_size=1, cursor=first.next_cursor)


async def read_history(client: SlackClient) -> None:
    channel = await _sandbox(client)
    first = await client.read_history(channel, page_size=HISTORY_PAGE)
    await client.read_history(channel, page_size=HISTORY_PAGE, cursor=first.next_cursor)


async def read_history_not_member(client: SlackClient) -> None:
    page = await client.list_channels(page_size=100)
    other = next(c for c in page.items if not c.is_member)
    await expect_tool_error(client.read_history(other.id))


async def read_history_missing_channel(client: SlackClient) -> None:
    await expect_tool_error(client.read_history(MISSING_CHANNEL))


async def read_thread(client: SlackClient) -> None:
    channel = await _sandbox(client)
    await client.read_thread(channel, (await _release(client, channel)).ts)


async def read_thread_not_found(client: SlackClient) -> None:
    await expect_tool_error(client.read_thread(await _sandbox(client), MISSING_TS))


async def post_message(client: SlackClient) -> tuple[str, Message]:
    channel = await _sandbox(client)
    return channel, await client.post_message(channel, POSTED)


async def reply_in_thread(client: SlackClient) -> tuple[str, Message]:
    channel = await _sandbox(client)
    release = await _release(client, channel)
    return channel, await client.reply_in_thread(channel, release.ts, REPLIED)


async def _delete_posted(api: ApiClient, posted: tuple[str, Message]) -> None:
    channel, message = posted
    await SlackClient(api).delete_message(channel, message.ts)


async def reply_to_a_reply(client: SlackClient) -> None:
    channel = await _sandbox(client)
    thread = await client.read_thread(channel, (await _release(client, channel)).ts)
    await expect_tool_error(client.reply_in_thread(channel, thread.items[-1].ts, "nested?"))


async def reply_thread_not_found(client: SlackClient) -> None:
    await expect_tool_error(client.reply_in_thread(await _sandbox(client), MISSING_TS, "lost"))


async def post_message_forbidden(client: SlackClient) -> None:
    await expect_tool_error(client.post_message(await _sandbox(client), "not allowed"))


def _read_only_token(fake: SlackFake) -> None:
    fake.issue_token(SYNTHETIC_TOKEN, [CHANNELS_READ, CHANNELS_HISTORY])


SCENARIOS: dict[str, Scenario] = {
    "list_channels": _scenario(list_channels, "All public channels in one page"),
    "list_channels_paged": _scenario(list_channels_paged, "One channel per page, two pages"),
    "read_history": _scenario(read_history, "Newest sandbox messages, two pages of three"),
    "read_history_not_member": _scenario(
        read_history_not_member, "History of a channel the app has not joined"
    ),
    "read_history_missing_channel": _scenario(
        read_history_missing_channel, "History of a channel id that does not exist"
    ),
    "read_thread": _scenario(read_thread, "The release thread with two replies"),
    "read_thread_not_found": _scenario(read_thread_not_found, "Thread ts that matches nothing"),
    "post_message": _scenario(
        post_message, "New top-level message in the sandbox", teardown=_delete_posted
    ),
    "reply_in_thread": _scenario(
        reply_in_thread, "Reply in the release thread", teardown=_delete_posted
    ),
    "reply_to_a_reply": _scenario(reply_to_a_reply, "thread_ts of a reply is refused"),
    "reply_thread_not_found": _scenario(
        reply_thread_not_found, "Reply to a ts that matches nothing (nothing is posted)"
    ),
    "post_message_forbidden": _scenario(
        post_message_forbidden, "Token without chat:write", prepare=_read_only_token
    ),
}

SUITE = CassetteSuite(
    integration="slack",
    cassette_dir=CASSETTE_DIR,
    scenarios=SCENARIOS,
    fake=_fake,
    api=_api,
)
