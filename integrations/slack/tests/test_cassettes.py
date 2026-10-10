"""Client against cassettes: synthetic now, recorded from the sandbox later.

Assertions only use facts the sandbox seed guarantees, so they hold for both kinds.
"""

from collections.abc import AsyncIterator

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from vcr.errors import CannotOverwriteExistingCassetteException

from bench_core.permissions import NotFoundOrNoAccess, PermissionDenied
from bench_core.testing import read_manifest, stale_cassettes

from bench_slack.client import SlackClient
from bench_slack.fake import MESSAGES, RELEASE, SANDBOX
from bench_slack.scenarios import (
    HISTORY_PAGE,
    MISSING_CHANNEL,
    MISSING_TS,
    POSTED,
    REPLIED,
    SCENARIOS,
    SUITE,
    _release,
    _sandbox,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
async def client() -> AsyncIterator[SlackClient]:
    async with SUITE.replay_api() as api:
        yield SlackClient(api)


def test_manifest_covers_every_scenario() -> None:
    manifest = read_manifest(SUITE.cassette_dir)
    assert set(manifest) == set(SCENARIOS), "run `bench synth slack`"
    assert set(manifest.values()) <= {"synthetic", "recorded"}
    for name in SCENARIOS:
        assert SUITE.path(name).exists(), name


async def test_synthetic_cassettes_are_up_to_date(tmp_path) -> None:
    assert await stale_cassettes(SUITE, tmp_path) == [], "run `bench synth slack`"


async def test_list_channels(client: SlackClient) -> None:
    with SUITE.replay("list_channels") as cassette:
        page = await client.list_channels(page_size=100)
    assert cassette.all_played
    sandbox = [c for c in page.items if c.name == SANDBOX]
    assert len(sandbox) == 1
    assert sandbox[0].is_member
    assert any(not c.is_member for c in page.items)
    assert page.next_cursor is None


async def test_list_channels_paged(client: SlackClient) -> None:
    with SUITE.replay("list_channels_paged") as cassette:
        first = await client.list_channels(page_size=1)
        second = await client.list_channels(page_size=1, cursor=first.next_cursor)
    assert cassette.all_played
    assert len(first.items) == len(second.items) == 1
    assert first.next_cursor is not None
    assert first.items[0].id != second.items[0].id


async def test_read_history_newest_first(client: SlackClient) -> None:
    with SUITE.replay("read_history") as cassette:
        channel = await _sandbox(client)
        first = await client.read_history(channel, page_size=HISTORY_PAGE)
        second = await client.read_history(
            channel, page_size=HISTORY_PAGE, cursor=first.next_cursor
        )
    assert cassette.all_played
    texts = [m.text for m in first.items + second.items]
    assert texts == [m.text for m in reversed(MESSAGES)]
    release = next(m for m in second.items if m.text == RELEASE)
    assert release.reply_count == 2
    assert release.is_thread_parent


async def test_read_history_not_member(client: SlackClient) -> None:
    with SUITE.replay("read_history_not_member"):
        page = await client.list_channels(page_size=100)
        other = next(c for c in page.items if not c.is_member)
        with pytest.raises(ToolError, match="not a member"):
            await client.read_history(other.id)


async def test_read_history_missing_channel(client: SlackClient) -> None:
    with SUITE.replay("read_history_missing_channel"), pytest.raises(NotFoundOrNoAccess):
        await client.read_history(MISSING_CHANNEL)


async def test_read_thread(client: SlackClient) -> None:
    with SUITE.replay("read_thread"):
        channel = await _sandbox(client)
        release = await _release(client, channel)
        thread = await client.read_thread(channel, release.ts)
    assert [m.text for m in thread.items] == [RELEASE, *MESSAGES[1].replies]
    assert all(m.thread_ts == release.ts for m in thread.items)


async def test_read_thread_not_found(client: SlackClient) -> None:
    with SUITE.replay("read_thread_not_found"):
        channel = await _sandbox(client)
        with pytest.raises(NotFoundOrNoAccess):
            await client.read_thread(channel, MISSING_TS)


async def test_post_message(client: SlackClient) -> None:
    with SUITE.replay("post_message"):
        message = await client.post_message(await _sandbox(client), POSTED)
    assert message.text == POSTED
    assert message.thread_ts is None


async def test_reply_in_thread(client: SlackClient) -> None:
    with SUITE.replay("reply_in_thread"):
        channel = await _sandbox(client)
        release = await _release(client, channel)
        reply = await client.reply_in_thread(channel, release.ts, REPLIED)
    assert reply.text == REPLIED
    assert reply.thread_ts == release.ts


async def test_reply_to_a_reply_is_refused(client: SlackClient) -> None:
    with SUITE.replay("reply_to_a_reply") as cassette:
        channel = await _sandbox(client)
        thread = await client.read_thread(channel, (await _release(client, channel)).ts)
        with pytest.raises(ToolError, match="itself a reply"):
            await client.reply_in_thread(channel, thread.items[-1].ts, "nested?")
    assert cassette.all_played  # the check happens before anything is posted


async def test_reply_to_a_missing_thread_posts_nothing(client: SlackClient) -> None:
    with SUITE.replay("reply_thread_not_found") as cassette:
        channel = await _sandbox(client)
        with pytest.raises(NotFoundOrNoAccess):
            await client.reply_in_thread(channel, MISSING_TS, "lost")
    assert cassette.all_played
    assert not any("chat.postMessage" in r.uri for r in cassette.requests)


async def test_post_message_forbidden(client: SlackClient) -> None:
    with SUITE.replay("post_message_forbidden"):
        channel = await _sandbox(client)
        with pytest.raises(PermissionDenied):
            await client.post_message(channel, "not allowed")


async def test_unrecorded_request_fails_instead_of_going_out(client: SlackClient) -> None:
    with (
        SUITE.replay("list_channels"),
        pytest.raises(CannotOverwriteExistingCassetteException),
    ):
        await client.list_channels(page_size=7)
