"""Typed client for the slice of the Slack Web API the tools need.

Every method validates channel ids and message timestamps before calling Slack, returns
models from `bench_slack.models`, and turns error responses into `ToolError`s the model
can act on. Slack answers almost every error with HTTP 200 and `{"ok": false, "error":
code}`: auth codes -> sign in again, `missing_scope` -> `PermissionDenied`, `*_not_found`
-> `NotFoundOrNoAccess`, `not_in_channel` -> pick a channel the app is in. Rate limits
(429 + Retry-After) are retried by the `ApiClient` and surface as `RateLimited` once the
wait budget is used up.
"""

import re
from collections.abc import Mapping
from typing import Any

import httpx2
from mcp.server.mcpserver.exceptions import ToolError

from bench_core.auth import Credential, CredentialAuth
from bench_core.clock import Clock
from bench_core.http import (
    ApiClient,
    SlackRetryPolicy,
    cursor,
    fetch_page,
)
from bench_core.permissions import NotFoundOrNoAccess, PermissionDenied

from bench_slack.models import Channel, Message, Page

PROVIDER = "Slack"
BASE_URL = "https://slack.com/api"
USER_AGENT = "integration-bench"
MAX_PAGE_SIZE = 100
PAGE_SIZE_PARAM = "limit"
PAGINATION = cursor()


def make_api(
    *,
    credential: Credential | None = None,
    base_url: str = BASE_URL,
    transport: httpx2.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
    default_wait_budget: float = 30.0,
) -> ApiClient:
    return ApiClient(
        base_url,
        policy=SlackRetryPolicy(name=PROVIDER),
        clock=clock,
        transport=transport,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        auth=CredentialAuth(credential) if credential else None,
        default_wait_budget=default_wait_budget,
    )


# --- Argument validation ------------------------------------------------------------------

_CHANNEL_ID = re.compile(r"^[CG][A-Z0-9]{6,20}$")
_TS = re.compile(r"^\d{9,10}\.\d{6}$")
MAX_TEXT = 4000
"""Slack truncates longer messages; the tools refuse them instead."""
HIDDEN_SUBTYPES = frozenset({"channel_join", "channel_leave"})
"""Membership notices: noise for the model, so history and threads leave them out."""


def channel_id(value: str) -> str:
    if not _CHANNEL_ID.match(value):
        raise ToolError(
            f"Invalid channel id {value!r}: use the id from slack_list_channels "
            "(e.g. 'C0123456789'), not the channel name."
        )
    return value


def message_ts(value: str, name: str = "thread_ts") -> str:
    if not _TS.match(value):
        raise ToolError(
            f"Invalid {name} {value!r}: use the ts of a message from a history result, "
            "e.g. '1791655876.394709'."
        )
    return value


def message_text(text: str) -> str:
    if not text.strip():
        raise ToolError("The message text must not be empty.")
    if len(text) > MAX_TEXT:
        raise ToolError(f"The message text is {len(text)} characters; the limit is {MAX_TEXT}.")
    return text


def _positive(name: str, value: int) -> int:
    if value < 1:
        raise ToolError(f"{name} must be at least 1, got {value}.")
    return value


# --- Error translation --------------------------------------------------------------------


def _body(response: httpx2.Response) -> Mapping[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, Mapping) else {}


def error_text(response: httpx2.Response) -> str:
    """`{"ok": false, "error": "code", "response_metadata": {"messages": ["..."]}}`."""
    body = _body(response)
    code = str(body.get("error") or response.status_code)
    metadata = body.get("response_metadata")
    messages = metadata.get("messages") if isinstance(metadata, Mapping) else None
    return f"{code} ({'; '.join(map(str, messages))})" if messages else code


_AUTH_ERRORS = {"not_authed", "invalid_auth", "token_expired", "token_revoked", "account_inactive"}
_ARGUMENT_ERRORS = {
    "invalid_arguments",
    "invalid_cursor",
    "invalid_limit",
    "invalid_ts_latest",
    "invalid_ts_oldest",
    "msg_too_long",
    "no_text",
    "too_long",
}


def raise_for_status(response: httpx2.Response, *, action: str, what: str) -> None:
    """Raise a model-facing `ToolError` for HTTP errors and for `ok: false` bodies."""
    body = _body(response)
    if response.is_success and body.get("ok") is not False:
        return
    status = response.status_code
    if status >= 500:
        raise ToolError(
            f"{PROVIDER} had a server error ({status}) while trying to {action}. Try again later."
        )
    code = str(body.get("error") or status)
    if code in _AUTH_ERRORS or status == 401:
        raise ToolError(
            f"{PROVIDER} rejected the credentials while trying to {action} ({code}). "
            "Do not retry; the user has to sign in again."
        )
    if code == "missing_scope":
        needed = str(body.get("needed") or "")
        raise PermissionDenied(PROVIDER, action=action, missing=[s for s in needed.split(",") if s])
    if code == "not_in_channel":
        raise ToolError(
            f"Cannot {action}: the app is not a member of this channel. Pick a channel with "
            "is_member=true from slack_list_channels, or ask the user to invite the app."
        )
    if code == "is_archived":
        raise ToolError(f"Cannot {action}: the channel is archived.")
    if code.endswith("_not_found"):
        raise NotFoundOrNoAccess(PROVIDER, what)
    if code in _ARGUMENT_ERRORS:
        raise ToolError(
            f"{PROVIDER} rejected the request to {action}: {error_text(response)}. "
            "Fix the arguments and call again."
        )
    raise ToolError(f"{PROVIDER} could not {action}: {error_text(response)}.")


# --- Client -------------------------------------------------------------------------------

JSON_UTF8 = {"Content-Type": "application/json; charset=utf-8"}
"""Without the charset Slack answers with a `missing_charset` warning."""


class SlackClient:
    def __init__(self, api: ApiClient) -> None:
        self.api = api

    async def _page(
        self,
        method: str,
        key: str,
        *,
        cursor: str | None,
        page_size: int,
        params: dict[str, Any],
        action: str,
        what: str,
    ) -> tuple[list[Any], str | None]:
        params = {PAGE_SIZE_PARAM: min(_positive("page_size", page_size), MAX_PAGE_SIZE), **params}
        page = await fetch_page(self.api, method, PAGINATION, cursor=cursor, params=params)
        raise_for_status(page.response, action=action, what=what)
        return list(_body(page.response).get(key) or []), page.next_cursor

    async def list_channels(
        self, *, cursor: str | None = None, page_size: int = 20
    ) -> Page[Channel]:
        """Public, not archived channels of the workspace, one page at a time."""
        data, next_cursor = await self._page(
            "/conversations.list",
            "channels",
            cursor=cursor,
            page_size=page_size,
            params={"types": "public_channel", "exclude_archived": "true"},
            action="list channels",
            what="The channel list",
        )
        return Page[Channel](
            items=[Channel.model_validate(c) for c in data], next_cursor=next_cursor
        )

    async def read_history(
        self, channel: str, *, cursor: str | None = None, page_size: int = 20
    ) -> Page[Message]:
        """Top-level messages of a channel, newest first (thread replies not included)."""
        data, next_cursor = await self._page(
            "/conversations.history",
            "messages",
            cursor=cursor,
            page_size=page_size,
            params={"channel": channel_id(channel)},
            action="read the channel history",
            what=f"Channel {channel}",
        )
        return Page[Message](items=_visible(data), next_cursor=next_cursor)

    async def read_thread(
        self, channel: str, thread_ts: str, *, cursor: str | None = None, page_size: int = 20
    ) -> Page[Message]:
        """The parent message and its replies, oldest first."""
        data, next_cursor = await self._page(
            "/conversations.replies",
            "messages",
            cursor=cursor,
            page_size=page_size,
            params={"channel": channel_id(channel), "ts": message_ts(thread_ts)},
            action="read the thread",
            what=f"Thread {thread_ts} in channel {channel}",
        )
        return Page[Message](items=_visible(data), next_cursor=next_cursor)

    async def post_message(self, channel: str, text: str) -> Message:
        return await self._post(
            {"channel": channel_id(channel), "text": message_text(text)},
            action="post the message",
            what=f"Channel {channel}",
        )

    async def reply_in_thread(self, channel: str, thread_ts: str, text: str) -> Message:
        """Reply to a top-level message. Slack itself would post a reply to an unknown
        `thread_ts` as a new top-level message, so the parent is checked first."""
        payload = {
            "channel": channel_id(channel),
            "thread_ts": message_ts(thread_ts),
            "text": message_text(text),
        }
        thread = await self.read_thread(channel, thread_ts, page_size=1)
        parent = next((m for m in thread.items if m.ts == thread_ts), None)
        if parent is None:
            raise NotFoundOrNoAccess(PROVIDER, f"Thread {thread_ts} in channel {channel}")
        if parent.thread_ts not in (None, parent.ts):
            raise ToolError(
                f"Message {thread_ts} is itself a reply in thread {parent.thread_ts}; "
                "pass that thread_ts to reply in the same thread."
            )
        return await self._post(payload, action="reply in the thread", what=f"Channel {channel}")

    async def delete_message(self, channel: str, ts: str) -> None:
        response = await self.api.post(
            "/chat.delete",
            json={"channel": channel_id(channel), "ts": message_ts(ts, "ts")},
            headers=JSON_UTF8,
        )
        raise_for_status(response, action="delete the message", what=f"Message {ts}")

    async def auth_test(self) -> Mapping[str, Any]:
        """Who the token belongs to (`user`, `team`, `bot_id`); `bench doctor` calls it."""
        response = await self.api.post("/auth.test")
        raise_for_status(response, action="check the token", what="The token")
        return _body(response)

    async def _post(self, payload: dict[str, Any], *, action: str, what: str) -> Message:
        response = await self.api.post("/chat.postMessage", json=payload, headers=JSON_UTF8)
        raise_for_status(response, action=action, what=what)
        return Message.model_validate(_body(response).get("message"))


def _visible(data: list[Any]) -> list[Message]:
    messages = [Message.model_validate(m) for m in data]
    return [m for m in messages if m.subtype not in HIDDEN_SUBTYPES]
