"""In-memory Slack Web API for development and unit tests.

Serves the five methods the client uses plus `chat.delete` (scenario teardown),
`auth.test` and the token endpoint, in Slack's raw JSON shape, so the client's models
parse exactly what they would parse from the real API. Behaviour mirrors what the live
API did in step 29's probe: errors are HTTP 200 with `{"ok": false, "error": code}`,
reading or posting in a channel the app has not joined gives `not_in_channel`, a
`thread_ts` that matches no message still posts (as a top-level message), and
`conversations.replies` on a reply returns just that reply.

Every request acts as the one fake bot user; channels flagged `is_member` are the ones
it has joined.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx2

from bench_core.auth import StaticToken
from bench_core.clock import Clock
from bench_core.testing import FakeBackend, FakeClock, FakeRequest

from bench_slack.auth import TOKEN_URL
from bench_slack.client import BASE_URL, PAGE_SIZE_PARAM
from bench_slack.scopes import CHANNELS_HISTORY, CHANNELS_READ, CHAT_WRITE

BASE_PATH = urlsplit(BASE_URL).path
TOKEN_PATH = urlsplit(TOKEN_URL).path
TEAM = "T0FAKETEAM"
BOT_USER = "U0FAKEBOT"
BOT_ID = "B0FAKEBOT"
MAX_TEXT = 40_000
FIRST_TS = 1_791_600_000
"""Message timestamps count up from here (2026-10-09), one second per message."""


def credential(token: str) -> StaticToken:
    """The static credential that presents `token` (see `SlackFake.issue_token`)."""
    return StaticToken("slack", token)


def _oauth_error(error: str) -> httpx2.Response:
    return httpx2.Response(200, json={"ok": False, "error": error})


@dataclass
class SlackFake(FakeBackend):
    clock: Clock = field(default_factory=FakeClock)
    channels: dict[str, dict[str, Any]] = field(default_factory=dict)
    messages: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    """Channel id -> all its messages (top-level and replies), oldest first."""
    oauth_clients: set[str] = field(default_factory=set)
    """Client ids of PKCE apps that may refresh user tokens (no client secret)."""
    refresh_tokens: dict[str, frozenset[str]] = field(default_factory=dict)
    """Valid (unused) refresh tokens -> scopes of the access tokens they mint."""
    _minted: int = field(default=0, init=False, repr=False)
    _posted: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        api = BASE_PATH + "/"
        self.route("POST", TOKEN_PATH, public=True)(self._token)
        self.route("POST", api + "auth.test")(self._auth_test)
        self.route("GET", api + "conversations.list", scopes=[CHANNELS_READ])(self._list)
        self.route("GET", api + "conversations.history", scopes=[CHANNELS_HISTORY])(self._history)
        self.route("GET", api + "conversations.replies", scopes=[CHANNELS_HISTORY])(self._replies)
        self.route("POST", api + "chat.postMessage", scopes=[CHAT_WRITE])(self._post)
        self.route("POST", api + "chat.delete", scopes=[CHAT_WRITE])(self._delete)

    # --- setup ----------------------------------------------------------------------------

    def _mint(self, prefix: str) -> str:
        self._minted += 1
        return f"fake-{prefix}-{self._minted}"

    def user_tokens(self, scopes: Sequence[str] = ()) -> tuple[str, str]:
        """A fresh access + refresh token pair, as after `bench auth login slack`."""
        access = self.issue_token(self._mint("access"), scopes)
        refresh = self._mint("refresh")
        self.refresh_tokens[refresh] = frozenset(scopes)
        return access, refresh

    def add_channel(
        self, name: str, *, is_member: bool = True, is_general: bool = False, purpose: str = ""
    ) -> dict[str, Any]:
        channel_id = f"C0FAKE{len(self.channels) + 1:05d}"
        channel = {
            "id": channel_id,
            "name": name,
            "is_channel": True,
            "is_private": False,
            "is_archived": False,
            "is_general": is_general,
            "is_member": is_member,
            "num_members": 2 if is_member else 1,
            "topic": {"value": "", "creator": "", "last_set": 0},
            "purpose": {"value": purpose, "creator": "", "last_set": 0},
        }
        self.channels[channel_id] = channel
        self.messages[channel_id] = []
        return channel

    def add_message(
        self, channel_id: str, text: str, *, thread_ts: str | None = None
    ) -> dict[str, Any]:
        """Post as the bot; a `thread_ts` that matches no message posts top-level."""
        self._posted += 1
        ts = f"{FIRST_TS + self._posted}.{self._posted:06d}"
        message: dict[str, Any] = {
            "user": BOT_USER,
            "type": "message",
            "ts": ts,
            "bot_id": BOT_ID,
            "text": text,
            "team": TEAM,
        }
        parent = self._find(channel_id, thread_ts) if thread_ts else None
        if parent is not None and parent.get("thread_ts", parent["ts"]) == parent["ts"]:
            parent["thread_ts"] = parent["ts"]
            parent["reply_count"] = parent.get("reply_count", 0) + 1
            parent["latest_reply"] = ts
            message["thread_ts"] = parent["ts"]
            message["parent_user_id"] = parent["user"]
        self.messages[channel_id].append(message)
        return message

    def _find(self, channel_id: str, ts: str | None) -> dict[str, Any] | None:
        return next((m for m in self.messages[channel_id] if m["ts"] == ts), None)

    # --- provider error format ------------------------------------------------------------

    def _error(self, code: str, *, status: int = 200, **extra: Any) -> httpx2.Response:
        # Slack answers errors with HTTP 200 and `ok: false`; only rate limits and server
        # errors use real status codes.
        return httpx2.Response(status, json={"ok": False, "error": code, **extra})

    def unauthorized_response(self, request: httpx2.Request) -> httpx2.Response:
        return self._error("invalid_auth")

    def forbidden_response(
        self, request: httpx2.Request, missing: Sequence[str]
    ) -> httpx2.Response:
        return self._error("missing_scope", needed=",".join(missing), provided="")

    def not_found_response(self, request: httpx2.Request) -> httpx2.Response:
        return self._error("unknown_method", status=404)

    def server_error_response(self, request: httpx2.Request, status: int) -> httpx2.Response:
        return self._error("internal_error", status=status)

    def rate_limit_response(
        self, request: httpx2.Request, retry_after: float | None
    ) -> httpx2.Response:
        response = self._error("ratelimited", status=429)
        if retry_after is not None:
            response.headers["Retry-After"] = f"{retry_after:g}"
        return response

    def _ok(self, body: dict[str, Any]) -> httpx2.Response:
        return httpx2.Response(200, json={"ok": True, **body})

    def _channel(self, channel_id: str) -> dict[str, Any] | httpx2.Response:
        """The joined channel, or the error Slack gives for it."""
        channel = self.channels.get(channel_id)
        if channel is None:
            return self._error("channel_not_found")
        if not channel["is_member"]:
            return self._error("not_in_channel")
        return channel

    def _paged(self, r: FakeRequest, key: str, items: list[Any]) -> httpx2.Response:
        try:
            size = min(max(int(r.query.get(PAGE_SIZE_PARAM, "100")), 1), 1000)
            token = r.query.get("cursor")
            offset = int(token.removeprefix("o")) if token else 0
        except ValueError:
            return self._error("invalid_cursor")
        more = offset + size < len(items)
        body: dict[str, Any] = {key: items[offset : offset + size]}
        if key == "messages":
            body["has_more"] = more
        body["response_metadata"] = {"next_cursor": f"o{offset + size}" if more else ""}
        return self._ok(body)

    # --- token endpoint -------------------------------------------------------------------

    def _token(self, r: FakeRequest) -> httpx2.Response:
        """refresh_token grant of a PKCE app; every refresh token works once (rotation)."""
        form = r.form()
        if form.get("client_id", "") not in self.oauth_clients:
            return _oauth_error("invalid_client_id")
        if form.get("grant_type") != "refresh_token":
            return _oauth_error("unsupported_grant_type")
        scopes = self.refresh_tokens.pop(form.get("refresh_token", ""), None)
        if scopes is None:
            return _oauth_error("invalid_refresh_token")
        access, refresh = self.user_tokens(sorted(scopes))
        return self._ok(
            {
                "access_token": access,
                "refresh_token": refresh,
                "expires_in": 43200,
                "token_type": "user",
                "scope": ",".join(sorted(scopes)),
            }
        )

    # --- methods --------------------------------------------------------------------------

    def _auth_test(self, r: FakeRequest) -> httpx2.Response:
        return self._ok(
            {
                "team": "integration-bench",
                "team_id": TEAM,
                "user": "bench-ci",
                "user_id": BOT_USER,
                "bot_id": BOT_ID,
            }
        )

    def _list(self, r: FakeRequest) -> httpx2.Response:
        channels = [c for c in self.channels.values() if not c["is_archived"]]
        return self._paged(r, "channels", channels)

    def _history(self, r: FakeRequest) -> httpx2.Response:
        channel = self._channel(r.query.get("channel", ""))
        if isinstance(channel, httpx2.Response):
            return channel
        top = [m for m in self.messages[channel["id"]] if m.get("thread_ts", m["ts"]) == m["ts"]]
        return self._paged(r, "messages", top[::-1])

    def _replies(self, r: FakeRequest) -> httpx2.Response:
        channel = self._channel(r.query.get("channel", ""))
        if isinstance(channel, httpx2.Response):
            return channel
        message = self._find(channel["id"], r.query.get("ts"))
        if message is None:
            return self._error("thread_not_found")
        if message.get("thread_ts", message["ts"]) != message["ts"]:
            return self._paged(r, "messages", [message])
        thread = [m for m in self.messages[channel["id"]] if m.get("thread_ts") == message["ts"]]
        return self._paged(r, "messages", thread or [message])

    def _post(self, r: FakeRequest) -> httpx2.Response:
        body = r.json() or {}
        channel = self._channel(str(body.get("channel", "")))
        if isinstance(channel, httpx2.Response):
            return channel
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            return self._error("no_text")
        if len(text) > MAX_TEXT:
            return self._error("msg_too_long")
        message = self.add_message(channel["id"], text, thread_ts=body.get("thread_ts"))
        return self._ok({"channel": channel["id"], "ts": message["ts"], "message": message})

    def _delete(self, r: FakeRequest) -> httpx2.Response:
        body = r.json() or {}
        channel = self._channel(str(body.get("channel", "")))
        if isinstance(channel, httpx2.Response):
            return channel
        message = self._find(channel["id"], body.get("ts"))
        if message is None:
            return self._error("message_not_found")
        self.messages[channel["id"]].remove(message)
        parent = self._find(channel["id"], message.get("thread_ts"))
        if parent is not None and parent is not message:
            replies = [
                m for m in self.messages[channel["id"]] if m.get("thread_ts") == parent["ts"]
            ]
            parent["reply_count"] = len(replies) - 1
            parent["latest_reply"] = replies[-1]["ts"]
        return self._ok({"channel": channel["id"], "ts": message["ts"]})
