"""In-memory Todoist API for development and unit tests.

Serves the template's example resource ("items") in the API style chosen at generation
time: cursor pagination and message-style errors. Replace it with
the real endpoints as the client grows, and keep responses in the provider's raw JSON
shape, so the client's models parse exactly what they would parse from the real API.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import httpx2

from bench_core.auth import StaticToken
from bench_core.clock import Clock
from bench_core.testing import FakeBackend, FakeClock, FakeRequest

from bench_todoist.client import BASE_URL, PAGE_SIZE_PARAM
from bench_todoist.scopes import READ, WRITE

BASE_PATH = urlsplit(BASE_URL).path
MAX_TITLE = 200
TOKEN_PARAM = "cursor"


def timestamp(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def credential(token: str) -> StaticToken:
    """The static credential that presents `token` (see `TodoistFake.issue_token`)."""
    return StaticToken("todoist", token)


@dataclass
class TodoistFake(FakeBackend):
    clock: Clock = field(default_factory=FakeClock)
    items: dict[str, dict[str, Any]] = field(default_factory=dict)
    _next_id: int = field(default=1000, init=False, repr=False)

    def __post_init__(self) -> None:
        self.route("GET", BASE_PATH + "/items", scopes=[READ])(self._list_items)
        self.route("GET", BASE_PATH + "/items/{item_id}", scopes=[READ])(self._get_item)
        self.route("POST", BASE_PATH + "/items", scopes=[WRITE])(self._create_item)

    # --- setup ----------------------------------------------------------------------------

    def now(self) -> str:
        return timestamp(self.clock.time())

    def add_item(
        self,
        title: str,
        *,
        description: str | None = None,
        status: str = "open",
        created_at: str | None = None,
    ) -> dict[str, Any]:
        self._next_id += 1
        item_id = str(self._next_id)
        created = created_at or self.now()
        item = {
            "id": item_id,
            "title": title,
            "status": status,
            "description": description,
            "created_at": created,
            "updated_at": created,
            "url": f"{BASE_URL}/items/{item_id}",
        }
        self.items[item_id] = item
        return item

    # --- provider error format ------------------------------------------------------------

    def _error(self, status: int, message: str, code: str, **extra: Any) -> httpx2.Response:
        return httpx2.Response(status, json={"message": message, **extra})

    def _validation(self, message: str, code: str = "invalid_arguments") -> httpx2.Response:
        return self._error(422, message, code)

    def unauthorized_response(self, request: httpx2.Request) -> httpx2.Response:
        return self._error(401, "The access token is invalid or expired.", "invalid_auth")

    def forbidden_response(
        self, request: httpx2.Request, missing: Sequence[str]
    ) -> httpx2.Response:
        return self._error(403, "The token lacks the scope for this call.", "forbidden")

    def not_found_response(self, request: httpx2.Request) -> httpx2.Response:
        return self._error(404, "Not found.", "not_found")

    def server_error_response(self, request: httpx2.Request, status: int) -> httpx2.Response:
        return self._error(status, "Internal error.", "internal_error")

    def rate_limit_response(
        self, request: httpx2.Request, retry_after: float | None
    ) -> httpx2.Response:
        response = self._error(429, "Rate limit exceeded.", "ratelimited")
        if retry_after is not None:
            response.headers["Retry-After"] = f"{retry_after:g}"
        return response

    def _ok(
        self, body: Any, *, status: int = 200, headers: dict[str, str] | None = None
    ) -> httpx2.Response:
        return httpx2.Response(status, headers=headers, json=body)

    # --- routes ---------------------------------------------------------------------------

    def _list_items(self, r: FakeRequest) -> httpx2.Response:
        """Items whose title contains `q`, oldest first, one page at a time."""
        query = r.query.get("q", "").lower()
        items = [i for i in self.items.values() if query in i["title"].lower()]
        try:
            size = min(max(int(r.query.get(PAGE_SIZE_PARAM, "20")), 1), 100)
            token = r.query.get(TOKEN_PARAM)
            offset = int(token.removeprefix("o")) if token else 0
        except ValueError:
            return self._validation("Invalid page size or page token.", "invalid_cursor")
        chunk = items[offset : offset + size]
        more = offset + size < len(items)
        next_cursor = f"o{offset + size}" if more else ""
        return self._ok({"items": chunk, "response_metadata": {"next_cursor": next_cursor}})

    def _get_item(self, r: FakeRequest) -> httpx2.Response:
        item = self.items.get(r.params["item_id"])
        if item is None:
            return self._error(404, f"Item {r.params['item_id']} does not exist.", "item_not_found")
        return self._ok(item)

    def _create_item(self, r: FakeRequest) -> httpx2.Response:
        body = r.json() or {}
        title = body.get("title")
        if not isinstance(title, str) or not title.strip():
            return self._validation("title is required.")
        if len(title) > MAX_TITLE:
            return self._validation(f"title is too long (maximum is {MAX_TITLE} characters).")
        item = self.add_item(title, description=body.get("description"))
        return self._ok(item, status=201)
