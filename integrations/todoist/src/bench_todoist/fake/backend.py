"""In-memory Todoist API for development and unit tests.

Serves `/api/v1/tasks` (list, filter by `search:`, get, create) in Todoist's raw JSON
shape: lists as `{"results": [...], "next_cursor": ...}`, errors as HTTP status plus
`{"error", "error_code", "error_tag", "error_extra", "http_code"}`, so the client's models
parse exactly what they would parse from the real API. Like the live API, the lists only
hold open tasks.
"""

import re
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
TOKEN_PARAM = "cursor"
MAX_LIMIT = 200
_ID = re.compile(r"^[0-9A-Za-z]{16}$")
_CURSOR = re.compile(r"^o(\d+)\.fake$")


def timestamp(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%S.000000Z")


def credential(token: str) -> StaticToken:
    """The static credential that presents `token` (see `TodoistFake.issue_token`)."""
    return StaticToken("todoist", token)


@dataclass
class TodoistFake(FakeBackend):
    clock: Clock = field(default_factory=FakeClock)
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    _next_id: int = field(default=1000, init=False, repr=False)

    def __post_init__(self) -> None:
        self.route("GET", BASE_PATH + "/tasks", scopes=[READ])(self._list_tasks)
        self.route("GET", BASE_PATH + "/tasks/filter", scopes=[READ])(self._filter_tasks)
        self.route("GET", BASE_PATH + "/tasks/{task_id}", scopes=[READ])(self._get_task)
        self.route("POST", BASE_PATH + "/tasks", scopes=[WRITE])(self._create_task)

    # --- setup ----------------------------------------------------------------------------

    def now(self) -> str:
        return timestamp(self.clock.time())

    def add_task(
        self,
        title: str,
        *,
        description: str | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        self._next_id += 1
        task_id = f"6Fake{self._next_id:011d}"
        created = created_at or self.now()
        task = {
            "id": task_id,
            "project_id": "6FakeInbox000000",
            "content": title,
            "description": description or "",
            "checked": False,
            "is_deleted": False,
            "labels": [],
            "priority": 1,
            "due": None,
            "added_at": created,
            "updated_at": created,
            "completed_at": None,
        }
        self.tasks[task_id] = task
        return task

    # --- provider error format ------------------------------------------------------------

    def _error(
        self, status: int, message: str, tag: str, code: int = 0, **extra: Any
    ) -> httpx2.Response:
        body = {
            "error": message,
            "error_code": code,
            "error_extra": {"retry_after": 3, **extra},
            "error_tag": tag,
            "http_code": status,
        }
        return httpx2.Response(status, json=body)

    def _invalid(self, argument: str, expected: str) -> httpx2.Response:
        return self._error(
            400,
            "Invalid argument value",
            "INVALID_ARGUMENT_VALUE",
            20,
            argument=argument,
            expected=expected,
        )

    def unauthorized_response(self, request: httpx2.Request) -> httpx2.Response:
        return self._error(401, "Unauthorized", "UNAUTHORIZED", 477)

    def forbidden_response(
        self, request: httpx2.Request, missing: Sequence[str]
    ) -> httpx2.Response:
        return self._error(403, "Insufficient token scope", "AUTH_INSUFFICIENT_TOKEN_SCOPE", 403)

    def not_found_response(self, request: httpx2.Request) -> httpx2.Response:
        return self._error(404, "Not found", "NOT_FOUND", 478)

    def server_error_response(self, request: httpx2.Request, status: int) -> httpx2.Response:
        return self._error(status, "Service unavailable", "SERVICE_UNAVAILABLE", 0)

    def rate_limit_response(
        self, request: httpx2.Request, retry_after: float | None
    ) -> httpx2.Response:
        response = self._error(429, "Too many requests", "LIMITS_REACHED", 429)
        if retry_after is not None:
            response.headers["Retry-After"] = f"{retry_after:g}"
        return response

    def _ok(self, body: Any, *, status: int = 200) -> httpx2.Response:
        return httpx2.Response(status, json=body)

    # --- routes ---------------------------------------------------------------------------

    def _page(self, r: FakeRequest, tasks: list[dict[str, Any]]) -> httpx2.Response:
        """One page, oldest first, with an opaque `next_cursor` (null on the last page)."""
        try:
            size = int(r.query.get(PAGE_SIZE_PARAM, "50"))
        except ValueError:
            return self._invalid("limit", "Input should be a valid integer")
        if not 1 <= size <= MAX_LIMIT:
            return self._invalid("limit", f"Input should be less than or equal to {MAX_LIMIT}")
        offset = 0
        if token := r.query.get(TOKEN_PARAM):
            match = _CURSOR.match(token)
            if match is None:
                return self._invalid("cursor", "String should match the cursor pattern")
            offset = int(match[1])
        chunk = tasks[offset : offset + size]
        more = offset + size < len(tasks)
        return self._ok(
            {"results": chunk, "next_cursor": f"o{offset + size}.fake" if more else None}
        )

    def _open(self) -> list[dict[str, Any]]:
        return [t for t in self.tasks.values() if not t["checked"]]

    def _list_tasks(self, r: FakeRequest) -> httpx2.Response:
        return self._page(r, self._open())

    def _filter_tasks(self, r: FakeRequest) -> httpx2.Response:
        """Only the `search: <text>` filter (case-insensitive substring of the content)."""
        query = r.query.get("query", "")
        if not query.startswith("search: "):
            return self._error(400, "The search query is incorrect", "INVALID_SEARCH_QUERY", 55)
        text = query.removeprefix("search: ").lower()
        return self._page(r, [t for t in self._open() if text in t["content"].lower()])

    def _get_task(self, r: FakeRequest) -> httpx2.Response:
        task_id = r.params["task_id"]
        if not _ID.match(task_id):
            return self._invalid("task_id", "Value error, Non-base32 digit found")
        task = self.tasks.get(task_id)
        if task is None:
            return self._error(404, "Task not found", "NOT_FOUND", 478)
        return self._ok(task)

    def _create_task(self, r: FakeRequest) -> httpx2.Response:
        body = r.json() or {}
        content = body.get("content")
        if content is None:
            return self._error(
                400, "Required argument is missing", "ARGUMENT_MISSING", 19, argument="content"
            )
        if not isinstance(content, str) or not content.strip():
            return self._error(400, "Item content is empty", "ITEM_CONTENT_EMPTY", 61)
        task = self.add_task(content, description=body.get("description"))
        return self._ok(task)
