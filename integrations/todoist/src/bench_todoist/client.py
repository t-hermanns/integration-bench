"""Typed client for the slice of the Todoist API the tools need.

Every method validates its identifiers before building a URL (no path injection from
model-supplied arguments), returns models from `bench_todoist.models`, and turns error
responses into `ToolError`s the model can act on: 401 -> sign in again, 403 ->
`PermissionDenied`, 404 -> `NotFoundOrNoAccess`, validation errors -> the provider's
messages. Rate limits are retried by the `ApiClient` and surface as `RateLimited` once
the wait budget is used up.
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
    fetch_page,
    next_cursor,
)
from bench_core.permissions import NotFoundOrNoAccess, PermissionDenied

from bench_todoist.models import Page, Task

PROVIDER = "Todoist"
BASE_URL = "https://api.todoist.com/api/v1"
USER_AGENT = "integration-bench"
MAX_PAGE_SIZE = 100
PAGE_SIZE_PARAM = "limit"
PAGINATION = next_cursor()


def make_api(
    *,
    credential: Credential | None = None,
    base_url: str = BASE_URL,
    transport: httpx2.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
) -> ApiClient:
    return ApiClient(
        base_url,
        policy=SlackRetryPolicy(name=PROVIDER),
        clock=clock,
        transport=transport,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        auth=CredentialAuth(credential) if credential else None,
    )


# --- Argument validation ------------------------------------------------------------------

_TASK_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def task_path(task_id: str) -> str:
    if not _TASK_ID.match(task_id):
        raise ToolError(f"Invalid task id {task_id!r}: use the id from a list result.")
    return f"/tasks/{task_id}"


# Todoist's filter language treats & | ! ( ) , and more as operators.
_SEARCH_TEXT = re.compile(r"^[\w .'-]{1,100}$")


def search_filter(query: str) -> str:
    if not _SEARCH_TEXT.match(query):
        raise ToolError(
            f"Invalid search text {query!r}: use words, spaces, dots, hyphens or apostrophes."
        )
    return f"search: {query}"


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
    """`{"error": "...", "error_tag": "...", "error_extra": {"argument", "expected"}}`."""
    body = _body(response)
    error = body.get("error")
    text = error if isinstance(error, str) and error else response.reason_phrase
    extra = body.get("error_extra")
    if isinstance(extra, Mapping) and extra.get("argument"):
        text += f" ({extra['argument']}"
        text += f": {extra['expected']})" if extra.get("expected") else ")"
    return text


def raise_for_status(response: httpx2.Response, *, action: str, what: str) -> None:
    """Raise a model-facing `ToolError` for any non-2xx response."""
    if response.is_success:
        return
    status = response.status_code
    if status == 401:
        raise ToolError(
            f"{PROVIDER} rejected the credentials while trying to {action} (401). "
            "Do not retry; the user has to sign in again."
        )
    if status == 403:
        # Read the missing scope from the response once the real API shows where it is
        # (a header, a body field); without it the model still learns what failed.
        raise PermissionDenied(PROVIDER, action=action)
    if status == 404:
        raise NotFoundOrNoAccess(PROVIDER, what)
    if status in (400, 422):
        raise ToolError(
            f"{PROVIDER} rejected the request to {action}: {error_text(response)}. "
            "Fix the arguments and call again."
        )
    if status == 409:
        raise ToolError(f"{PROVIDER} could not {action}: {error_text(response)}.")
    if status >= 500:
        raise ToolError(
            f"{PROVIDER} had a server error ({status}) while trying to {action}. Try again later."
        )
    raise ToolError(
        f"{PROVIDER} returned {status} while trying to {action}: {error_text(response)}."
    )


# --- Client -------------------------------------------------------------------------------


class TodoistClient:
    def __init__(self, api: ApiClient) -> None:
        self.api = api

    async def list_tasks(
        self, *, query: str | None = None, cursor: str | None = None, page_size: int = 20
    ) -> Page[Task]:
        """Tasks whose title contains `query`, one page at a time."""
        params: dict[str, Any] = {
            PAGE_SIZE_PARAM: min(_positive("page_size", page_size), MAX_PAGE_SIZE)
        }
        path = "/tasks"
        if query:
            path, params["query"] = "/tasks/filter", search_filter(query)
        page = await fetch_page(self.api, path, PAGINATION, cursor=cursor, params=params)
        raise_for_status(page.response, action="list tasks", what="The task list")
        data = _body(page.response).get("results") or []
        return Page[Task](
            items=[Task.model_validate(task) for task in data], next_cursor=page.next_cursor
        )

    async def get_task(self, task_id: str) -> Task:
        response = await self.api.get(task_path(task_id))
        raise_for_status(response, action="read the task", what=f"Task {task_id}")
        return Task.model_validate(response.json())

    async def create_task(self, title: str, *, description: str | None = None) -> Task:
        payload: dict[str, Any] = {"content": title}
        if description:
            payload["description"] = description
        response = await self.api.post("/tasks", json=payload)
        raise_for_status(response, action="create a task", what="The task list")
        return Task.model_validate(response.json())
