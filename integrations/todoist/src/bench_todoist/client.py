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
    cursor,
    fetch_page,
)
from bench_core.permissions import NotFoundOrNoAccess, PermissionDenied

from bench_todoist.models import Item, Page

PROVIDER = "Todoist"
BASE_URL = "https://api.todoist.com/api/v1"
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

_ITEM_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def item_path(item_id: str) -> str:
    if not _ITEM_ID.match(item_id):
        raise ToolError(f"Invalid item id {item_id!r}: use the id from a list result, e.g. '1001'.")
    return f"/items/{item_id}"


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
    """`{"message": "...", "errors": [{"field", "code", "message"} | "..."]}`."""
    body = _body(response)
    parts: list[str] = []
    for error in body.get("errors") or []:
        if isinstance(error, str):
            parts.append(error)
        elif isinstance(error, Mapping):
            text = error.get("message") or " ".join(
                str(error[k]) for k in ("field", "code") if error.get(k)
            )
            if text:
                parts.append(str(text))
    message = body.get("message")
    if not parts and isinstance(message, str) and message:
        parts.append(message)
    return "; ".join(parts) or response.reason_phrase


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

    async def list_items(
        self, *, query: str | None = None, cursor: str | None = None, page_size: int = 20
    ) -> Page[Item]:
        """Items whose title contains `query`, one page at a time."""
        params: dict[str, Any] = {
            PAGE_SIZE_PARAM: min(_positive("page_size", page_size), MAX_PAGE_SIZE)
        }
        if query:
            params["q"] = query
        page = await fetch_page(self.api, "/items", PAGINATION, cursor=cursor, params=params)
        raise_for_status(page.response, action="list items", what="The item list")
        data = _body(page.response).get("items") or []
        return Page[Item](
            items=[Item.model_validate(item) for item in data], next_cursor=page.next_cursor
        )

    async def get_item(self, item_id: str) -> Item:
        response = await self.api.get(item_path(item_id))
        raise_for_status(response, action="read the item", what=f"Item {item_id}")
        return Item.model_validate(response.json())

    async def create_item(self, title: str, *, description: str | None = None) -> Item:
        if not title.strip():
            raise ToolError("The item title must not be empty.")
        payload: dict[str, Any] = {"title": title}
        if description:
            payload["description"] = description
        response = await self.api.post("/items", json=payload)
        raise_for_status(response, action="create an item", what="The item list")
        return Item.model_validate(response.json())
