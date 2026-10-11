"""Pagination helpers for the four styles our providers use.

Tools return the model one page plus an opaque `next_cursor`. The cursor wraps the
provider's raw continuation value with the paginator kind, so a cursor from one tool
cannot be fed into another, and a garbled cursor becomes an actionable `ToolError`.

Styles:
- `link_header`     GitHub: `Link: <...>; rel="next"`
- `next_page_token` Jira `/search/jql`: body `nextPageToken` (+ `isLast`), sent back as
                    `nextPageToken`; guards against the API repeating the same token
- `page_token`      Drive: body `nextPageToken`, sent back as `pageToken`
- `cursor`          Slack: body `response_metadata.next_cursor` ("" = end), sent as `cursor`
"""

import base64
import binascii
import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx2
from mcp.server.mcpserver.exceptions import ToolError

from bench_core.http.client import ApiClient

Params = dict[str, Any]


class Paginator(Protocol):
    @property
    def kind(self) -> str: ...

    def prepare(
        self, url: str, params: Params, raw_cursor: str | None, *, base: httpx2.URL
    ) -> tuple[str, Params]:
        """URL and query params for the request that continues at `raw_cursor`.

        `base` is the API client's base URL; relative URLs are resolved against it.
        """
        ...

    def next_raw(self, response: httpx2.Response, sent: str | None) -> str | None:
        """Raw continuation value from a successful response, or None on the last page."""
        ...


@dataclass(frozen=True)
class LinkHeaderPaginator:
    kind: str = "link_header"

    def prepare(
        self, url: str, params: Params, raw_cursor: str | None, *, base: httpx2.URL
    ) -> tuple[str, Params]:
        if not raw_cursor:
            return url, params
        # The next URL already carries every query param. Its path is absolute on the
        # base URL's host, so a base path like `/rest/api/3` must not be prepended again.
        return str(base.copy_with(raw_path=raw_cursor.encode("ascii"))), {}

    def next_raw(self, response: httpx2.Response, sent: str | None) -> str | None:
        next_url = response.links.get("next", {}).get("url")
        if not next_url:
            return None
        # Keep path + query only: the request then always goes to the client's base URL,
        # so a cursor can never redirect credentials to another host.
        raw = httpx2.URL(next_url).raw_path.decode("ascii")
        return None if raw == sent else raw


@dataclass(frozen=True)
class BodyTokenPaginator:
    """Continuation token in the JSON body, sent back as a query param."""

    kind: str
    token_path: tuple[str, ...]
    param: str
    is_last_field: str | None = None

    def prepare(
        self, url: str, params: Params, raw_cursor: str | None, *, base: httpx2.URL
    ) -> tuple[str, Params]:
        if raw_cursor is None:
            return url, params
        return url, {**params, self.param: raw_cursor}

    def next_raw(self, response: httpx2.Response, sent: str | None) -> str | None:
        body = response.json()
        if self.is_last_field and body.get(self.is_last_field) is True:
            return None
        value: Any = body
        for key in self.token_path:
            value = value.get(key) if isinstance(value, Mapping) else None
        if not isinstance(value, str) or not value:
            return None
        # Jira's /search/jql has been seen to hand back the token it was given; stop
        # instead of looping forever.
        return None if value == sent else value


def link_header() -> LinkHeaderPaginator:
    return LinkHeaderPaginator()


def next_page_token() -> BodyTokenPaginator:
    return BodyTokenPaginator(
        kind="next_page_token",
        token_path=("nextPageToken",),
        param="nextPageToken",
        is_last_field="isLast",
    )


def page_token() -> BodyTokenPaginator:
    return BodyTokenPaginator(kind="page_token", token_path=("nextPageToken",), param="pageToken")


def cursor() -> BodyTokenPaginator:
    return BodyTokenPaginator(
        kind="cursor", token_path=("response_metadata", "next_cursor"), param="cursor"
    )


def next_cursor() -> BodyTokenPaginator:
    """`{"results": [...], "next_cursor": "..." | null}`, sent back as `cursor` (Todoist)."""
    return BodyTokenPaginator(kind="next_cursor", token_path=("next_cursor",), param="cursor")


# --- Opaque cursors for the model ---------------------------------------------------------


def encode_cursor(kind: str, raw: str) -> str:
    payload = json.dumps({"k": kind, "v": raw}, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_cursor(kind: str, cursor: str) -> str:
    invalid = ToolError(
        "Invalid cursor. Pass the exact `next_cursor` value from the previous result of "
        "this same tool, or omit it to start from the first page."
    )
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except (ValueError, UnicodeError, binascii.Error):
        raise invalid from None
    if not isinstance(data, dict) or data.get("k") != kind or not isinstance(data.get("v"), str):
        raise invalid
    return data["v"]


# --- Fetching -----------------------------------------------------------------------------


async def _get(api: ApiClient, url: str, params: Params) -> httpx2.Response:
    # httpx2 replaces a URL's query string with `params` even when `params` is empty,
    # which would drop the query of a Link-header next URL.
    return await api.get(url, params=params) if params else await api.get(url)


@dataclass(frozen=True)
class PageResponse:
    response: httpx2.Response
    next_cursor: str | None
    """Opaque cursor for the model, None on the last page or on an error response."""


async def fetch_page(
    api: ApiClient,
    url: str,
    paginator: Paginator,
    *,
    cursor: str | None = None,
    params: Params | None = None,
) -> PageResponse:
    """Fetch the page at the model-supplied `cursor` (or the first page)."""
    raw = decode_cursor(paginator.kind, cursor) if cursor else None
    req_url, req_params = paginator.prepare(url, dict(params or {}), raw, base=api.base_url)
    response = await _get(api, req_url, req_params)
    if not response.is_success:
        return PageResponse(response, None)
    next_raw = paginator.next_raw(response, raw)
    return PageResponse(response, encode_cursor(paginator.kind, next_raw) if next_raw else None)


class PaginationLoop(RuntimeError):
    """The API returned a continuation value it had already returned before."""


async def iter_pages(
    api: ApiClient,
    url: str,
    paginator: Paginator,
    *,
    params: Params | None = None,
    max_pages: int = 50,
) -> AsyncIterator[httpx2.Response]:
    """Walk all pages (sandbox seeding/teardown, not tools). Stops at the first error."""
    raw: str | None = None
    seen: set[str] = set()
    for _ in range(max_pages):
        req_url, req_params = paginator.prepare(url, dict(params or {}), raw, base=api.base_url)
        response = await _get(api, req_url, req_params)
        yield response
        if not response.is_success:
            return
        raw = paginator.next_raw(response, raw)
        if raw is None:
            return
        if raw in seen:
            raise PaginationLoop(f"{paginator.kind}: continuation {raw!r} repeated")
        seen.add(raw)
