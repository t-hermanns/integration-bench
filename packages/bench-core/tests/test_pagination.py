from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from bench_core.http import (
    ApiClient,
    PaginationLoop,
    Paginator,
    RetryPolicy,
    cursor,
    decode_cursor,
    encode_cursor,
    fetch_page,
    iter_pages,
    link_header,
    next_cursor,
    next_page_token,
    page_token,
)
from bench_core.testing import FakeClock

ITEMS = [f"item{i}" for i in range(7)]
PAGE_SIZE = 3
Handler = Callable[[httpx2.Request], httpx2.Response]


def make_api(handler: Handler) -> tuple[ApiClient, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return handler(request)

    api = ApiClient(
        "https://api.example.invalid",
        policy=RetryPolicy(),
        clock=FakeClock(),
        transport=httpx2.MockTransport(record),
    )
    return api, seen


def _slice(offset: int) -> tuple[list[str], int | None]:
    nxt = offset + PAGE_SIZE
    return ITEMS[offset:nxt], (nxt if nxt < len(ITEMS) else None)


# --- Fake backends, one per style ---------------------------------------------------------


def github(request: httpx2.Request) -> httpx2.Response:
    page = int(request.url.params.get("page", "1"))
    items, nxt = _slice((page - 1) * PAGE_SIZE)
    headers = {}
    if nxt is not None:
        # Absolute URL on the API host, like GitHub; carries the original filter params.
        headers["Link"] = (
            f"<https://api.example.invalid/repos/o/r/issues?state=open&page={page + 1}>; "
            'rel="next", <https://api.example.invalid/repos/o/r/issues?page=3>; rel="last"'
        )
    return httpx2.Response(200, json=items, headers=headers)


def jira(request: httpx2.Request) -> httpx2.Response:
    offset = int(request.url.params.get("nextPageToken", "tok0")[3:])
    items, nxt = _slice(offset)
    body: dict[str, Any] = {"issues": items, "isLast": nxt is None}
    if nxt is not None:
        body["nextPageToken"] = f"tok{nxt}"
    return httpx2.Response(200, json=body)


def drive(request: httpx2.Request) -> httpx2.Response:
    offset = int(request.url.params.get("pageToken", "0"))
    items, nxt = _slice(offset)
    body: dict[str, Any] = {"files": items}
    if nxt is not None:
        body["nextPageToken"] = str(nxt)
    return httpx2.Response(200, json=body)


def slack(request: httpx2.Request) -> httpx2.Response:
    offset = int(request.url.params.get("cursor") or 0)
    items, nxt = _slice(offset)
    meta = {"next_cursor": str(nxt) if nxt is not None else ""}
    return httpx2.Response(200, json={"ok": True, "messages": items, "response_metadata": meta})


def todoist(request: httpx2.Request) -> httpx2.Response:
    offset = int(request.url.params.get("cursor") or 0)
    items, nxt = _slice(offset)
    return httpx2.Response(200, json={"results": items, "next_cursor": nxt and str(nxt)})


def items_of(response: httpx2.Response) -> list[str]:
    body = response.json()
    if isinstance(body, list):
        return body
    return next(v for k, v in body.items() if k in ("issues", "files", "messages", "results"))


STYLES: list[tuple[str, Callable[[], Paginator], Handler, str]] = [
    ("link_header", link_header, github, "/repos/o/r/issues"),
    ("next_page_token", next_page_token, jira, "/rest/api/3/search/jql"),
    ("page_token", page_token, drive, "/drive/v3/files"),
    ("cursor", cursor, slack, "/api/conversations.history"),
    ("next_cursor", next_cursor, todoist, "/api/v1/tasks"),
]
style_params = pytest.mark.parametrize(
    ("paginator", "handler", "url"),
    [pytest.param(p, h, u, id=name) for name, p, h, u in STYLES],
)


# --- Model-facing: one page + opaque next_cursor ------------------------------------------


@pytest.mark.anyio
@style_params
async def test_fetch_page_walks_all_items_via_next_cursor(
    paginator: Callable[[], Paginator], handler: Handler, url: str
) -> None:
    api, _ = make_api(handler)
    collected: list[str] = []
    next_cursor: str | None = None
    for _ in range(10):
        page = await fetch_page(api, url, paginator(), cursor=next_cursor, params={"state": "open"})
        collected += items_of(page.response)
        next_cursor = page.next_cursor
        if next_cursor is None:
            break
    assert collected == ITEMS


@pytest.mark.anyio
async def test_link_header_cursor_keeps_only_path_and_query() -> None:
    api, seen = make_api(github)
    first = await fetch_page(api, "/repos/o/r/issues", link_header(), params={"state": "open"})
    assert first.next_cursor is not None
    assert decode_cursor("link_header", first.next_cursor) == "/repos/o/r/issues?state=open&page=2"
    await fetch_page(api, "/repos/o/r/issues", link_header(), cursor=first.next_cursor)
    assert str(seen[1].url) == "https://api.example.invalid/repos/o/r/issues?state=open&page=2"


@pytest.mark.anyio
async def test_link_header_to_foreign_host_still_targets_base_url() -> None:
    def evil(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, json=[], headers={"Link": '<https://evil.invalid/x?a=1>; rel="next"'}
        )

    api, seen = make_api(evil)
    page = await fetch_page(api, "/start", link_header())
    assert page.next_cursor is not None
    await fetch_page(api, "/start", link_header(), cursor=page.next_cursor)
    assert seen[1].url.host == "api.example.invalid"


@pytest.mark.anyio
async def test_link_header_with_base_path_does_not_double_it() -> None:
    def prefixed(request: httpx2.Request) -> httpx2.Response:
        if request.url.params.get("page") == "2":
            return httpx2.Response(200, json=["b"])
        link = '<https://api.example.invalid/rest/api/3/items?page=2>; rel="next"'
        return httpx2.Response(200, json=["a"], headers={"Link": link})

    seen: list[httpx2.Request] = []
    api = ApiClient(
        "https://api.example.invalid/rest/api/3",
        policy=RetryPolicy(),
        clock=FakeClock(),
        transport=httpx2.MockTransport(lambda r: seen.append(r) or prefixed(r)),
    )
    first = await fetch_page(api, "/items", link_header())
    await fetch_page(api, "/items", link_header(), cursor=first.next_cursor)
    pages = [r async for r in iter_pages(api, "/items", link_header())]
    assert [r.json() for r in pages] == [["a"], ["b"]]
    paths = {r.url.path for r in seen}
    assert paths == {"/rest/api/3/items"}


@pytest.mark.anyio
async def test_repeated_token_ends_pagination() -> None:
    def stuck(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, json={"issues": ["a"], "nextPageToken": "same", "isLast": False}
        )

    api, _ = make_api(stuck)
    first = await fetch_page(api, "/search/jql", next_page_token())
    assert first.next_cursor is not None
    second = await fetch_page(api, "/search/jql", next_page_token(), cursor=first.next_cursor)
    assert second.next_cursor is None


@pytest.mark.anyio
async def test_is_last_wins_over_token() -> None:
    def last(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"issues": [], "nextPageToken": "x", "isLast": True})

    api, _ = make_api(last)
    assert (await fetch_page(api, "/search/jql", next_page_token())).next_cursor is None


@pytest.mark.anyio
async def test_error_response_has_no_next_cursor() -> None:
    api, _ = make_api(lambda _: httpx2.Response(404, json={"message": "Not Found"}))
    page = await fetch_page(api, "/x", page_token())
    assert page.response.status_code == 404
    assert page.next_cursor is None


# --- Cursor encoding ----------------------------------------------------------------------


def test_cursor_round_trip_and_opacity() -> None:
    encoded = encode_cursor("page_token", "abc/+=")
    assert "=" not in encoded
    assert "abc" not in encoded
    assert decode_cursor("page_token", encoded) == "abc/+="


@pytest.mark.parametrize(
    "bad",
    ["", "not-base64!", encode_cursor("cursor", "x"), "eyJmb28iOjF9", "bm90IGpzb24"],
)
def test_invalid_or_foreign_cursor_is_a_tool_error(bad: str) -> None:
    with pytest.raises(ToolError, match="Invalid cursor"):
        decode_cursor("page_token", bad)


# --- Internal iteration (sandbox) ---------------------------------------------------------


@pytest.mark.anyio
@style_params
async def test_iter_pages_collects_everything(
    paginator: Callable[[], Paginator], handler: Handler, url: str
) -> None:
    api, seen = make_api(handler)
    collected = [i async for r in iter_pages(api, url, paginator()) for i in items_of(r)]
    assert collected == ITEMS
    assert len(seen) == 3


@pytest.mark.anyio
async def test_iter_pages_detects_cycles() -> None:
    tokens = iter(["a", "b", "a"])

    def cycling(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"files": [], "nextPageToken": next(tokens)})

    api, _ = make_api(cycling)
    with pytest.raises(PaginationLoop, match="'a' repeated"):
        async for _ in iter_pages(api, "/files", page_token()):
            pass


@pytest.mark.anyio
async def test_iter_pages_respects_max_pages() -> None:
    counter = iter(range(100))

    def endless(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"files": [], "nextPageToken": str(next(counter))})

    api, seen = make_api(endless)
    pages = [r async for r in iter_pages(api, "/files", page_token(), max_pages=4)]
    assert len(pages) == 4
    assert len(seen) == 4
