"""Todoist MCP tools: thin wrappers that shape client results for the model.

`response_format="concise"` returns what the model needs for its next step (ids, titles,
states); `"detailed"` adds descriptions, timestamps and URLs. Fields that a format leaves
out are omitted from the output instead of being sent as null. Long texts are clipped
with a flag that says so.

Every tool call gets one wait budget for rate-limit retries (`TOOL_WAIT_BUDGET`).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

from mcp.server.mcpserver import Context
from pydantic import BaseModel, Field

from bench_core.mcp import ResponseFormat, ToolSpec, clip, lifespan_state, omit, tool_call

from bench_todoist import scopes
from bench_todoist.client import PROVIDER, TodoistClient
from bench_todoist.models import Item

TOOL_WAIT_BUDGET = 20.0
"""Seconds one tool call may spend waiting on rate limits before it reports them."""

DESCRIPTION_LIMIT = {"concise": 1_500, "detailed": 10_000}


@dataclass
class TodoistState:
    client: TodoistClient


@contextmanager
def _todoist(ctx: Context[TodoistState]) -> Iterator[TodoistClient]:
    """The client for one tool call: one wait budget for all its requests."""
    client = lifespan_state(ctx).client
    with tool_call(client.api, provider=PROVIDER, wait_budget=TOOL_WAIT_BUDGET):
        yield client


# --- Output models ------------------------------------------------------------------------


class Output(BaseModel):
    pass


class ItemHit(Output):
    id: str
    title: str
    status: str
    description: str | None = omit()
    description_truncated: bool | None = omit()
    updated_at: datetime | None = omit()
    url: str | None = omit()


class ItemList(Output):
    items: list[ItemHit]
    next_cursor: str | None = omit()


class ItemOut(Output):
    id: str
    title: str
    status: str
    description: str | None
    description_truncated: bool | None = omit()
    created_at: datetime | None = omit()
    updated_at: datetime | None = omit()
    url: str | None = omit()


class CreatedItem(Output):
    id: str
    title: str
    url: str


# --- Conversions --------------------------------------------------------------------------


def _hit(item: Item, fmt: ResponseFormat) -> ItemHit:
    hit = ItemHit(id=item.id, title=item.title, status=item.status)
    if fmt == "detailed":
        description, clipped = clip(item.description, DESCRIPTION_LIMIT["concise"])
        hit = hit.model_copy(
            update={
                "description": description,
                "description_truncated": clipped,
                "updated_at": item.updated_at,
                "url": item.url,
            }
        )
    return hit


def _item(item: Item, fmt: ResponseFormat) -> ItemOut:
    description, clipped = clip(item.description, DESCRIPTION_LIMIT[fmt])
    out = ItemOut(
        id=item.id,
        title=item.title,
        status=item.status,
        description=description,
        description_truncated=clipped,
    )
    if fmt == "detailed":
        out = out.model_copy(
            update={"created_at": item.created_at, "updated_at": item.updated_at, "url": item.url}
        )
    return out


# --- Tools --------------------------------------------------------------------------------

PageSize = Annotated[int, Field(ge=1, le=50)]


async def list_items(
    ctx: Context[TodoistState],
    query: str | None = None,
    cursor: str | None = None,
    page_size: PageSize = 10,
    response_format: ResponseFormat = "concise",
) -> ItemList:
    with _todoist(ctx) as client:
        page = await client.list_items(query=query, cursor=cursor, page_size=page_size)
    return ItemList(
        items=[_hit(i, response_format) for i in page.items], next_cursor=page.next_cursor
    )


async def get_item(
    item_id: str, ctx: Context[TodoistState], response_format: ResponseFormat = "concise"
) -> ItemOut:
    with _todoist(ctx) as client:
        item = await client.get_item(item_id)
    return _item(item, response_format)


async def create_item(
    title: str, ctx: Context[TodoistState], description: str | None = None
) -> CreatedItem:
    with _todoist(ctx) as client:
        item = await client.create_item(title, description=description)
    return CreatedItem(id=item.id, title=item.title, url=item.url)


TOOLS = [
    ToolSpec(list_items, scopes.LIST_ITEMS),
    ToolSpec(get_item, scopes.GET_ITEM),
    ToolSpec(create_item, scopes.CREATE_ITEM),
]
