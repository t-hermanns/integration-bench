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
from bench_todoist.models import Task

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


class TaskHit(Output):
    id: str
    title: str
    status: str
    description: str | None = omit()
    description_truncated: bool | None = omit()
    updated_at: datetime | None = omit()
    url: str | None = omit()


class TaskList(Output):
    tasks: list[TaskHit]
    next_cursor: str | None = omit()


class TaskOut(Output):
    id: str
    title: str
    status: str
    description: str | None
    description_truncated: bool | None = omit()
    created_at: datetime | None = omit()
    updated_at: datetime | None = omit()
    url: str | None = omit()


class CreatedTask(Output):
    id: str
    title: str
    url: str


# --- Conversions --------------------------------------------------------------------------


def _hit(task: Task, fmt: ResponseFormat) -> TaskHit:
    hit = TaskHit(id=task.id, title=task.title, status=task.status)
    if fmt == "detailed":
        description, clipped = clip(task.description, DESCRIPTION_LIMIT["concise"])
        hit = hit.model_copy(
            update={
                "description": description,
                "description_truncated": clipped,
                "updated_at": task.updated_at,
                "url": task.url,
            }
        )
    return hit


def _task(task: Task, fmt: ResponseFormat) -> TaskOut:
    description, clipped = clip(task.description, DESCRIPTION_LIMIT[fmt])
    out = TaskOut(
        id=task.id,
        title=task.title,
        status=task.status,
        description=description,
        description_truncated=clipped,
    )
    if fmt == "detailed":
        out = out.model_copy(
            update={"created_at": task.created_at, "updated_at": task.updated_at, "url": task.url}
        )
    return out


# --- Tools --------------------------------------------------------------------------------

PageSize = Annotated[int, Field(ge=1, le=50)]


async def list_tasks(
    ctx: Context[TodoistState],
    query: str | None = None,
    cursor: str | None = None,
    page_size: PageSize = 10,
    response_format: ResponseFormat = "concise",
) -> TaskList:
    with _todoist(ctx) as client:
        page = await client.list_tasks(query=query, cursor=cursor, page_size=page_size)
    return TaskList(
        tasks=[_hit(i, response_format) for i in page.items], next_cursor=page.next_cursor
    )


async def get_task(
    task_id: str, ctx: Context[TodoistState], response_format: ResponseFormat = "concise"
) -> TaskOut:
    with _todoist(ctx) as client:
        task = await client.get_task(task_id)
    return _task(task, response_format)


async def create_task(
    title: str, ctx: Context[TodoistState], description: str | None = None
) -> CreatedTask:
    with _todoist(ctx) as client:
        task = await client.create_task(title, description=description)
    return CreatedTask(id=task.id, title=task.title, url=task.url)


TOOLS = [
    ToolSpec(list_tasks, scopes.LIST_TASKS),
    ToolSpec(get_task, scopes.GET_TASK),
    ToolSpec(create_task, scopes.CREATE_TASK),
]
