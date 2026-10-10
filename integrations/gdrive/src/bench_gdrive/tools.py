"""Google Drive MCP tools: thin wrappers that shape client results for the model.

`response_format="concise"` returns what the model needs for its next step (ids, names,
kinds); `"detailed"` adds timestamps, sizes and links. Fields that a format leaves out
are omitted from the output instead of being sent as null. Long file texts are clipped
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

from bench_gdrive import scopes
from bench_gdrive.client import PROVIDER, GdriveClient
from bench_gdrive.models import File

TOOL_WAIT_BUDGET = 20.0
"""Seconds one tool call may spend waiting on rate limits before it reports them."""

DESCRIPTION_LIMIT = {"concise": 300, "detailed": 2_000}


@dataclass
class GdriveState:
    client: GdriveClient


@contextmanager
def _gdrive(ctx: Context[GdriveState]) -> Iterator[GdriveClient]:
    """The client for one tool call: one wait budget for all its requests."""
    client = lifespan_state(ctx).client
    with tool_call(client.api, provider=PROVIDER, wait_budget=TOOL_WAIT_BUDGET):
        yield client


# --- Output models ------------------------------------------------------------------------


class Output(BaseModel):
    pass


class FileHit(Output):
    id: str
    name: str
    kind: str
    modified_time: datetime | None = omit()
    size: int | None = omit()
    parent_id: str | None = omit()
    url: str | None = omit()


class FileList(Output):
    files: list[FileHit]
    next_cursor: str | None = omit()


class FileOut(Output):
    id: str
    name: str
    kind: str
    parent_id: str | None = omit()
    modified_time: datetime | None = omit()
    mime_type: str | None = omit()
    created_time: datetime | None = omit()
    size: int | None = omit()
    description: str | None = omit()
    description_truncated: bool | None = omit()
    url: str | None = omit()


class FileContent(Output):
    id: str
    name: str
    kind: str
    text: str
    total_chars: int
    truncated: bool | None = omit()


class CreatedDocument(Output):
    id: str
    name: str
    url: str | None = omit()


# --- Conversions --------------------------------------------------------------------------


def _parent(file: File) -> str | None:
    return file.parents[0] if file.parents else None


def _hit(file: File, fmt: ResponseFormat) -> FileHit:
    hit = FileHit(id=file.id, name=file.name, kind=file.kind)
    if fmt == "detailed":
        hit = hit.model_copy(
            update={
                "modified_time": file.modified_time,
                "size": file.size,
                "parent_id": _parent(file),
                "url": file.web_view_link,
            }
        )
    return hit


def _file(file: File, fmt: ResponseFormat) -> FileOut:
    out = FileOut(
        id=file.id,
        name=file.name,
        kind=file.kind,
        parent_id=_parent(file),
        modified_time=file.modified_time,
    )
    if fmt == "detailed":
        description, clipped = clip(file.description, DESCRIPTION_LIMIT[fmt])
        out = out.model_copy(
            update={
                "mime_type": file.mime_type,
                "created_time": file.created_time,
                "size": file.size,
                "description": description,
                "description_truncated": clipped,
                "url": file.web_view_link,
            }
        )
    elif file.description:
        description, clipped = clip(file.description, DESCRIPTION_LIMIT[fmt])
        out = out.model_copy(update={"description": description, "description_truncated": clipped})
    return out


# --- Tools --------------------------------------------------------------------------------

PageSize = Annotated[int, Field(ge=1, le=50)]
MaxChars = Annotated[int, Field(ge=100, le=50_000)]


async def search_files(
    query: str,
    ctx: Context[GdriveState],
    folder_id: str | None = None,
    cursor: str | None = None,
    page_size: PageSize = 10,
    response_format: ResponseFormat = "concise",
) -> FileList:
    with _gdrive(ctx) as client:
        page = await client.search_files(
            query, folder_id=folder_id, cursor=cursor, page_size=page_size
        )
    return FileList(
        files=[_hit(f, response_format) for f in page.items], next_cursor=page.next_cursor
    )


async def list_folder(
    ctx: Context[GdriveState],
    folder_id: str = "root",
    cursor: str | None = None,
    page_size: PageSize = 20,
    response_format: ResponseFormat = "concise",
) -> FileList:
    with _gdrive(ctx) as client:
        page = await client.list_folder(folder_id, cursor=cursor, page_size=page_size)
    return FileList(
        files=[_hit(f, response_format) for f in page.items], next_cursor=page.next_cursor
    )


async def get_file_metadata(
    file_id: str, ctx: Context[GdriveState], response_format: ResponseFormat = "concise"
) -> FileOut:
    with _gdrive(ctx) as client:
        file = await client.get_file(file_id)
    return _file(file, response_format)


async def read_file_content(
    file_id: str, ctx: Context[GdriveState], max_chars: MaxChars = 8_000
) -> FileContent:
    with _gdrive(ctx) as client:
        content = await client.read_file(file_id)
    text, clipped = clip(content.text, max_chars)
    file = content.file
    return FileContent(
        id=file.id,
        name=file.name,
        kind=file.kind,
        text=text or "",
        total_chars=len(content.text),
        truncated=clipped,
    )


async def create_document(
    name: str, content: str, ctx: Context[GdriveState], folder_id: str | None = None
) -> CreatedDocument:
    with _gdrive(ctx) as client:
        file = await client.create_document(name, content, folder_id=folder_id)
    return CreatedDocument(id=file.id, name=file.name, url=file.web_view_link)


TOOLS = [
    ToolSpec(search_files, scopes.SEARCH_FILES),
    ToolSpec(get_file_metadata, scopes.GET_FILE_METADATA),
    ToolSpec(read_file_content, scopes.READ_FILE_CONTENT),
    ToolSpec(list_folder, scopes.LIST_FOLDER),
    ToolSpec(create_document, scopes.CREATE_DOCUMENT),
]
