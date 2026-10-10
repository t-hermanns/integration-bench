"""MCP server end to end: `Client(server)` -> tools -> client -> in-memory fake."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from mcp.client import Client
from mcp.types import CallToolResult, TextContent

from bench_core.testing import FakeClock

from bench_gdrive.fake import SANDBOX, GdriveFake, credential, seeded_fake
from bench_gdrive.fake.backend import ROOT_ID
from bench_gdrive.scopes import DRIVE, DRIVE_FILE
from bench_gdrive.server import create_server

pytestmark = pytest.mark.anyio

TOKEN = "server-token"
SEARCH = "gdrive_search_files"
META = "gdrive_get_file_metadata"
READ = "gdrive_read_file_content"
LIST = "gdrive_list_folder"
CREATE = "gdrive_create_document"
TOOL_NAMES = {SEARCH, META, READ, LIST, CREATE}


@pytest.fixture
def fake() -> GdriveFake:
    fake = seeded_fake(FakeClock())
    fake.issue_token(TOKEN, [DRIVE_FILE])
    return fake


def file_id(fake: GdriveFake, name: str) -> str:
    return next(f["id"] for f in fake.files.values() if f["name"] == name)


def server_for(fake: GdriveFake, **kwargs: Any):
    kwargs.setdefault("read_only", False)
    return create_server(
        credential=credential(TOKEN),
        transport=fake.transport(),
        clock=FakeClock(),
        variant="v1",
        **kwargs,
    )


@pytest.fixture
async def client(fake: GdriveFake) -> AsyncIterator[Client]:
    async with Client(server_for(fake)) as client:
        yield client


async def call(client: Client, tool: str, **arguments: Any) -> dict[str, Any]:
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, text(result)
    assert result.structured_content is not None
    return result.structured_content


async def error(client: Client, tool: str, **arguments: Any) -> str:
    result = await client.call_tool(tool, arguments)
    assert result.is_error
    return text(result)


def text(result: CallToolResult) -> str:
    return "\n".join(c.text for c in result.content if isinstance(c, TextContent))


# --- Registration -------------------------------------------------------------------------


async def test_tools_descriptions_and_annotations(client: Client) -> None:
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == TOOL_NAMES
    search = tools[SEARCH]
    assert search.description is not None
    assert search.description.startswith("Search files")
    assert "ctx" not in search.input_schema["properties"]
    assert search.input_schema["required"] == ["query"]
    assert search.input_schema["properties"]["page_size"]["maximum"] == 50
    assert search.output_schema is not None
    assert search.annotations is not None
    assert search.annotations.read_only_hint is True
    create = tools[CREATE]
    assert create.annotations is not None
    assert create.annotations.read_only_hint is False


async def test_read_only_hides_create(fake: GdriveFake) -> None:
    async with Client(server_for(fake, read_only=True)) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert names == TOOL_NAMES - {CREATE}


@pytest.mark.parametrize(
    ("granted", "expected"),
    [(frozenset(), set()), (frozenset({DRIVE}), TOOL_NAMES), (frozenset({DRIVE_FILE}), TOOL_NAMES)],
)
async def test_granted_scopes_filter_tools(
    fake: GdriveFake, granted: frozenset[str], expected: set[str]
) -> None:
    async with Client(server_for(fake, granted=granted)) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert names == expected


# --- Tools --------------------------------------------------------------------------------


async def test_list_root_then_folder_with_cursor(client: Client, fake: GdriveFake) -> None:
    root = await call(client, LIST)
    assert root["files"] == [{"id": file_id(fake, SANDBOX), "name": SANDBOX, "kind": "folder"}]
    first = await call(client, LIST, folder_id=root["files"][0]["id"], page_size=3)
    assert [f["name"] for f in first["files"]] == ["Meeting notes", "Projects", "glossary.txt"]
    second = await call(
        client, LIST, folder_id=root["files"][0]["id"], page_size=3, cursor=first["next_cursor"]
    )
    assert [f["name"] for f in second["files"]] == ["Onboarding guide"]
    assert "next_cursor" not in second


async def test_search_concise_and_detailed(client: Client, fake: GdriveFake) -> None:
    concise = await call(client, SEARCH, query="roadmap")
    assert [f["name"] for f in concise["files"]] == ["Onboarding guide", "Roadmap 2026"]
    assert set(concise["files"][0]) == {"id", "name", "kind"}
    detailed = await call(client, SEARCH, query="roadmap", response_format="detailed")
    assert {"modified_time", "size", "parent_id", "url"} <= set(detailed["files"][0])


async def test_search_quotes_the_text(client: Client, fake: GdriveFake) -> None:
    fake.add_file("Dana's notes", content="it's done")
    found = await call(client, SEARCH, query="dana's")
    assert [f["name"] for f in found["files"]] == ["Dana's notes"]
    assert "must not be empty" in await error(client, SEARCH, query="  ")


async def test_metadata_concise_and_detailed(client: Client, fake: GdriveFake) -> None:
    fid = file_id(fake, "Roadmap 2026")
    concise = await call(client, META, file_id=fid)
    assert concise["kind"] == "document"
    assert concise["parent_id"] == file_id(fake, "Projects")
    assert "url" not in concise
    assert "mime_type" not in concise
    detailed = await call(client, META, file_id=fid, response_format="detailed")
    assert detailed["mime_type"] == "application/vnd.google-apps.document"
    assert {"created_time", "size", "url"} <= set(detailed)


async def test_read_document_and_text_file(client: Client, fake: GdriveFake) -> None:
    doc = await call(client, READ, file_id=file_id(fake, "Roadmap 2026"))
    assert doc["text"].splitlines()[1] == "Q1: ship the CSV importer"
    assert "truncated" not in doc
    txt = await call(client, READ, file_id=file_id(fake, "glossary.txt"), max_chars=100)
    assert txt["kind"] == "text/plain"
    assert txt["text"].startswith("SSO: single sign-on\n")


async def test_long_text_is_clipped(client: Client, fake: GdriveFake) -> None:
    fid = fake.add_file("Long", content="x" * 500)["id"]
    result = await call(client, READ, file_id=fid, max_chars=100)
    assert (len(result["text"]), result["total_chars"], result["truncated"]) == (100, 500, True)


async def test_unreadable_files_are_explained(client: Client, fake: GdriveFake) -> None:
    folder = await error(client, READ, file_id=file_id(fake, "Projects"))
    assert "is a folder; use list_folder" in folder
    pdf = fake.add_file("scan.pdf", mime_type="application/pdf", content="%PDF")["id"]
    assert "without readable text" in await error(client, READ, file_id=pdf)


async def test_create_document_then_read_it(client: Client, fake: GdriveFake) -> None:
    folder = file_id(fake, "Projects")
    created = await call(
        client, CREATE, name="Retro", content="Line one\nLine two", folder_id=folder
    )
    assert created["name"] == "Retro"
    assert created["url"].startswith("https://docs.google.com/")
    assert fake.files[created["id"]]["parents"] == [folder]
    read = await call(client, READ, file_id=created["id"])
    assert read["text"] == "Line one\nLine two"
    in_root = await call(client, CREATE, name="Top level", content="")
    assert fake.files[in_root["id"]]["parents"] == [ROOT_ID]


async def test_create_document_validates(client: Client, fake: GdriveFake) -> None:
    assert "must not be empty" in await error(client, CREATE, name=" ", content="x")
    assert "at most 200" in await error(client, CREATE, name="x" * 201, content="x")
    doc = file_id(fake, "Roadmap 2026")
    assert "not found" in await error(client, CREATE, name="x", content="x", folder_id=doc)


async def test_errors_reach_the_model_as_tool_errors(client: Client) -> None:
    assert "not found" in await error(client, META, file_id="1AbCdEfGhIjKlMnOpQrStUv")
    assert "Invalid file id" in await error(client, META, file_id="../etc/passwd")
    assert "Invalid file id" in await error(client, LIST, folder_id="x' or 'a")
    assert "page_size" in await error(client, LIST, page_size=500)


async def test_missing_scope_names_it(fake: GdriveFake) -> None:
    fake.issue_token(TOKEN, [])
    async with Client(server_for(fake)) as client:
        message = await error(client, SEARCH, query="roadmap")
    assert "Permission denied" in message
    assert DRIVE_FILE in message
