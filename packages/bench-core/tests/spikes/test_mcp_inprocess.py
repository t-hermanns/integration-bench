"""Spike: mcp 2.3 `MCPServer` driven in-process through `Client(server)`.

Covers what bench_core.mcp will rely on: a lifespan that owns an httpx2 client,
tools reading it from the context, descriptions/annotations set at registration time,
conditional registration via `add_tool`, `ToolError` surfacing as `is_error`, and
structured output.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx2
import pytest
from mcp.client import Client
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import TextContent, ToolAnnotations
from pydantic import BaseModel


@dataclass
class AppState:
    http: httpx2.AsyncClient


class Issue(BaseModel):
    number: int
    title: str


def _backend(request: httpx2.Request) -> httpx2.Response:
    if request.url.path == "/issues/1":
        return httpx2.Response(200, json={"number": 1, "title": "First"})
    if request.url.path == "/issues/403":
        return httpx2.Response(403, json={"message": "Resource not accessible"})
    return httpx2.Response(404)


def build_server(*, read_only: bool) -> tuple[MCPServer, list[str]]:
    events: list[str] = []

    @asynccontextmanager
    async def lifespan(_: MCPServer) -> AsyncIterator[AppState]:
        events.append("open")
        async with httpx2.AsyncClient(
            base_url="https://api.example.invalid", transport=httpx2.MockTransport(_backend)
        ) as http:
            yield AppState(http=http)
        events.append("close")

    server = MCPServer("spike", lifespan=lifespan)

    async def github_get_issue(number: int, ctx: Context[AppState]) -> Issue:
        http = ctx.request_context.lifespan_context.http
        r = await http.get(f"/issues/{number}")
        if r.status_code == 403:
            raise ToolError("Permission denied: needs scope 'issues:read'.")
        if r.status_code == 404:
            raise ToolError(f"Issue {number} not found or not accessible.")
        return Issue.model_validate(r.json())

    async def github_create_issue(title: str) -> str:
        return f"created {title}"

    server.add_tool(
        github_get_issue,
        description="Get one issue by number. (v1 description from YAML)",
        annotations=ToolAnnotations(read_only_hint=True),
    )
    if not read_only:
        server.add_tool(
            github_create_issue,
            description="Create an issue.",
            annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False),
        )
    return server, events


@pytest.mark.anyio
async def test_list_tools_exposes_schema_description_annotations() -> None:
    server, _ = build_server(read_only=False)
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}

    assert set(tools) == {"github_get_issue", "github_create_issue"}
    get = tools["github_get_issue"]
    assert get.description == "Get one issue by number. (v1 description from YAML)"
    assert get.annotations is not None
    assert get.annotations.read_only_hint is True
    assert get.input_schema["required"] == ["number"]
    assert "ctx" not in get.input_schema["properties"]
    assert get.output_schema is not None
    assert set(get.output_schema["properties"]) == {"number", "title"}


@pytest.mark.anyio
async def test_read_only_hides_write_tools() -> None:
    server, _ = build_server(read_only=True)
    async with Client(server) as client:
        names = [t.name for t in (await client.list_tools()).tools]
    assert names == ["github_get_issue"]


@pytest.mark.anyio
async def test_call_tool_uses_lifespan_client_and_structured_output() -> None:
    server, events = build_server(read_only=True)
    async with Client(server) as client:
        result = await client.call_tool("github_get_issue", {"number": 1})
        assert events == ["open"]
    assert events == ["open", "close"]

    assert not result.is_error
    assert result.structured_content == {"number": 1, "title": "First"}
    assert isinstance(result.content[0], TextContent)
    assert '"title"' in result.content[0].text


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("number", "message"),
    [(403, "Permission denied: needs scope 'issues:read'."), (7, "not found or not accessible")],
)
async def test_tool_error_becomes_is_error_result(number: int, message: str) -> None:
    server, _ = build_server(read_only=True)
    async with Client(server) as client:
        result = await client.call_tool("github_get_issue", {"number": number})
    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    assert message in result.content[0].text


@pytest.mark.anyio
async def test_invalid_arguments_are_reported_to_model() -> None:
    server, _ = build_server(read_only=True)
    async with Client(server) as client:
        result = await client.call_tool("github_get_issue", {"number": "not-a-number"})
    assert result.is_error
