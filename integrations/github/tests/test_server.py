"""GitHub MCP server end to end: `Client(server)` -> tools -> client -> in-memory fake."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from mcp.client import Client
from mcp.types import CallToolResult, TextContent

from bench_core.auth import StaticToken
from bench_core.testing import FakeClock, Fault

from bench_github.fake import OWNER, REPO, GitHubFake, seeded_fake
from bench_github.server import create_server
from bench_github.tools import BODY_LIMIT, FILE_LIMIT

pytestmark = pytest.mark.anyio

TOKEN = "server-token"
ALL = ["issues:write", "pull_requests:read", "contents:read"]
REPO_ARGS = {"owner": OWNER, "repo": REPO}
TOOL_NAMES = {
    "github_search_issues",
    "github_get_issue",
    "github_create_issue",
    "github_add_issue_comment",
    "github_list_pull_requests",
    "github_get_file_contents",
}


@pytest.fixture
def fake() -> GitHubFake:
    fake = seeded_fake(FakeClock())
    fake.issue_token(TOKEN, ALL)
    return fake


def server_for(fake: GitHubFake, **kwargs: Any):
    kwargs.setdefault("read_only", False)
    return create_server(
        credential=StaticToken("github", TOKEN),
        transport=fake.transport(),
        clock=FakeClock(),
        variant="v1",
        **kwargs,
    )


@pytest.fixture
async def client(fake: GitHubFake) -> AsyncIterator[Client]:
    async with Client(server_for(fake)) as client:
        yield client


async def call(client: Client, tool: str, **arguments: Any) -> dict[str, Any]:
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, text(result)
    assert result.structured_content is not None
    return result.structured_content


def text(result: CallToolResult) -> str:
    return "\n".join(c.text for c in result.content if isinstance(c, TextContent))


# --- Registration -------------------------------------------------------------------------


async def test_tools_descriptions_and_annotations(client: Client) -> None:
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == TOOL_NAMES
    search = tools["github_search_issues"]
    assert search.description is not None
    assert search.description.startswith("Search issues")
    assert "repo:octocat/hello-world" in search.input_schema["properties"]["query"]["description"]
    assert search.input_schema["properties"]["per_page"]["maximum"] == 50
    assert "ctx" not in search.input_schema["properties"]
    assert search.output_schema is not None
    create = tools["github_create_issue"]
    assert create.annotations is not None
    assert create.annotations.read_only_hint is False
    assert tools["github_get_issue"].annotations.read_only_hint is True  # type: ignore[union-attr]


async def test_read_only_hides_write_tools(fake: GitHubFake) -> None:
    server = server_for(fake, read_only=True)
    async with Client(server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert names == TOOL_NAMES - {"github_create_issue", "github_add_issue_comment"}
    assert "BENCH_READ_ONLY" in server.selection.disabled["github_create_issue"]


async def test_granted_permissions_filter_tools(fake: GitHubFake) -> None:
    server = server_for(fake, granted=frozenset({"issues:write"}))
    async with Client(server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    # issues:write implies issues:read for GitHub Apps.
    assert names == {
        "github_search_issues",
        "github_get_issue",
        "github_create_issue",
        "github_add_issue_comment",
    }
    assert server.selection.disabled["github_get_file_contents"] == (
        "missing scope(s): contents:read"
    )


# --- Search -------------------------------------------------------------------------------


async def test_search_concise_then_next_page(client: Client) -> None:
    query = f"repo:{OWNER}/{REPO} is:open label:bug"
    first = await call(client, "github_search_issues", query=query, per_page=3)
    assert first["total_count"] == 5
    assert set(first["items"][0]) == {"repository", "number", "title", "state", "labels"}
    second = await call(
        client, "github_search_issues", query=query, per_page=3, cursor=first["next_cursor"]
    )
    assert len(second["items"]) == 2
    assert "next_cursor" not in second


async def test_search_detailed_and_pull_request_flag(client: Client) -> None:
    result = await call(
        client,
        "github_search_issues",
        query=f"repo:{OWNER}/{REPO} is:pr",
        per_page=1,
        response_format="detailed",
    )
    (hit,) = result["items"]
    assert hit["is_pull_request"] is True
    assert {"author", "comments", "created_at", "updated_at", "html_url"} <= set(hit)


async def test_argument_errors_are_short(client: Client) -> None:
    result = await client.call_tool("github_search_issues", {"query": "bug", "per_page": 500})
    assert result.is_error
    message = text(result)
    assert "per_page" in message
    assert "less than or equal to 50" in message
    assert "pydantic" not in message


async def test_bad_cursor_is_actionable(client: Client) -> None:
    result = await client.call_tool("github_search_issues", {"query": "bug", "cursor": "zzz"})
    assert result.is_error
    assert "Invalid cursor" in text(result)


# --- Issues -------------------------------------------------------------------------------


async def test_get_issue_concise(client: Client) -> None:
    issue = await call(client, "github_get_issue", **REPO_ARGS, number=1)
    assert issue["title"] == "Crash when config file is empty"
    assert issue["comments"] == 3
    assert "KeyError" in issue["body"]
    assert "comment_list" not in issue
    assert "author" not in issue


async def test_get_issue_detailed_includes_comments(client: Client) -> None:
    issue = await call(
        client, "github_get_issue", **REPO_ARGS, number=1, response_format="detailed"
    )
    assert [c["author"] for c in issue["comment_list"]] == [
        "bench-contributor",
        "bench-maintainer",
        "bench-contributor",
    ]
    assert issue["milestone"] == "v2.4"
    assert "more_comments" not in issue


async def test_long_body_is_clipped_and_flagged(client: Client, fake: GitHubFake) -> None:
    repo = fake.repos[f"{OWNER}/{REPO}"]
    fake.add_issue(repo, "Huge log", author=OWNER, body="x" * 5_000)
    number = max(repo.issues)
    issue = await call(client, "github_get_issue", **REPO_ARGS, number=number)
    assert len(issue["body"]) == BODY_LIMIT["concise"]
    assert issue["body_truncated"] is True


async def test_missing_issue_is_a_tool_error(client: Client) -> None:
    result = await client.call_tool("github_get_issue", {**REPO_ARGS, "number": 999})
    assert result.is_error
    assert "was not found on GitHub, or the credentials have no access" in text(result)


async def test_create_issue_and_comment(client: Client, fake: GitHubFake) -> None:
    created = await call(
        client,
        "github_create_issue",
        **REPO_ARGS,
        title="Export fails",
        body="Steps ...",
        labels=["bug"],
    )
    assert created["labels"] == ["bug"]
    comment = await call(
        client, "github_add_issue_comment", **REPO_ARGS, number=created["number"], body="On it."
    )
    assert comment["issue_number"] == created["number"]
    assert fake.repos[f"{OWNER}/{REPO}"].issues[created["number"]]["comments"] == 1


async def test_permission_denied_reaches_the_model(fake: GitHubFake) -> None:
    fake.issue_token(TOKEN, ["issues:read", "pull_requests:read", "contents:read"])
    async with Client(server_for(fake)) as client:
        result = await client.call_tool("github_create_issue", {**REPO_ARGS, "title": "x"})
    assert result.is_error
    assert "Missing permission: issues:write" in text(result)


async def test_rate_limit_beyond_wait_budget(client: Client, fake: GitHubFake) -> None:
    fake.inject(Fault.rate_limit(retry_after=120), path="/search/issues")
    result = await client.call_tool("github_search_issues", {"query": "bug"})
    assert result.is_error
    assert "rate limited, retry after 120 s" in text(result)


# --- Pull requests and files --------------------------------------------------------------


async def test_list_pull_requests(client: Client) -> None:
    result = await call(client, "github_list_pull_requests", **REPO_ARGS, state="all")
    assert len(result["items"]) == 4
    first = result["items"][0]
    assert {"number", "title", "state", "draft", "author", "head", "base"} <= set(first)
    assert "html_url" not in first
    assert any("merged_at" in pr for pr in result["items"])


async def test_get_file_and_directory(client: Client) -> None:
    readme = await call(client, "github_get_file_contents", **REPO_ARGS, path="README.md")
    assert readme["type"] == "file"
    assert readme["text"].startswith("# widgets")
    assert "sha" not in readme

    root = await call(client, "github_get_file_contents", **REPO_ARGS)
    assert root["type"] == "dir"
    assert {"name": "src", "type": "dir"} in root["entries"]


async def test_large_text_file_is_clipped_with_hint(client: Client, fake: GitHubFake) -> None:
    fake.add_file(fake.repos[f"{OWNER}/{REPO}"], "notes.txt", "a" * 10_000)
    result = await call(client, "github_get_file_contents", **REPO_ARGS, path="notes.txt")
    assert len(result["text"]) == FILE_LIMIT["concise"]
    assert result["text_truncated"] is True
    assert "response_format='detailed'" in result["note"]


async def test_binary_file_has_note_and_size(client: Client) -> None:
    result = await call(client, "github_get_file_contents", **REPO_ARGS, path="assets/logo.png")
    assert "text" not in result
    assert "Binary" in result["note"]
    assert result["size"] > 0


async def test_text_content_matches_structured_output(client: Client) -> None:
    result = await client.call_tool("github_get_issue", {**REPO_ARGS, "number": 1})
    assert '"title": "Crash when config file is empty"' in text(result)
    assert "null" not in text(result)
