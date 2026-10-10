"""MCP server end to end: `Client(server)` -> tools -> client -> in-memory fake."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from mcp.client import Client
from mcp.types import CallToolResult, TextContent

from bench_core.testing import FakeClock

from bench_jira.client import SITE_URL
from bench_jira.fake import JiraFake, credential, seeded_fake
from bench_jira.scopes import READ, WRITE
from bench_jira.server import create_server
from bench_jira.tools import DESCRIPTION_LIMIT, MAX_COMMENTS

pytestmark = pytest.mark.anyio

TOKEN = "server-token"
SEARCH = "jira_search_issues"
GET = "jira_get_issue"
CREATE = "jira_create_issue"
COMMENT = "jira_add_comment"
TRANSITION = "jira_transition_issue"
TOOL_NAMES = {SEARCH, GET, CREATE, COMMENT, TRANSITION}
WRITE_TOOLS = {CREATE, COMMENT, TRANSITION}


@pytest.fixture
def fake() -> JiraFake:
    fake = seeded_fake(FakeClock())
    fake.issue_token(TOKEN, [READ, WRITE])
    return fake


def server_for(fake: JiraFake, **kwargs: Any):
    kwargs.setdefault("read_only", False)
    return create_server(
        credential=credential(TOKEN),
        transport=fake.transport(),
        clock=FakeClock(),
        variant="v1",
        **kwargs,
    )


@pytest.fixture
async def client(fake: JiraFake) -> AsyncIterator[Client]:
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
    assert "JQL" in search.description
    assert "ctx" not in search.input_schema["properties"]
    assert search.input_schema["required"] == ["jql"]
    assert search.input_schema["properties"]["page_size"]["maximum"] == 50
    assert search.output_schema is not None
    assert search.annotations is not None
    assert search.annotations.read_only_hint is True
    for name in WRITE_TOOLS:
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.read_only_hint is False, name


async def test_read_only_hides_write_tools(fake: JiraFake) -> None:
    async with Client(server_for(fake, read_only=True)) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert names == TOOL_NAMES - WRITE_TOOLS


async def test_missing_scopes_hide_tools(fake: JiraFake) -> None:
    async with Client(server_for(fake, granted=frozenset({READ}))) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert names == TOOL_NAMES - WRITE_TOOLS


async def test_api_tokens_keep_every_tool(fake: JiraFake) -> None:
    async with Client(server_for(fake, granted=None)) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert names == TOOL_NAMES


# --- Search -------------------------------------------------------------------------------


async def test_search_pages_with_cursor(client: Client) -> None:
    jql = "project = BENCH AND labels = crash ORDER BY key ASC"
    first = await call(client, SEARCH, jql=jql, page_size=2)
    assert [i["key"] for i in first["issues"]] == ["BENCH-1", "BENCH-7"]
    assert set(first["issues"][0]) == {"key", "summary", "status", "issue_type"}
    second = await call(client, SEARCH, jql=jql, page_size=2, cursor=first["next_cursor"])
    assert [i["key"] for i in second["issues"]] == ["BENCH-12"]
    assert "next_cursor" not in second


async def test_search_detailed(client: Client) -> None:
    page = await call(client, SEARCH, jql="key = BENCH-8", response_format="detailed")
    hit = page["issues"][0]
    assert hit["assignee"] == "Bench Bot"
    assert hit["labels"] == ["i18n", "ui"]
    assert hit["url"] == f"{SITE_URL}/browse/BENCH-8"
    assert {"priority", "updated"} <= set(hit)


@pytest.mark.parametrize(
    ("jql", "keys"),
    [
        (
            'project = BENCH AND status = "In Progress" ORDER BY key',
            ["BENCH-3", "BENCH-8", "BENCH-12"],
        ),
        (
            "project = BENCH AND assignee = currentUser() ORDER BY key",
            ["BENCH-6", "BENCH-8", "BENCH-10"],
        ),
        (
            "project = BENCH AND statusCategory != Done AND priority = Highest ORDER BY key",
            ["BENCH-1", "BENCH-12"],
        ),
        (
            "project = BENCH AND type = Story AND assignee is EMPTY ORDER BY key",
            ["BENCH-2", "BENCH-11"],
        ),
        ('project = BENCH AND text ~ "stale file"', ["BENCH-12"]),
        ("labels in (docs, i18n) ORDER BY priority DESC, key ASC", ["BENCH-4", "BENCH-8"]),
    ],
)
async def test_search_jql(client: Client, jql: str, keys: list[str]) -> None:
    page = await call(client, SEARCH, jql=jql, page_size=50)
    assert [i["key"] for i in page["issues"]] == keys


async def test_search_errors_reach_the_model(client: Client) -> None:
    assert "Unbounded JQL" in await error(client, SEARCH, jql="ORDER BY created DESC")
    assert "must not be empty" in await error(client, SEARCH, jql="  ")
    message = await error(client, SEARCH, jql="project = BENCH AND (status = Done")
    assert "Expecting ')' before the end of the query." in message
    assert "Fix the arguments" in message
    assert "Invalid cursor" in await error(client, SEARCH, jql="project = BENCH", cursor="abc")


# --- Get ----------------------------------------------------------------------------------


async def test_get_issue_concise_and_detailed(client: Client) -> None:
    concise = await call(client, GET, issue_key="bench-1")
    assert concise["key"] == "BENCH-1"
    assert concise["status"] == "To Do"
    assert concise["comment_count"] == 2
    assert concise["description"].startswith("Steps to reproduce:\n1. Open the shop")
    assert "comments" not in concise
    assert "url" not in concise
    detailed = await call(client, GET, issue_key="BENCH-1", response_format="detailed")
    assert detailed["reporter"] == "Bench Bot"
    assert detailed["status_category"] == "new"
    assert [c["author"] for c in detailed["comments"]] == ["Bench Bot", "Bench Bot"]
    assert "more_comments" not in detailed
    assert {"created", "updated", "url", "project"} <= set(detailed)


async def test_long_texts_are_clipped(client: Client, fake: JiraFake) -> None:
    issue = fake.add_issue("Long", description="x" * 20_000)
    for n in range(MAX_COMMENTS + 2):
        fake.add_comment(issue["key"], f"comment {n} " + "y" * 2_000)
    concise = await call(client, GET, issue_key=issue["key"])
    assert len(concise["description"]) == DESCRIPTION_LIMIT["concise"]
    assert concise["description_truncated"] is True
    detailed = await call(client, GET, issue_key=issue["key"], response_format="detailed")
    assert len(detailed["description"]) == DESCRIPTION_LIMIT["detailed"]
    assert len(detailed["comments"]) == MAX_COMMENTS
    assert detailed["more_comments"] is True
    assert detailed["comments"][0]["body_truncated"] is True


# --- Writes -------------------------------------------------------------------------------


async def test_create_issue_then_get_it(client: Client, fake: JiraFake) -> None:
    created = await call(
        client,
        CREATE,
        project_key="bench",
        summary="New bug",
        issue_type="Bug",
        description="First line\nsecond line\n\n- a\n- b",
        labels=["triage"],
    )
    assert created == {"key": "BENCH-13", "url": f"{SITE_URL}/browse/BENCH-13"}
    stored = fake.issues["BENCH-13"]["fields"]["description"]
    assert [n["type"] for n in stored["content"]] == ["paragraph", "bulletList"]
    fetched = await call(client, GET, issue_key="BENCH-13")
    assert fetched["description"] == "First line\nsecond line\n\n- a\n- b"
    assert (fetched["issue_type"], fetched["labels"]) == ("Bug", ["triage"])


async def test_create_issue_errors(client: Client) -> None:
    assert "Specify a valid issue type" in await error(
        client, CREATE, project_key="BENCH", summary="x", issue_type="Epic"
    )
    assert "valid project is required" in await error(
        client, CREATE, project_key="NOPE", summary="x"
    )
    assert "Invalid project key" in await error(client, CREATE, project_key="b", summary="x")
    assert "contain spaces" in await error(
        client, CREATE, project_key="BENCH", summary="x", labels=["two words"]
    )
    assert "one line" in await error(client, CREATE, project_key="BENCH", summary="a\nb")


async def test_add_comment(client: Client, fake: JiraFake) -> None:
    added = await call(client, COMMENT, issue_key="BENCH-2", body="Looks good.")
    assert added["issue_key"] == "BENCH-2"
    assert added["url"].endswith(f"/browse/BENCH-2?focusedCommentId={added['comment_id']}")
    issue = await call(client, GET, issue_key="BENCH-2", response_format="detailed")
    assert issue["comments"][-1] == {
        "author": "Bench Bot",
        "body": "Looks good.",
        "created": issue["comments"][-1]["created"],
    }
    assert "must not be empty" in await error(client, COMMENT, issue_key="BENCH-2", body=" ")


async def test_transition_by_status_name(client: Client) -> None:
    moved = await call(client, TRANSITION, issue_key="BENCH-4", status="in progress")
    assert moved == {"key": "BENCH-4", "status": "In Progress", "url": f"{SITE_URL}/browse/BENCH-4"}
    assert (await call(client, GET, issue_key="BENCH-4"))["status"] == "In Progress"
    review = await call(client, TRANSITION, issue_key="BENCH-4", status="In Review")
    assert review["status"] == "In Review"
    done = await call(client, TRANSITION, issue_key="BENCH-4", status="Done")
    assert done["status"] == "Done"
    reopened = await call(client, TRANSITION, issue_key="BENCH-4", status="to do")
    assert reopened["status"] == "To Do"


async def test_transition_to_unknown_status_lists_options(client: Client) -> None:
    message = await error(client, TRANSITION, issue_key="BENCH-5", status="Closed")
    assert "cannot move to 'Closed'" in message
    assert "In Progress, In Review, To Do" in message


async def test_errors_reach_the_model_as_tool_errors(client: Client) -> None:
    missing = await error(client, GET, issue_key="BENCH-999")
    assert "not found" in missing
    assert "Invalid issue key" in await error(client, GET, issue_key="../etc")
    assert "page_size" in await error(client, SEARCH, jql="project = BENCH", page_size=500)
