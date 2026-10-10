import pytest
from mcp.client import Client
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import TextContent

from bench_core.permissions import (
    NotFoundOrNoAccess,
    PermissionDenied,
    Requirement,
    ToolAccess,
    any_of,
    expand_implied,
    expand_levels,
    read_only_from_env,
    requires,
    scopes_from_header,
    select_tools,
)

TOOLS = [
    ToolAccess("github_search_issues", requires("issues:read")),
    ToolAccess("github_get_file_contents", requires("contents:read")),
    ToolAccess("github_create_issue", requires("issues:write"), write=True),
    ToolAccess(
        "github_merge_pull_request",
        requires("pull_requests:write", "contents:write"),
        write=True,
        destructive=True,
    ),
    ToolAccess("github_whoami"),
]


def names(tools: tuple[ToolAccess, ...]) -> list[str]:
    return [t.name for t in tools]


# --- Scope expansion and parsing ----------------------------------------------------------


def test_expand_levels() -> None:
    assert expand_levels({"issues:write", "contents:admin", "metadata:read", "repo"}) == {
        "issues:write",
        "issues:read",
        "contents:admin",
        "contents:write",
        "contents:read",
        "metadata:read",
        "repo",
    }


def test_expand_levels_keeps_urls_and_non_level_suffixes() -> None:
    granted = {"https://www.googleapis.com/auth/drive.file", "read:jira-work", "chat:write"}
    assert expand_levels(granted) == granted | {"chat:read"}


def test_expand_implied_is_transitive() -> None:
    implies = {"repo": ["public_repo", "repo:status"], "repo:status": ["repo:status:read"]}
    assert expand_implied({"repo"}, implies) == {
        "repo",
        "public_repo",
        "repo:status",
        "repo:status:read",
    }


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("repo, read:org", {"repo", "read:org"}),
        ("channels:read,chat:write", {"channels:read", "chat:write"}),
        ("", set()),
        (None, None),
    ],
)
def test_scopes_from_header(header: str | None, expected: set[str] | None) -> None:
    result = scopes_from_header(header)
    assert result == (None if expected is None else frozenset(expected))


# --- Requirements -------------------------------------------------------------------------


def test_requirement_missing() -> None:
    req = any_of(requires("a", "b"), requires("c"))
    assert req.missing(frozenset({"c"})) is None
    assert req.missing(frozenset({"a", "b"})) is None
    assert req.missing(frozenset({"a"})) == {"b"}
    assert req.missing(frozenset()) == {"c"}  # smallest gap
    assert req.missing(None) is None  # unknown grants: do not hide
    assert req.describe() == "a + b or c"


def test_empty_requirement_is_always_satisfied() -> None:
    assert Requirement().missing(frozenset()) is None
    assert requires().describe() == "no scopes"


# --- Tool selection -----------------------------------------------------------------------


def test_select_tools_drops_uncovered_tools_with_reason() -> None:
    selection = select_tools(
        TOOLS, frozenset({"issues:write", "metadata:read"}), expand=expand_levels
    )
    assert names(selection.enabled) == [
        "github_search_issues",  # issues:write implies issues:read
        "github_create_issue",
        "github_whoami",
    ]
    assert selection.disabled == {
        "github_get_file_contents": "missing scope(s): contents:read",
        "github_merge_pull_request": "missing scope(s): contents:write, pull_requests:write",
    }


def test_select_tools_read_only_hides_writes() -> None:
    selection = select_tools(TOOLS, None, read_only=True)
    assert names(selection.enabled) == [
        "github_search_issues",
        "github_get_file_contents",
        "github_whoami",
    ]
    assert selection.disabled["github_create_issue"] == "write tool hidden by BENCH_READ_ONLY"


def test_select_tools_unknown_grants_keep_everything() -> None:
    assert names(select_tools(TOOLS, None).enabled) == names(tuple(TOOLS))


def test_select_tools_does_not_expand_by_default() -> None:
    # Slack: channels:write does not grant channels:read.
    tools = [ToolAccess("slack_list_channels", requires("channels:read"))]
    selection = select_tools(tools, frozenset({"channels:write"}))
    assert selection.disabled == {"slack_list_channels": "missing scope(s): channels:read"}


def test_select_tools_custom_expansion() -> None:
    drive = "https://www.googleapis.com/auth/drive"
    drive_file = "https://www.googleapis.com/auth/drive.file"
    tools = [ToolAccess("gdrive_create_document", requires(drive_file), write=True)]
    selection = select_tools(
        tools, frozenset({drive}), expand=lambda g: expand_implied(g, {drive: [drive_file]})
    )
    assert names(selection.enabled) == ["gdrive_create_document"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), (" YES ", True), ("on", True), ("0", False), ("", False)],
)
def test_read_only_from_env(value: str, expected: bool) -> None:
    assert read_only_from_env({"BENCH_READ_ONLY": value}) is expected


def test_read_only_from_env_missing() -> None:
    assert read_only_from_env({}) is False


# --- Annotations --------------------------------------------------------------------------


def test_annotations() -> None:
    read, _, write, destructive, _ = TOOLS
    assert read.annotations().read_only_hint is True
    assert read.annotations().destructive_hint is None
    assert write.annotations().read_only_hint is False
    assert write.annotations().destructive_hint is False
    assert destructive.annotations(title="Merge PR").destructive_hint is True
    assert destructive.annotations(title="Merge PR").title == "Merge PR"
    assert all(t.annotations().open_world_hint for t in TOOLS)


# --- Errors -------------------------------------------------------------------------------


def test_permission_denied_message() -> None:
    err = PermissionDenied("github", action="create issue", missing=["issues:write"])
    assert isinstance(err, ToolError)
    assert str(err) == (
        "Permission denied by github for create issue. Missing permission: issues:write. "
        "Do not retry; tell the user which permission is needed or choose another action."
    )


def test_permission_denied_without_known_scope_and_custom_hint() -> None:
    err = PermissionDenied("jira", action="transition BENCH-1", hint="Ask a project admin.")
    assert str(err) == "Permission denied by jira for transition BENCH-1. Ask a project admin."


def test_not_found_or_no_access_message() -> None:
    err = NotFoundOrNoAccess("github", "Repository o/secret")
    assert isinstance(err, ToolError)
    assert "Repository o/secret was not found on github, or the credentials have no access" in str(
        err
    )


@pytest.mark.anyio
async def test_annotations_and_errors_reach_the_mcp_client() -> None:
    server = MCPServer("perm")
    selection = select_tools(
        TOOLS, frozenset({"issues:write"}), read_only=True, expand=expand_levels
    )

    def make(access: ToolAccess):
        async def tool() -> str:
            raise PermissionDenied("github", action=access.name, missing=["issues:write"])

        return tool

    for access in selection.enabled:
        server.add_tool(make(access), name=access.name, annotations=access.annotations())

    async with Client(server) as client:
        listed = {t.name: t for t in (await client.list_tools()).tools}
        result = await client.call_tool("github_search_issues", {})

    assert set(listed) == {"github_search_issues", "github_whoami"}
    assert listed["github_search_issues"].annotations is not None
    assert listed["github_search_issues"].annotations.read_only_hint is True
    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    assert "Missing permission: issues:write" in result.content[0].text
