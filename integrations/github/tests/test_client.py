"""GitHub client and models against an in-memory backend (cassettes follow in step 12)."""

import base64
from typing import Any

import httpx2
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from bench_core.auth import StaticToken
from bench_core.http import decode_cursor
from bench_core.permissions import NotFoundOrNoAccess, PermissionDenied
from bench_core.testing import FakeBackend, FakeClock, FakeRequest

from bench_github.client import (
    GitHubClient,
    accepted_permissions,
    content_path,
    ensure_issue_qualifier,
    make_api,
)
from bench_github.models import (
    DirectoryListing,
    FileContent,
    Issue,
    IssueSummary,
    PullRequestSummary,
)

pytestmark = pytest.mark.anyio

API = "https://api.github.com"


def raw_issue(number: int = 7, **overrides: Any) -> dict[str, Any]:
    issue = {
        "url": f"{API}/repos/acme/widgets/issues/{number}",
        "repository_url": f"{API}/repos/acme/widgets",
        "html_url": f"https://github.com/acme/widgets/issues/{number}",
        "id": 1000 + number,
        "number": number,
        "title": f"Issue {number}",
        "state": "open",
        "state_reason": None,
        "user": {"login": "octocat", "id": 1, "type": "User"},
        "labels": [{"id": 1, "name": "bug", "color": "d73a4a"}],
        "assignees": [{"login": "hubot", "id": 2}],
        "milestone": None,
        "comments": 2,
        "created_at": "2026-10-01T09:00:00Z",
        "updated_at": "2026-10-02T09:00:00Z",
        "closed_at": None,
        "body": "Steps to reproduce ...",
        "reactions": {"total_count": 0},
    }
    issue.update(overrides)
    return issue


def raw_pull(number: int = 3) -> dict[str, Any]:
    return {
        "number": number,
        "title": "Add widget",
        "state": "open",
        "draft": True,
        "user": {"login": "octocat"},
        "head": {"ref": "feature/widget", "sha": "abc"},
        "base": {"ref": "main", "sha": "def"},
        "labels": [],
        "created_at": "2026-10-03T09:00:00Z",
        "updated_at": "2026-10-04T09:00:00Z",
        "merged_at": None,
        "html_url": f"https://github.com/acme/widgets/pull/{number}",
    }


def raw_file(path: str, blob: bytes | None, **overrides: Any) -> dict[str, Any]:
    encoded = base64.b64encode(blob or b"").decode()
    data = {
        "type": "file",
        "name": path.rsplit("/", 1)[-1],
        "path": path,
        "sha": "f00",
        "size": len(blob or b""),
        "encoding": "base64",
        # GitHub wraps base64 at 60 characters.
        "content": "\n".join(encoded[i : i + 60] for i in range(0, len(encoded), 60)),
        "html_url": f"https://github.com/acme/widgets/blob/main/{path}",
    }
    data.update(overrides)
    return data


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend(require_auth=False)


@pytest.fixture
async def client(backend: FakeBackend):
    api = make_api(transport=backend.transport(), clock=FakeClock())
    async with api:
        yield GitHubClient(api)


# --- Models -------------------------------------------------------------------------------


def test_issue_summary_flattens_github_json() -> None:
    summary = IssueSummary.model_validate(raw_issue())
    assert summary.repository == "acme/widgets"
    assert summary.author == "octocat"
    assert summary.labels == ["bug"]
    assert summary.is_pull_request is False
    dumped = summary.model_dump(mode="json")
    assert dumped["repository"] == "acme/widgets"
    assert "user" not in dumped
    assert "reactions" not in dumped


def test_issue_summary_marks_pull_requests_and_deleted_users() -> None:
    hit = raw_issue(user=None, pull_request={"url": "...", "merged_at": None})
    summary = IssueSummary.model_validate(hit)
    assert summary.is_pull_request is True
    assert summary.author is None


def test_issue_detail_fields() -> None:
    issue = Issue.model_validate(raw_issue(milestone={"title": "v1.0", "number": 1}))
    assert issue.assignees == ["hubot"]
    assert issue.milestone == "v1.0"
    assert issue.body == "Steps to reproduce ..."
    assert Issue.model_validate(raw_issue()).milestone is None


def test_pull_request_summary_uses_branch_names() -> None:
    pr = PullRequestSummary.model_validate(raw_pull())
    assert (pr.head, pr.base, pr.draft) == ("feature/widget", "main", True)


# --- Request basics -----------------------------------------------------------------------


async def test_sends_github_headers_and_token(backend: FakeBackend) -> None:
    backend.route("GET", "/repos/{owner}/{repo}/issues/{number}")(lambda r: raw_issue())
    credential = StaticToken("github", "ghu_test")
    async with make_api(credential=credential, transport=backend.transport()) as api:
        await GitHubClient(api).get_issue("acme", "widgets", 7)
    request = backend.requests[0]
    assert request.headers["accept"] == "application/vnd.github+json"
    assert request.headers["x-github-api-version"] == "2022-11-28"
    assert request.headers["authorization"] == "Bearer ghu_test"
    assert request.url.host == "api.github.com"


@pytest.mark.parametrize(
    ("owner", "repo"),
    [("acme/evil", "widgets"), ("acme", "../widgets"), ("acme", ".."), ("", "widgets")],
)
async def test_rejects_bad_identifiers_without_a_request(
    client: GitHubClient, backend: FakeBackend, owner: str, repo: str
) -> None:
    with pytest.raises(ToolError, match="Invalid GitHub"):
        await client.get_issue(owner, repo, 1)
    assert backend.requests == []


async def test_rejects_non_positive_numbers(client: GitHubClient) -> None:
    with pytest.raises(ToolError, match="number must be at least 1"):
        await client.get_issue("acme", "widgets", 0)


# --- Search -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("repo:acme/widgets crash", "repo:acme/widgets crash is:issue"),
        ("is:pr author:octocat", "is:pr author:octocat"),
        ("crash type:issue", "crash type:issue"),
        ("crash -is:pull-request", "crash -is:pull-request"),
        ("  label:bug  ", "label:bug is:issue"),
    ],
)
def test_issue_qualifier(query: str, expected: str) -> None:
    assert ensure_issue_qualifier(query) == expected


def test_empty_query_is_rejected() -> None:
    with pytest.raises(ToolError, match="empty"):
        ensure_issue_qualifier("   ")


async def test_search_pages_with_link_header(client: GitHubClient, backend: FakeBackend) -> None:
    @backend.route("GET", "/search/issues")
    def search(r: FakeRequest) -> httpx2.Response:
        page = int(r.query.get("page", "1"))
        headers = {}
        if page == 1:
            nxt = f"{API}/search/issues?q=bug+is%3Aissue&per_page=2&page=2"
            headers["Link"] = f'<{nxt}>; rel="next", <{nxt}>; rel="last"'
        body = {
            "total_count": 3,
            "incomplete_results": False,
            "items": [raw_issue(1), raw_issue(2)] if page == 1 else [raw_issue(3)],
        }
        return httpx2.Response(200, headers=headers, json=body)

    first = await client.search_issues("bug", per_page=2)
    assert [i.number for i in first.items] == [1, 2]
    assert first.total_count == 3
    assert first.next_cursor is not None
    assert backend.requests[0].url.params["q"] == "bug is:issue"

    second = await client.search_issues("bug", cursor=first.next_cursor, per_page=2)
    assert [i.number for i in second.items] == [3]
    assert second.next_cursor is None
    assert backend.requests[1].url.params["page"] == "2"


async def test_search_caps_page_size(client: GitHubClient, backend: FakeBackend) -> None:
    backend.route("GET", "/search/issues")(lambda r: {"total_count": 0, "items": []})
    await client.search_issues("bug", per_page=500)
    assert backend.requests[0].url.params["per_page"] == "100"


async def test_search_validation_error_is_actionable(
    client: GitHubClient, backend: FakeBackend
) -> None:
    body = {
        "message": "Validation Failed",
        "errors": [
            {
                "message": "The listed users cannot be searched.",
                "resource": "Search",
                "field": "q",
                "code": "invalid",
            }
        ],
    }
    backend.route("GET", "/search/issues")(lambda r: httpx2.Response(422, json=body))
    with pytest.raises(ToolError, match=r"listed users cannot be searched.*Fix the arguments"):
        await client.search_issues("author:nobody-xyz")


async def test_foreign_cursor_is_rejected(client: GitHubClient) -> None:
    with pytest.raises(ToolError, match="Invalid cursor"):
        await client.search_issues("bug", cursor="not-a-cursor")


# --- Issues and comments ------------------------------------------------------------------


async def test_get_issue(client: GitHubClient, backend: FakeBackend) -> None:
    backend.route("GET", "/repos/{owner}/{repo}/issues/{number}")(
        lambda r: raw_issue(int(r.params["number"]))
    )
    issue = await client.get_issue("acme", "widgets", 42)
    assert (issue.number, issue.repository, issue.labels) == (42, "acme/widgets", ["bug"])


async def test_missing_issue_is_not_found_or_no_access(client: GitHubClient) -> None:
    with pytest.raises(NotFoundOrNoAccess, match=r"Issue acme/widgets#9 was not found"):
        await client.get_issue("acme", "widgets", 9)


async def test_list_issue_comments(client: GitHubClient, backend: FakeBackend) -> None:
    comment = {
        "id": 11,
        "user": {"login": "hubot"},
        "body": "Confirmed.",
        "created_at": "2026-10-05T09:00:00Z",
        "updated_at": "2026-10-05T09:00:00Z",
        "html_url": "https://github.com/acme/widgets/issues/7#issuecomment-11",
    }
    backend.route("GET", "/repos/{owner}/{repo}/issues/{number}/comments")(lambda r: [comment])
    page = await client.list_issue_comments("acme", "widgets", 7)
    assert [(c.author, c.body) for c in page.items] == [("hubot", "Confirmed.")]
    assert page.next_cursor is None


async def test_create_issue_sends_only_given_fields(
    client: GitHubClient, backend: FakeBackend
) -> None:
    @backend.route("POST", "/repos/{owner}/{repo}/issues")
    def create(r: FakeRequest) -> httpx2.Response:
        payload = r.json()
        return httpx2.Response(201, json=raw_issue(8, title=payload["title"], body=None))

    issue = await client.create_issue("acme", "widgets", "Crash on start", labels=["bug"])
    assert issue.title == "Crash on start"
    assert FakeRequest(backend.requests[0], {}, None).json() == {
        "title": "Crash on start",
        "labels": ["bug"],
    }


async def test_create_issue_needs_a_title(client: GitHubClient, backend: FakeBackend) -> None:
    with pytest.raises(ToolError, match="title must not be empty"):
        await client.create_issue("acme", "widgets", "  ")
    assert backend.requests == []


async def test_add_issue_comment(client: GitHubClient, backend: FakeBackend) -> None:
    @backend.route("POST", "/repos/{owner}/{repo}/issues/{number}/comments")
    def comment(r: FakeRequest) -> httpx2.Response:
        return httpx2.Response(
            201,
            json={
                "id": 12,
                "user": {"login": "bench-bot[bot]"},
                "body": r.json()["body"],
                "created_at": "2026-10-06T09:00:00Z",
                "updated_at": "2026-10-06T09:00:00Z",
                "html_url": "https://github.com/acme/widgets/issues/7#issuecomment-12",
            },
        )

    created = await client.add_issue_comment("acme", "widgets", 7, "Looking into it.")
    assert (created.id, created.author, created.body) == (12, "bench-bot[bot]", "Looking into it.")


# --- Pull requests ------------------------------------------------------------------------


async def test_list_pull_requests(client: GitHubClient, backend: FakeBackend) -> None:
    @backend.route("GET", "/repos/{owner}/{repo}/pulls")
    def pulls(r: FakeRequest) -> httpx2.Response:
        nxt = f"{API}/repositories/99/pulls?state=all&page=2"
        return httpx2.Response(200, headers={"Link": f'<{nxt}>; rel="next"'}, json=[raw_pull()])

    page = await client.list_pull_requests("acme", "widgets", state="all", per_page=1)
    params = backend.requests[0].url.params
    assert (params["state"], params["sort"], params["per_page"]) == ("all", "updated", "1")
    assert [pr.number for pr in page.items] == [3]
    # The cursor keeps path + query only, so it can never point at another host.
    assert page.next_cursor is not None
    assert (
        decode_cursor("link_header", page.next_cursor) == "/repositories/99/pulls?state=all&page=2"
    )


# --- Contents -----------------------------------------------------------------------------


async def test_file_contents_are_decoded(client: GitHubClient, backend: FakeBackend) -> None:
    text = "# Widgets\n\nÄrger-frei.\n" * 10
    backend.route("GET", "/repos/{owner}/{repo}/contents/docs/{name}")(
        lambda r: raw_file("docs/READ ME.md", text.encode())
    )
    result = await client.get_file_contents("acme", "widgets", "/docs/READ ME.md", ref="v1")
    assert isinstance(result, FileContent)
    assert result.text == text
    request = backend.requests[0]
    assert request.url.raw_path.decode().startswith(
        "/repos/acme/widgets/contents/docs/READ%20ME.md"
    )
    assert request.url.params["ref"] == "v1"


@pytest.mark.parametrize(
    ("data", "note"),
    [
        (raw_file("logo.png", b"\x89PNG\r\n\x1a\n\xff\xfe"), "Binary file"),
        (raw_file("big.csv", None, encoding="none", content="", size=5_000_000), "too large"),
        (raw_file("link", None, type="symlink", target="docs/README.md"), "Symlink to"),
    ],
)
async def test_file_contents_without_text(
    client: GitHubClient, backend: FakeBackend, data: dict[str, Any], note: str
) -> None:
    backend.route("GET", "/repos/{owner}/{repo}/contents/{path}")(lambda r: data)
    result = await client.get_file_contents("acme", "widgets", data["path"])
    assert isinstance(result, FileContent)
    assert result.text is None
    assert result.note is not None
    assert note in result.note


async def test_directory_listing_at_root(client: GitHubClient, backend: FakeBackend) -> None:
    entries = [
        {"name": "README.md", "path": "README.md", "type": "file", "size": 120, "sha": "a"},
        {"name": "src", "path": "src", "type": "dir", "size": 0, "sha": "b"},
    ]
    backend.route("GET", "/repos/{owner}/{repo}/contents")(lambda r: entries)
    result = await client.get_file_contents("acme", "widgets")
    assert isinstance(result, DirectoryListing)
    assert [(e.name, e.type) for e in result.entries] == [("README.md", "file"), ("src", "dir")]


def test_content_path_rejects_traversal() -> None:
    assert content_path("/a b/c.md/") == "a%20b/c.md"
    assert content_path("") == ""
    with pytest.raises(ToolError, match="Invalid path"):
        content_path("docs/../../secrets")


# --- Error translation --------------------------------------------------------------------


async def test_403_names_the_missing_app_permission(
    client: GitHubClient, backend: FakeBackend
) -> None:
    backend.route("POST", "/repos/{owner}/{repo}/issues")(
        lambda r: httpx2.Response(
            403,
            headers={"X-Accepted-GitHub-Permissions": "issues=write"},
            json={"message": "Resource not accessible by integration"},
        )
    )
    with pytest.raises(PermissionDenied, match="Missing permission: issues:write") as exc:
        await client.create_issue("acme", "widgets", "x")
    assert exc.value.missing == frozenset({"issues:write"})


def test_accepted_permissions_parsing() -> None:
    def parse(headers: dict[str, str]) -> list[frozenset[str]]:
        return accepted_permissions(httpx2.Response(403, headers=headers))

    assert parse(
        {"X-Accepted-GitHub-Permissions": "contents=read, metadata=read; issues=write"}
    ) == [
        frozenset({"contents:read", "metadata:read"}),
        frozenset({"issues:write"}),
    ]
    assert parse({"X-Accepted-OAuth-Scopes": "repo, public_repo"}) == [
        frozenset({"repo"}),
        frozenset({"public_repo"}),
    ]
    assert parse({}) == []


async def test_403_with_alternatives_lists_them(client: GitHubClient, backend: FakeBackend) -> None:
    backend.route("GET", "/repos/{owner}/{repo}/pulls")(
        lambda r: httpx2.Response(
            403,
            headers={"X-Accepted-OAuth-Scopes": "repo, public_repo"},
            json={"message": "Must have push access"},
        )
    )
    with pytest.raises(PermissionDenied, match=r"needed: repo or public_repo. Do not retry"):
        await client.list_pull_requests("acme", "widgets")


async def test_401_tells_the_model_not_to_retry(client: GitHubClient, backend: FakeBackend) -> None:
    backend.route("GET", "/search/issues")(
        lambda r: httpx2.Response(401, json={"message": "Bad credentials"})
    )
    with pytest.raises(ToolError, match=r"rejected the credentials.*\(401\)"):
        await client.search_issues("bug")


async def test_server_error_is_not_retried_and_reported(
    client: GitHubClient, backend: FakeBackend
) -> None:
    backend.route("GET", "/repos/{owner}/{repo}/issues/{number}")(
        lambda r: httpx2.Response(502, json={"message": "Server Error"})
    )
    with pytest.raises(ToolError, match=r"server error \(502\).*Try again later"):
        await client.get_issue("acme", "widgets", 1)
    assert len(backend.requests) == 1


async def test_gone_issue(client: GitHubClient, backend: FakeBackend) -> None:
    backend.route("GET", "/repos/{owner}/{repo}/issues/{number}")(
        lambda r: httpx2.Response(410, json={"message": "Issues are disabled for this repo"})
    )
    with pytest.raises(ToolError, match="is gone on GitHub: Issues are disabled"):
        await client.get_issue("acme", "widgets", 1)
