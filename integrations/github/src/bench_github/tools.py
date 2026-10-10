"""GitHub MCP tools: thin wrappers that shape client results for the model.

`response_format="concise"` returns what the model needs for its next step (numbers,
titles, states, a clipped body); `"detailed"` adds authors, timestamps, URLs and, for
issues, the comments. Fields that a format leaves out are omitted from the output
instead of being sent as null. Long texts are clipped with a flag that says so.

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

from bench_github import scopes
from bench_github.client import GitHubClient, PullState
from bench_github.models import (
    Comment,
    DirectoryListing,
    Issue,
    IssueSummary,
    PullRequestSummary,
)

TOOL_WAIT_BUDGET = 20.0
"""Seconds one tool call may spend waiting on rate limits before it reports them."""

BODY_LIMIT = {"concise": 1_500, "detailed": 10_000}
COMMENT_LIMIT = 1_000
FILE_LIMIT = {"concise": 4_000, "detailed": 40_000}
MAX_COMMENTS = 20


@dataclass
class GitHubState:
    gh: GitHubClient


class Output(BaseModel):
    pass


@contextmanager
def _github(ctx: Context[GitHubState]) -> Iterator[GitHubClient]:
    """The client for one tool call: one wait budget for all its requests."""
    gh = lifespan_state(ctx).gh
    with tool_call(gh.api, provider="GitHub", wait_budget=TOOL_WAIT_BUDGET):
        yield gh


# --- Output models ------------------------------------------------------------------------


class IssueHit(Output):
    repository: str
    number: int
    title: str
    state: str
    labels: list[str]
    is_pull_request: bool | None = omit()
    author: str | None = omit()
    comments: int | None = omit()
    created_at: datetime | None = omit()
    updated_at: datetime | None = omit()
    html_url: str | None = omit()


class SearchResult(Output):
    total_count: int
    items: list[IssueHit]
    next_cursor: str | None = omit()
    incomplete_results: bool | None = omit()


class CommentOut(Output):
    author: str | None
    body: str
    body_truncated: bool | None = omit()
    created_at: datetime
    html_url: str | None = omit()


class IssueOut(Output):
    repository: str
    number: int
    title: str
    state: str
    labels: list[str]
    assignees: list[str]
    body: str | None
    body_truncated: bool | None = omit()
    comments: int
    is_pull_request: bool | None = omit()
    author: str | None = omit()
    state_reason: str | None = omit()
    milestone: str | None = omit()
    created_at: datetime | None = omit()
    updated_at: datetime | None = omit()
    closed_at: datetime | None = omit()
    html_url: str | None = omit()
    comment_list: list[CommentOut] | None = omit()
    more_comments: bool | None = omit()


class CreatedIssue(Output):
    repository: str
    number: int
    title: str
    state: str
    labels: list[str]
    assignees: list[str]
    html_url: str


class CreatedComment(Output):
    id: int
    issue_number: int
    html_url: str


class PullRequestOut(Output):
    number: int
    title: str
    state: str
    draft: bool
    author: str | None
    head: str
    base: str
    updated_at: datetime
    merged_at: datetime | None = omit()
    created_at: datetime | None = omit()
    labels: list[str] | None = omit()
    html_url: str | None = omit()


class PullRequestList(Output):
    items: list[PullRequestOut]
    next_cursor: str | None = omit()


class EntryOut(Output):
    name: str
    type: str
    size: int | None = omit()
    path: str | None = omit()


class FileOut(Output):
    path: str
    type: str
    """"file", "dir", "symlink" or "submodule"."""
    text: str | None = omit()
    text_truncated: bool | None = omit()
    note: str | None = omit()
    entries: list[EntryOut] | None = omit()
    size: int | None = omit()
    sha: str | None = omit()
    html_url: str | None = omit()


# --- Conversions --------------------------------------------------------------------------


def _hit(issue: IssueSummary, fmt: ResponseFormat) -> IssueHit:
    hit = IssueHit(
        repository=issue.repository,
        number=issue.number,
        title=issue.title,
        state=issue.state,
        labels=issue.labels,
        # Only flag pull requests; plain issues stay compact.
        is_pull_request=True if issue.is_pull_request else None,
    )
    if fmt == "detailed":
        hit = hit.model_copy(
            update={
                "is_pull_request": issue.is_pull_request,
                "author": issue.author,
                "comments": issue.comments,
                "created_at": issue.created_at,
                "updated_at": issue.updated_at,
                "html_url": issue.html_url,
            }
        )
    return hit


def _comment(comment: Comment, fmt: ResponseFormat) -> CommentOut:
    body, clipped = clip(comment.body, COMMENT_LIMIT)
    return CommentOut(
        author=comment.author,
        body=body or "",
        body_truncated=clipped,
        created_at=comment.created_at,
        html_url=comment.html_url if fmt == "detailed" else None,
    )


def _issue(
    issue: Issue, fmt: ResponseFormat, comments: list[Comment] | None, more: bool
) -> IssueOut:
    body, clipped = clip(issue.body, BODY_LIMIT[fmt])
    out = IssueOut(
        repository=issue.repository,
        number=issue.number,
        title=issue.title,
        state=issue.state,
        labels=issue.labels,
        assignees=issue.assignees,
        body=body,
        body_truncated=clipped,
        comments=issue.comments,
        is_pull_request=True if issue.is_pull_request else None,
    )
    if fmt == "detailed":
        out = out.model_copy(
            update={
                "author": issue.author,
                "state_reason": issue.state_reason,
                "milestone": issue.milestone,
                "created_at": issue.created_at,
                "updated_at": issue.updated_at,
                "closed_at": issue.closed_at,
                "html_url": issue.html_url,
                "comment_list": [_comment(c, fmt) for c in comments or []],
                "more_comments": True if more else None,
            }
        )
    return out


def _pull(pr: PullRequestSummary, fmt: ResponseFormat) -> PullRequestOut:
    out = PullRequestOut(
        number=pr.number,
        title=pr.title,
        state=pr.state,
        draft=pr.draft,
        author=pr.author,
        head=pr.head,
        base=pr.base,
        updated_at=pr.updated_at,
        merged_at=pr.merged_at,
    )
    if fmt == "detailed":
        out = out.model_copy(
            update={"created_at": pr.created_at, "labels": pr.labels, "html_url": pr.html_url}
        )
    return out


# --- Tools --------------------------------------------------------------------------------

PerPage = Annotated[int, Field(ge=1, le=50)]
Number = Annotated[int, Field(ge=1)]


async def search_issues(
    query: str,
    ctx: Context[GitHubState],
    cursor: str | None = None,
    per_page: PerPage = 10,
    response_format: ResponseFormat = "concise",
) -> SearchResult:
    with _github(ctx) as gh:
        page = await gh.search_issues(query, cursor=cursor, per_page=per_page)
    return SearchResult(
        total_count=page.total_count or 0,
        items=[_hit(i, response_format) for i in page.items],
        next_cursor=page.next_cursor,
        incomplete_results=True if page.incomplete_results else None,
    )


async def get_issue(
    owner: str,
    repo: str,
    number: Number,
    ctx: Context[GitHubState],
    response_format: ResponseFormat = "concise",
) -> IssueOut:
    comments: list[Comment] | None = None
    more = False
    with _github(ctx) as gh:
        issue = await gh.get_issue(owner, repo, number)
        if response_format == "detailed" and issue.comments:
            page = await gh.list_issue_comments(owner, repo, number, per_page=MAX_COMMENTS)
            comments, more = page.items, page.next_cursor is not None
    return _issue(issue, response_format, comments, more)


async def create_issue(
    owner: str,
    repo: str,
    title: str,
    ctx: Context[GitHubState],
    body: str | None = None,
    labels: list[str] | None = None,
    assignees: list[str] | None = None,
) -> CreatedIssue:
    with _github(ctx) as gh:
        issue = await gh.create_issue(
            owner, repo, title, body=body, labels=labels, assignees=assignees
        )
    return CreatedIssue(
        repository=issue.repository,
        number=issue.number,
        title=issue.title,
        state=issue.state,
        labels=issue.labels,
        assignees=issue.assignees,
        html_url=issue.html_url,
    )


async def add_issue_comment(
    owner: str, repo: str, number: Number, body: str, ctx: Context[GitHubState]
) -> CreatedComment:
    with _github(ctx) as gh:
        comment = await gh.add_issue_comment(owner, repo, number, body)
    return CreatedComment(id=comment.id, issue_number=number, html_url=comment.html_url)


async def list_pull_requests(
    owner: str,
    repo: str,
    ctx: Context[GitHubState],
    state: PullState = "open",
    cursor: str | None = None,
    per_page: PerPage = 10,
    response_format: ResponseFormat = "concise",
) -> PullRequestList:
    with _github(ctx) as gh:
        page = await gh.list_pull_requests(
            owner, repo, state=state, cursor=cursor, per_page=per_page
        )
    return PullRequestList(
        items=[_pull(pr, response_format) for pr in page.items], next_cursor=page.next_cursor
    )


async def get_file_contents(
    owner: str,
    repo: str,
    ctx: Context[GitHubState],
    path: str = "",
    ref: str | None = None,
    response_format: ResponseFormat = "concise",
) -> FileOut:
    with _github(ctx) as gh:
        result = await gh.get_file_contents(owner, repo, path, ref=ref)
    detailed = response_format == "detailed"
    if isinstance(result, DirectoryListing):
        return FileOut(
            path=result.path,
            type="dir",
            entries=[
                EntryOut(
                    name=e.name,
                    type=e.type,
                    size=e.size if detailed and e.type == "file" else None,
                    path=e.path if detailed else None,
                )
                for e in result.entries
            ],
        )
    text, clipped = clip(result.text, FILE_LIMIT[response_format])
    note = result.note
    if clipped:
        note = f"Showing the first {len(text or '')} of {len(result.text or '')} characters." + (
            "" if detailed else " Use response_format='detailed' for more."
        )
    return FileOut(
        path=result.path,
        type=result.type,
        text=text,
        text_truncated=clipped,
        note=note,
        size=result.size if detailed or text is None else None,
        sha=result.sha if detailed else None,
        html_url=result.html_url if detailed else None,
    )


TOOLS = [
    ToolSpec(search_issues, scopes.SEARCH_ISSUES),
    ToolSpec(get_issue, scopes.GET_ISSUE),
    ToolSpec(create_issue, scopes.CREATE_ISSUE),
    ToolSpec(add_issue_comment, scopes.ADD_ISSUE_COMMENT),
    ToolSpec(list_pull_requests, scopes.LIST_PULL_REQUESTS),
    ToolSpec(get_file_contents, scopes.GET_FILE_CONTENTS),
]
