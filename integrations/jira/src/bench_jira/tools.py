"""Jira MCP tools: thin wrappers that shape client results for the model.

`response_format="concise"` returns what the model needs for its next step (keys,
summaries, statuses); `"detailed"` adds people, timestamps, URLs and, for one issue,
the comments. Fields that a format leaves out are omitted from the output instead of
being sent as null. Long texts are clipped with a flag that says so. Rich text (ADF)
reaches the model as plain text and is written back from plain text.

Every tool call gets one wait budget for rate-limit retries (`TOOL_WAIT_BUDGET`).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from bench_core.mcp import ResponseFormat, ToolSpec, clip, lifespan_state, omit, tool_call

from bench_jira import scopes
from bench_jira.client import PROVIDER, JiraClient
from bench_jira.client import issue_key as normalize_key
from bench_jira.models import Comment, Issue, IssueSummary

TOOL_WAIT_BUDGET = 20.0
"""Seconds one tool call may spend waiting on rate limits before it reports them."""

DESCRIPTION_LIMIT = {"concise": 1_500, "detailed": 10_000}
COMMENT_LIMIT = 1_000
MAX_COMMENTS = 20


@dataclass
class JiraState:
    client: JiraClient


@contextmanager
def _jira(ctx: Context[JiraState]) -> Iterator[JiraClient]:
    """The client for one tool call: one wait budget for all its requests."""
    client = lifespan_state(ctx).client
    with tool_call(client.api, provider=PROVIDER, wait_budget=TOOL_WAIT_BUDGET):
        yield client


# --- Output models ------------------------------------------------------------------------


class Output(BaseModel):
    pass


class IssueHit(Output):
    key: str
    summary: str
    status: str
    issue_type: str
    priority: str | None = omit()
    assignee: str | None = omit()
    labels: list[str] | None = omit()
    updated: datetime | None = omit()
    url: str | None = omit()


class IssueList(Output):
    issues: list[IssueHit]
    next_cursor: str | None = omit()


class CommentOut(Output):
    author: str | None
    body: str | None
    body_truncated: bool | None = omit()
    created: datetime


class IssueOut(Output):
    key: str
    summary: str
    status: str
    issue_type: str
    priority: str | None
    assignee: str | None
    labels: list[str]
    description: str | None
    description_truncated: bool | None = omit()
    comment_count: int
    project: str | None = omit()
    status_category: str | None = omit()
    reporter: str | None = omit()
    created: datetime | None = omit()
    updated: datetime | None = omit()
    url: str | None = omit()
    comments: list[CommentOut] | None = omit()
    more_comments: bool | None = omit()


class CreatedIssueOut(Output):
    key: str
    url: str


class CommentAdded(Output):
    issue_key: str
    comment_id: str
    url: str


class Transitioned(Output):
    key: str
    status: str
    url: str


# --- Conversions --------------------------------------------------------------------------


def _hit(issue: IssueSummary, fmt: ResponseFormat, client: JiraClient) -> IssueHit:
    hit = IssueHit(
        key=issue.key, summary=issue.summary, status=issue.status, issue_type=issue.issue_type
    )
    if fmt == "detailed":
        hit = hit.model_copy(
            update={
                "priority": issue.priority,
                "assignee": issue.assignee,
                "labels": issue.labels,
                "updated": issue.updated,
                "url": client.browse_url(issue.key),
            }
        )
    return hit


def _comment(comment: Comment) -> CommentOut:
    body, clipped = clip(comment.body, COMMENT_LIMIT)
    return CommentOut(
        author=comment.author, body=body, body_truncated=clipped, created=comment.created
    )


def _issue(issue: Issue, fmt: ResponseFormat, client: JiraClient) -> IssueOut:
    description, clipped = clip(issue.description, DESCRIPTION_LIMIT[fmt])
    out = IssueOut(
        key=issue.key,
        summary=issue.summary,
        status=issue.status,
        issue_type=issue.issue_type,
        priority=issue.priority,
        assignee=issue.assignee,
        labels=issue.labels,
        description=description,
        description_truncated=clipped,
        comment_count=max(issue.comment_total, len(issue.comments)),
    )
    if fmt == "detailed":
        shown = issue.comments[:MAX_COMMENTS]
        out = out.model_copy(
            update={
                "project": issue.project,
                "status_category": issue.status_category,
                "reporter": issue.reporter,
                "created": issue.created,
                "updated": issue.updated,
                "url": client.browse_url(issue.key),
                "comments": [_comment(c) for c in shown],
                "more_comments": True if out.comment_count > len(shown) else None,
            }
        )
    return out


# --- Tools --------------------------------------------------------------------------------

PageSize = Annotated[int, Field(ge=1, le=50)]


async def search_issues(
    jql: str,
    ctx: Context[JiraState],
    cursor: str | None = None,
    page_size: PageSize = 10,
    response_format: ResponseFormat = "concise",
) -> IssueList:
    with _jira(ctx) as client:
        page = await client.search_issues(jql, cursor=cursor, page_size=page_size)
    return IssueList(
        issues=[_hit(i, response_format, client) for i in page.items],
        next_cursor=page.next_cursor,
    )


async def get_issue(
    issue_key: str, ctx: Context[JiraState], response_format: ResponseFormat = "concise"
) -> IssueOut:
    with _jira(ctx) as client:
        issue = await client.get_issue(issue_key)
    return _issue(issue, response_format, client)


async def create_issue(
    project_key: str,
    summary: str,
    ctx: Context[JiraState],
    issue_type: str = "Task",
    description: str | None = None,
    labels: list[str] | None = None,
) -> CreatedIssueOut:
    with _jira(ctx) as client:
        created = await client.create_issue(
            project_key,
            summary,
            issue_type=issue_type,
            description=description,
            labels=labels or (),
        )
    return CreatedIssueOut(key=created.key, url=client.browse_url(created.key))


async def add_comment(issue_key: str, body: str, ctx: Context[JiraState]) -> CommentAdded:
    with _jira(ctx) as client:
        comment = await client.add_comment(issue_key, body)
    key = normalize_key(issue_key)
    return CommentAdded(
        issue_key=key,
        comment_id=comment.id,
        url=f"{client.browse_url(key)}?focusedCommentId={comment.id}",
    )


async def transition_issue(issue_key: str, status: str, ctx: Context[JiraState]) -> Transitioned:
    with _jira(ctx) as client:
        available = await client.transitions(issue_key)
        wanted = status.strip().casefold()
        match = next(
            (t for t in available if wanted in (t.to_status.casefold(), t.name.casefold())), None
        )
        if match is None:
            options = ", ".join(sorted({t.to_status for t in available})) or "none"
            raise ToolError(
                f"Issue {issue_key} cannot move to {status!r} from its current status. "
                f"Statuses it can move to now: {options}."
            )
        await client.transition_issue(issue_key, match.id)
    key = normalize_key(issue_key)
    return Transitioned(key=key, status=match.to_status, url=client.browse_url(key))


TOOLS = [
    ToolSpec(search_issues, scopes.SEARCH_ISSUES),
    ToolSpec(get_issue, scopes.GET_ISSUE),
    ToolSpec(create_issue, scopes.CREATE_ISSUE),
    ToolSpec(add_comment, scopes.ADD_COMMENT),
    ToolSpec(transition_issue, scopes.TRANSITION_ISSUE),
]
