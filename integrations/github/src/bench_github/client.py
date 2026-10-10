"""Typed client for the slice of the GitHub REST API the tools need.

Every method validates its identifiers before building a URL (no path injection from
model-supplied arguments), returns models from `bench_github.models`, and turns error
responses into `ToolError`s the model can act on: 404 -> `NotFoundOrNoAccess` (GitHub
hides private resources behind 404), 403 -> `PermissionDenied` with the permission GitHub
says it would accept, 422 -> the validation messages. Rate limits are retried by the
`ApiClient` and surface as `RateLimited` once the wait budget is used up.
"""

import base64
import binascii
import re
from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import quote

import httpx2
from mcp.server.mcpserver.exceptions import ToolError

from bench_core.auth import Credential, CredentialAuth
from bench_core.clock import Clock
from bench_core.http import ApiClient, GitHubRetryPolicy, fetch_page, link_header
from bench_core.permissions import NotFoundOrNoAccess, PermissionDenied

from bench_github.models import (
    Comment,
    DirectoryEntry,
    DirectoryListing,
    FileContent,
    Issue,
    IssueSummary,
    Page,
    PullRequestSummary,
)

PROVIDER = "GitHub"
BASE_URL = "https://api.github.com"
API_VERSION = "2022-11-28"
USER_AGENT = "integration-bench"
MAX_PER_PAGE = 100

PullState = Literal["open", "closed", "all"]


def make_api(
    *,
    credential: Credential | None = None,
    base_url: str = BASE_URL,
    transport: httpx2.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
    default_wait_budget: float = 30.0,
) -> ApiClient:
    return ApiClient(
        base_url,
        policy=GitHubRetryPolicy(name=PROVIDER),
        clock=clock,
        transport=transport,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": USER_AGENT,
        },
        auth=CredentialAuth(credential) if credential else None,
        default_wait_budget=default_wait_budget,
    )


# --- Argument validation ------------------------------------------------------------------

_OWNER = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
_REPO = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


def repo_path(owner: str, repo: str) -> str:
    if not _OWNER.match(owner):
        raise ToolError(
            f"Invalid GitHub owner {owner!r}: use the user or organization login, "
            "e.g. 'octocat' (letters, digits and hyphens)."
        )
    if not _REPO.match(repo) or repo in (".", ".."):
        raise ToolError(
            f"Invalid GitHub repository name {repo!r}: use the name only, e.g. 'hello-world', "
            "and pass the owner separately."
        )
    return f"/repos/{owner}/{repo}"


def content_path(path: str) -> str:
    """Repository file path -> URL-quoted path; '' or '/' is the repository root."""
    segments = [s for s in path.strip().strip("/").split("/") if s]
    if any(s in (".", "..") for s in segments):
        raise ToolError(f"Invalid path {path!r}: use a path relative to the repository root.")
    return "/".join(quote(s, safe="") for s in segments)


def _positive(name: str, value: int) -> int:
    if value < 1:
        raise ToolError(f"{name} must be at least 1, got {value}.")
    return value


_TYPE_QUALIFIER = re.compile(r"(?:^|\s)-?(?:is|type):(?:issue|pr|pull-request)\b", re.IGNORECASE)


def ensure_issue_qualifier(query: str) -> str:
    """GitHub's issue search wants `is:issue` or `is:pr`; default to issues."""
    query = query.strip()
    if not query:
        raise ToolError(
            "The search query is empty; pass GitHub search syntax, e.g. 'repo:o/r bug'."
        )
    return query if _TYPE_QUALIFIER.search(query) else f"{query} is:issue"


# --- Error translation --------------------------------------------------------------------


def _body(response: httpx2.Response) -> Mapping[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, Mapping) else {}


def _message(response: httpx2.Response) -> str:
    message = _body(response).get("message")
    return message if isinstance(message, str) and message else response.reason_phrase


def accepted_permissions(response: httpx2.Response) -> list[frozenset[str]]:
    """Alternatives from `X-Accepted-GitHub-Permissions` (App) or `X-Accepted-OAuth-Scopes`.

    App permissions come as `issues=write`, converted to our `issues:write` scope format.
    Parsed leniently: `;` separates alternatives, `,` permissions that are all needed.
    OAuth scopes are a comma list where any one scope is enough.
    """
    app = response.headers.get("x-accepted-github-permissions")
    if app:
        alternatives = []
        for group in app.split(";"):
            perms = frozenset(p.strip().replace("=", ":") for p in group.split(",") if p.strip())
            if perms:
                alternatives.append(perms)
        return alternatives
    oauth = response.headers.get("x-accepted-oauth-scopes")
    if oauth:
        return [frozenset({s.strip()}) for s in oauth.split(",") if s.strip()]
    return []


def _validation_details(response: httpx2.Response) -> str:
    body = _body(response)
    parts: list[str] = []
    for error in body.get("errors") or []:
        if isinstance(error, str):
            parts.append(error)
        elif isinstance(error, Mapping):
            text = error.get("message") or " ".join(
                str(error[k]) for k in ("field", "code") if error.get(k)
            )
            if text:
                parts.append(str(text))
    return "; ".join(parts) or _message(response)


def raise_for_status(response: httpx2.Response, *, action: str, what: str) -> None:
    """Raise a model-facing `ToolError` for any non-2xx response."""
    if response.is_success:
        return
    status = response.status_code
    if status == 401:
        raise ToolError(
            f"{PROVIDER} rejected the credentials while trying to {action} (401). "
            "Do not retry; the user has to sign in again."
        )
    if status == 403:
        alternatives = accepted_permissions(response)
        missing = alternatives[0] if len(alternatives) == 1 else None
        hint = None
        if len(alternatives) > 1:
            options = " or ".join(", ".join(sorted(a)) for a in alternatives)
            hint = f"One of these permissions is needed: {options}. Do not retry; tell the user."
        raise PermissionDenied(PROVIDER, action=action, missing=missing, hint=hint)
    if status == 404:
        raise NotFoundOrNoAccess(PROVIDER, what)
    if status == 410:
        raise ToolError(f"{what} is gone on {PROVIDER}: {_message(response)}.")
    if status == 409:
        raise ToolError(f"{PROVIDER} could not {action}: {_message(response)}.")
    if status == 422:
        raise ToolError(
            f"{PROVIDER} rejected the request to {action}: {_validation_details(response)}. "
            "Fix the arguments and call again."
        )
    if status >= 500:
        raise ToolError(
            f"{PROVIDER} had a server error ({status}) while trying to {action}. Try again later."
        )
    raise ToolError(f"{PROVIDER} returned {status} while trying to {action}: {_message(response)}.")


# --- Client -------------------------------------------------------------------------------


def _decode_file(data: Mapping[str, Any]) -> FileContent:
    path = data["path"]
    common = {
        "path": path,
        "sha": data["sha"],
        "size": data.get("size", 0),
        "html_url": data.get("html_url"),
    }
    kind = data.get("type")
    if kind == "symlink":
        return FileContent(**common, type="symlink", note=f"Symlink to {data.get('target')!r}.")
    if kind == "submodule":
        return FileContent(
            **common, type="submodule", note=f"Git submodule ({data.get('submodule_git_url')})."
        )
    if data.get("encoding") != "base64":
        # Files over 1 MB come back with encoding "none" and no content.
        return FileContent(**common, note="File is too large for the contents API (over 1 MB).")
    try:
        raw = base64.b64decode(data.get("content") or "")
        return FileContent(**common, text=raw.decode("utf-8"))
    except binascii.Error:
        return FileContent(**common, note="GitHub sent content that is not valid base64.")
    except UnicodeDecodeError:
        return FileContent(**common, note="Binary file; its content cannot be shown as text.")


class GitHubClient:
    def __init__(self, api: ApiClient) -> None:
        self.api = api

    async def _get_json(self, url: str, *, action: str, what: str, **kwargs: Any) -> Any:
        response = await self.api.get(url, **kwargs)
        raise_for_status(response, action=action, what=what)
        return response.json()

    async def search_issues(
        self, query: str, *, cursor: str | None = None, per_page: int = 20
    ) -> Page[IssueSummary]:
        params = {
            "q": ensure_issue_qualifier(query),
            "per_page": min(_positive("per_page", per_page), MAX_PER_PAGE),
        }
        page = await fetch_page(
            self.api, "/search/issues", link_header(), cursor=cursor, params=params
        )
        raise_for_status(page.response, action="search issues", what="The search")
        data = page.response.json()
        return Page[IssueSummary](
            items=[IssueSummary.model_validate(item) for item in data.get("items", [])],
            next_cursor=page.next_cursor,
            total_count=data.get("total_count"),
            incomplete_results=bool(data.get("incomplete_results")),
        )

    async def get_issue(self, owner: str, repo: str, number: int) -> Issue:
        url = f"{repo_path(owner, repo)}/issues/{_positive('number', number)}"
        data = await self._get_json(
            url, action="read the issue", what=f"Issue {owner}/{repo}#{number}"
        )
        return Issue.model_validate(data)

    async def list_issue_comments(
        self, owner: str, repo: str, number: int, *, cursor: str | None = None, per_page: int = 30
    ) -> Page[Comment]:
        url = f"{repo_path(owner, repo)}/issues/{_positive('number', number)}/comments"
        params = {"per_page": min(_positive("per_page", per_page), MAX_PER_PAGE)}
        page = await fetch_page(self.api, url, link_header(), cursor=cursor, params=params)
        raise_for_status(
            page.response, action="list issue comments", what=f"Issue {owner}/{repo}#{number}"
        )
        return Page[Comment](
            items=[Comment.model_validate(c) for c in page.response.json()],
            next_cursor=page.next_cursor,
        )

    async def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        *,
        body: str | None = None,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> Issue:
        if not title.strip():
            raise ToolError("The issue title must not be empty.")
        payload: dict[str, Any] = {"title": title}
        if body:
            payload["body"] = body
        if labels:
            payload["labels"] = labels
        if assignees:
            payload["assignees"] = assignees
        response = await self.api.post(f"{repo_path(owner, repo)}/issues", json=payload)
        raise_for_status(response, action="create an issue", what=f"Repository {owner}/{repo}")
        return Issue.model_validate(response.json())

    async def add_issue_comment(self, owner: str, repo: str, number: int, body: str) -> Comment:
        if not body.strip():
            raise ToolError("The comment body must not be empty.")
        url = f"{repo_path(owner, repo)}/issues/{_positive('number', number)}/comments"
        response = await self.api.post(url, json={"body": body})
        raise_for_status(
            response, action="comment on the issue", what=f"Issue {owner}/{repo}#{number}"
        )
        return Comment.model_validate(response.json())

    async def list_pull_requests(
        self,
        owner: str,
        repo: str,
        *,
        state: PullState = "open",
        cursor: str | None = None,
        per_page: int = 20,
    ) -> Page[PullRequestSummary]:
        params = {
            "state": state,
            "sort": "updated",
            "direction": "desc",
            "per_page": min(_positive("per_page", per_page), MAX_PER_PAGE),
        }
        url = f"{repo_path(owner, repo)}/pulls"
        page = await fetch_page(self.api, url, link_header(), cursor=cursor, params=params)
        raise_for_status(
            page.response, action="list pull requests", what=f"Repository {owner}/{repo}"
        )
        return Page[PullRequestSummary](
            items=[PullRequestSummary.model_validate(pr) for pr in page.response.json()],
            next_cursor=page.next_cursor,
        )

    async def get_file_contents(
        self, owner: str, repo: str, path: str = "", *, ref: str | None = None
    ) -> FileContent | DirectoryListing:
        """A file's text, or the listing if `path` is a directory ('' = repository root)."""
        quoted = content_path(path)
        url = (
            f"{repo_path(owner, repo)}/contents/{quoted}"
            if quoted
            else f"{repo_path(owner, repo)}/contents"
        )
        location = f"{owner}/{repo}/{path.strip('/')}" + (f" at {ref!r}" if ref else "")
        kwargs: dict[str, Any] = {"params": {"ref": ref}} if ref else {}
        data = await self._get_json(
            url, action="read repository contents", what=f"Path {location}", **kwargs
        )
        if isinstance(data, list):
            return DirectoryListing(
                path=path.strip("/"), entries=[DirectoryEntry.model_validate(e) for e in data]
            )
        return _decode_file(data)
