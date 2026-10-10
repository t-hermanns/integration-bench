"""Typed client for the slice of the Jira Cloud REST API (v3) the tools need.

Every method validates its identifiers before building a URL (no path injection from
model-supplied arguments), returns models from `bench_jira.models`, and turns error
responses into `ToolError`s the model can act on: 401 -> sign in again, missing OAuth
scope or 403 -> `PermissionDenied`, 404 -> `NotFoundOrNoAccess`, 400 -> Jira's own
messages (JQL errors included). Rate limits are retried by the `ApiClient` and surface
as `RateLimited` once the wait budget is used up.

Two ways in, one API: an API token talks to the site (`site_api_url`), an OAuth 2.0
(3LO) user token to the Atlassian gateway with the site's cloud id (`gateway_api_url`).
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

import httpx2
from mcp.server.mcpserver.exceptions import ToolError

from bench_core.auth import Credential, CredentialAuth
from bench_core.clock import Clock
from bench_core.http import ApiClient, JiraRetryPolicy, fetch_page, next_page_token
from bench_core.permissions import NotFoundOrNoAccess, PermissionDenied

from bench_jira.adf import text_to_adf
from bench_jira.models import Comment, CreatedIssue, Issue, IssueSummary, Page, Transition

PROVIDER = "Jira"
SITE_URL = "https://bench-sandbox.atlassian.net"
"""The sandbox site; browse links point here."""
BASE_URL = "https://bench-sandbox.atlassian.net/rest/api/3"
GATEWAY_URL = "https://api.atlassian.com/ex/jira"
USER_AGENT = "integration-bench"
MAX_PAGE_SIZE = 100
PAGE_SIZE_PARAM = "maxResults"
PAGINATION = next_page_token()

SEARCH_FIELDS = "summary,status,issuetype,priority,assignee,labels,created,updated"
ISSUE_FIELDS = SEARCH_FIELDS + ",project,reporter,description,comment"
MAX_SUMMARY = 255


def site_api_url(site: str) -> str:
    """REST base for a site, e.g. "bench-sandbox" -> https://bench-sandbox.atlassian.net/..."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", site):
        raise ValueError(f"Invalid Atlassian site name {site!r}, e.g. 'bench-sandbox'.")
    return f"https://{site}.atlassian.net/rest/api/3"


def gateway_api_url(cloud_id: str) -> str:
    """REST base for OAuth 2.0 (3LO) tokens: the gateway plus the site's cloud id."""
    if not re.fullmatch(r"[0-9a-f-]{36}", cloud_id):
        raise ValueError(f"Invalid cloud id {cloud_id!r}: expected a UUID.")
    return f"{GATEWAY_URL}/{cloud_id}/rest/api/3"


def is_scope_mismatch(response: httpx2.Response) -> bool:
    """Jira answers a call outside the token's OAuth scopes with 401, not 403."""
    return response.status_code == 401 and "scope does not match" in response.text.lower()


class JiraAuth(CredentialAuth):
    """`CredentialAuth`, but a scope mismatch is final: a new token has the same scopes."""

    async def async_auth_flow(self, request: httpx2.Request):  # type: ignore[override]
        header = await self.credential.authorization()
        request.headers["Authorization"] = header
        response = yield request
        if response.status_code != 401:
            return
        await response.aread()
        if not is_scope_mismatch(response):
            request.headers["Authorization"] = await self.credential.authorization(failed=header)
            yield request


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
        policy=JiraRetryPolicy(name=PROVIDER),
        clock=clock,
        transport=transport,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        auth=JiraAuth(credential) if credential else None,
        default_wait_budget=default_wait_budget,
    )


# --- Argument validation ------------------------------------------------------------------

_ISSUE_KEY = re.compile(r"^[A-Z][A-Z0-9_]{1,9}-[1-9][0-9]{0,8}$")
_ISSUE_ID = re.compile(r"^[1-9][0-9]{0,17}$")
_PROJECT_KEY = re.compile(r"^[A-Z][A-Z0-9_]{1,9}$")


def issue_key(value: str) -> str:
    """An issue key ("BENCH-12", any case) or numeric id, normalized for the URL."""
    key = value.strip().upper()
    if _ISSUE_KEY.match(key) or _ISSUE_ID.match(key):
        return key
    raise ToolError(
        f"Invalid issue key {value!r}: use the key from a search result, e.g. 'BENCH-12'."
    )


def project_key(value: str) -> str:
    key = value.strip().upper()
    if not _PROJECT_KEY.match(key):
        raise ToolError(
            f"Invalid project key {value!r}: use the short key in front of issue numbers, "
            "e.g. 'BENCH' for BENCH-12."
        )
    return key


def _positive(name: str, value: int) -> int:
    if value < 1:
        raise ToolError(f"{name} must be at least 1, got {value}.")
    return value


# --- Error translation --------------------------------------------------------------------


def _body(response: httpx2.Response) -> Mapping[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, Mapping) else {}


def error_text(response: httpx2.Response) -> str:
    """`{"errorMessages": ["..."], "errors": {"field": "..."}}`, or the gateway's `message`."""
    body = _body(response)
    parts = [str(m) for m in body.get("errorMessages") or [] if m]
    errors = body.get("errors")
    if isinstance(errors, Mapping):
        parts.extend(f"{field}: {message}" for field, message in errors.items())
    message = body.get("message")
    if not parts and isinstance(message, str) and message:
        parts.append(message)
    return "; ".join(parts) or response.reason_phrase


def raise_for_status(response: httpx2.Response, *, action: str, what: str) -> None:
    """Raise a model-facing `ToolError` for any non-2xx response."""
    if response.is_success:
        return
    status = response.status_code
    if is_scope_mismatch(response):
        raise PermissionDenied(
            PROVIDER,
            action=action,
            hint="The sign-in lacks the OAuth scope for this call. Do not retry; tell the user.",
        )
    if status == 401:
        raise ToolError(
            f"{PROVIDER} rejected the credentials while trying to {action} (401). "
            "Do not retry; the user has to sign in again."
        )
    if status == 403:
        raise PermissionDenied(
            PROVIDER,
            action=action,
            hint=f"{error_text(response)}. Do not retry; tell the user which permission is needed.",
        )
    if status == 404:
        raise NotFoundOrNoAccess(PROVIDER, what)
    if status == 400:
        raise ToolError(
            f"{PROVIDER} rejected the request to {action}: {error_text(response)}. "
            "Fix the arguments and call again."
        )
    if status == 409:
        raise ToolError(f"{PROVIDER} could not {action}: {error_text(response)}.")
    if status >= 500:
        raise ToolError(
            f"{PROVIDER} had a server error ({status}) while trying to {action}. Try again later."
        )
    raise ToolError(
        f"{PROVIDER} returned {status} while trying to {action}: {error_text(response)}."
    )


# --- Client -------------------------------------------------------------------------------


class JiraClient:
    def __init__(self, api: ApiClient, *, site: str = SITE_URL) -> None:
        self.api = api
        self.site = site

    def browse_url(self, key: str) -> str:
        return f"{self.site}/browse/{key}"

    async def search_issues(
        self, jql: str, *, cursor: str | None = None, page_size: int = 20
    ) -> Page[IssueSummary]:
        """Issues matching `jql` via the enhanced search (`/search/jql`), one page at a time."""
        if not jql.strip():
            raise ToolError(
                "The JQL query must not be empty: Jira rejects unbounded searches. "
                "Add a restriction, e.g. 'project = BENCH ORDER BY updated DESC'."
            )
        params = {
            "jql": jql,
            "fields": SEARCH_FIELDS,
            PAGE_SIZE_PARAM: min(_positive("page_size", page_size), MAX_PAGE_SIZE),
        }
        page = await fetch_page(self.api, "/search/jql", PAGINATION, cursor=cursor, params=params)
        raise_for_status(page.response, action="search issues", what="The search")
        issues = _body(page.response).get("issues") or []
        return Page[IssueSummary](
            items=[IssueSummary.model_validate(i) for i in issues], next_cursor=page.next_cursor
        )

    async def get_issue(self, key: str) -> Issue:
        key = issue_key(key)
        response = await self.api.get(f"/issue/{key}", params={"fields": ISSUE_FIELDS})
        raise_for_status(response, action="read the issue", what=f"Issue {key}")
        return Issue.model_validate(response.json())

    async def create_issue(
        self,
        project: str,
        summary: str,
        *,
        issue_type: str = "Task",
        description: str | None = None,
        labels: Sequence[str] = (),
    ) -> CreatedIssue:
        summary = summary.strip()
        if not summary:
            raise ToolError("The issue summary must not be empty.")
        if len(summary) > MAX_SUMMARY or "\n" in summary:
            raise ToolError(
                f"The summary must be one line of at most {MAX_SUMMARY} characters; "
                "put details into the description."
            )
        if bad := [label for label in labels if not label or re.search(r"\s", label)]:
            raise ToolError(f"Labels cannot be empty or contain spaces: {bad!r}.")
        fields: dict[str, Any] = {
            "project": {"key": project_key(project)},
            "summary": summary,
            "issuetype": {"name": issue_type},
        }
        if description:
            fields["description"] = text_to_adf(description)
        if labels:
            fields["labels"] = list(labels)
        response = await self.api.post("/issue", json={"fields": fields})
        raise_for_status(response, action="create an issue", what=f"Project {project}")
        return CreatedIssue.model_validate(response.json())

    async def add_comment(self, key: str, body: str) -> Comment:
        key = issue_key(key)
        if not body.strip():
            raise ToolError("The comment must not be empty.")
        response = await self.api.post(f"/issue/{key}/comment", json={"body": text_to_adf(body)})
        raise_for_status(response, action="comment on the issue", what=f"Issue {key}")
        return Comment.model_validate(response.json())

    async def transitions(self, key: str) -> list[Transition]:
        """Workflow transitions available for the issue right now."""
        key = issue_key(key)
        response = await self.api.get(f"/issue/{key}/transitions")
        raise_for_status(response, action="read the issue's transitions", what=f"Issue {key}")
        return [Transition.model_validate(t) for t in _body(response).get("transitions") or []]

    async def transition_issue(self, key: str, transition_id: str) -> None:
        key = issue_key(key)
        if not transition_id.isdigit():
            raise ToolError(f"Invalid transition id {transition_id!r}.")
        response = await self.api.post(
            f"/issue/{key}/transitions", json={"transition": {"id": transition_id}}
        )
        raise_for_status(response, action="transition the issue", what=f"Issue {key}")
