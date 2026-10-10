"""In-memory Jira Cloud API (v3) for development and unit tests.

Serves the endpoints the client uses in Jira's raw JSON shape, so the client's models
parse exactly what they would parse from the real API: enhanced search
(`/search/jql`, nextPageToken pagination, JQL subset from `jql.py`, `fields` filtering),
issues, comments (ADF bodies), workflow transitions and the OAuth token endpoint.
Errors look like Jira's: `{"errorMessages": [...], "errors": {...}}`, and a call outside
the token's OAuth scopes is a 401 "scope does not match", as on the Atlassian gateway.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypeGuard
from urllib.parse import urlsplit

import httpx2

from bench_core.auth import StaticToken
from bench_core.clock import Clock
from bench_core.testing import FakeBackend, FakeClock, FakeRequest

from bench_jira.adf import text_to_adf
from bench_jira.auth import ACCESSIBLE_RESOURCES_URL, TOKEN_URL
from bench_jira.client import BASE_URL, GATEWAY_URL, MAX_SUMMARY, PAGE_SIZE_PARAM, SITE_URL
from bench_jira.fake import jql
from bench_jira.scopes import READ, WRITE

BASE_PATH = urlsplit(BASE_URL).path
TOKEN_PATH = urlsplit(TOKEN_URL).path
CLOUD_ID = "11111111-2222-3333-4444-555555555555"
"""Cloud id of the fake site behind the gateway (OAuth user tokens)."""
TOKEN_PARAM = "nextPageToken"

CATEGORIES = {"new": "To Do", "indeterminate": "In Progress", "done": "Done"}
STATUSES = {
    "To Do": ("10000", "new"),
    "In Progress": ("3", "indeterminate"),
    "Done": ("10001", "done"),
}
TRANSITIONS = [
    ("11", "Back to To Do", "To Do"),
    ("21", "Start progress", "In Progress"),
    ("31", "Done", "Done"),
]
"""(id, name, target status): the simplified workflow, every status reachable from any other."""
ISSUE_TYPES = {"Task": "10001", "Bug": "10002", "Story": "10003"}
PRIORITIES = {"Highest": "1", "High": "2", "Medium": "3", "Low": "4", "Lowest": "5"}

BOT = {"accountId": "557058:bench-bot", "displayName": "Bench Bot", "accountType": "atlassian"}
"""The account behind every token: `currentUser()` and the author of everything created."""


def timestamp(seconds: float) -> str:
    """Jira's format: 2026-09-01T09:00:00.000+0000."""
    return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%S.000+0000")


def credential(token: str) -> StaticToken:
    """The static credential that presents `token` (see `JiraFake.issue_token`)."""
    return StaticToken("jira", token)


def _oauth_error(status: int, error: str, description: str) -> httpx2.Response:
    return httpx2.Response(status, json={"error": error, "error_description": description})


def _error(status: int, *messages: str, **errors: str) -> httpx2.Response:
    return httpx2.Response(status, json={"errorMessages": list(messages), "errors": errors})


def _is_adf(value: Any) -> TypeGuard[dict[str, Any]]:
    return isinstance(value, dict) and value.get("type") == "doc" and value.get("version") == 1


@dataclass
class JiraFake(FakeBackend):
    clock: Clock = field(default_factory=FakeClock)
    projects: dict[str, str] = field(default_factory=lambda: {"BENCH": "Bench Sandbox"})
    """Project key -> name."""
    users: dict[str, dict[str, str]] = field(default_factory=lambda: {BOT["accountId"]: BOT})
    issues: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Issue key -> raw issue JSON with all fields."""
    oauth_clients: dict[str, str] = field(default_factory=dict)
    """client_id -> client_secret of OAuth apps that may refresh user tokens."""
    refresh_tokens: dict[str, frozenset[str]] = field(default_factory=dict)
    """Valid (unused) refresh tokens -> scopes of the access tokens they mint."""
    _next_id: int = field(default=10000, init=False, repr=False)
    _next_comment: int = field(default=20000, init=False, repr=False)
    _minted: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.route("POST", TOKEN_PATH, public=True)(self._token)
        base = BASE_PATH
        self.route("GET", base + "/search/jql", scopes=[READ])(self._search)
        self.route("GET", base + "/issue/{key}", scopes=[READ])(self._get_issue)
        self.route("POST", base + "/issue", scopes=[WRITE])(self._create_issue)
        self.route("POST", base + "/issue/{key}/comment", scopes=[WRITE])(self._add_comment)
        self.route("GET", base + "/issue/{key}/transitions", scopes=[READ])(self._transitions)
        self.route("POST", base + "/issue/{key}/transitions", scopes=[WRITE])(self._transition)
        # `bench doctor` only: the site (API token) and the gateway (OAuth user token).
        gateway = urlsplit(GATEWAY_URL).path + "/{cloud_id}/rest/api/3"
        self.route("GET", urlsplit(ACCESSIBLE_RESOURCES_URL).path)(self._accessible_resources)
        for prefix in (base, gateway):
            self.route("GET", prefix + "/myself")(self._myself)
            self.route("GET", prefix + "/project/{key}", scopes=[READ])(self._get_project)

    # --- setup ----------------------------------------------------------------------------

    def _mint(self, prefix: str) -> str:
        self._minted += 1
        return f"fake-{prefix}-{self._minted}"

    def user_tokens(self, scopes: Sequence[str] = ()) -> tuple[str, str]:
        """A fresh access + refresh token pair, as after `bench auth login jira`."""
        access = self.issue_token(self._mint("access"), scopes)
        refresh = self._mint("refresh")
        self.refresh_tokens[refresh] = frozenset(scopes)
        return access, refresh

    def now(self) -> str:
        return timestamp(self.clock.time())

    def add_user(self, account_id: str, display_name: str) -> dict[str, str]:
        user = {"accountId": account_id, "displayName": display_name, "accountType": "atlassian"}
        self.users[account_id] = user
        return user

    def _status(self, name: str) -> dict[str, Any]:
        status_id, category = STATUSES[name]
        return {
            "id": status_id,
            "name": name,
            "statusCategory": {"key": category, "name": CATEGORIES[category]},
        }

    def add_issue(
        self,
        summary: str,
        *,
        project: str = "BENCH",
        issue_type: str = "Task",
        status: str = "To Do",
        priority: str = "Medium",
        description: Any = None,
        labels: Sequence[str] = (),
        assignee: str | None = None,
        reporter: str | None = None,
        created: str | None = None,
    ) -> dict[str, Any]:
        """A new issue; `description` is plain text (converted to ADF) or ADF already."""
        self._next_id += 1
        number = 1 + sum(
            1 for i in self.issues.values() if i["fields"]["project"]["key"] == project
        )
        key = f"{project}-{number}"
        when = created or self.now()
        issue = {
            "id": str(self._next_id),
            "key": key,
            "self": f"{BASE_URL}/issue/{self._next_id}",
            "fields": {
                "project": {"key": project, "name": self.projects[project]},
                "summary": summary,
                "issuetype": {"id": ISSUE_TYPES[issue_type], "name": issue_type},
                "status": self._status(status),
                "priority": {"id": PRIORITIES[priority], "name": priority},
                "labels": list(labels),
                "assignee": self.users[assignee] if assignee else None,
                "reporter": self.users[reporter or BOT["accountId"]],
                "description": text_to_adf(description)
                if isinstance(description, str)
                else description,
                "created": when,
                "updated": when,
                "comment": {"comments": [], "total": 0, "maxResults": 0, "startAt": 0},
            },
        }
        self.issues[key] = issue
        return issue

    def add_comment(
        self, key: str, body: Any, *, author: str | None = None, created: str | None = None
    ) -> dict[str, Any]:
        self._next_comment += 1
        when = created or self.now()
        issue = self.issues[key]
        comment = {
            "id": str(self._next_comment),
            "self": f"{issue['self']}/comment/{self._next_comment}",
            "author": self.users[author or BOT["accountId"]],
            "body": text_to_adf(body) if isinstance(body, str) else body,
            "created": when,
            "updated": when,
        }
        thread = issue["fields"]["comment"]
        thread["comments"].append(comment)
        thread["total"] = thread["maxResults"] = len(thread["comments"])
        issue["fields"]["updated"] = when
        return comment

    def find(self, key_or_id: str) -> dict[str, Any] | None:
        wanted = key_or_id.upper()
        return self.issues.get(wanted) or next(
            (i for i in self.issues.values() if i["id"] == wanted), None
        )

    def context(self) -> jql.Context:
        return jql.Context(
            current_user=BOT["accountId"],
            now=datetime.fromtimestamp(self.clock.time(), UTC),
            projects=self.projects,
            statuses={name: category for name, (_, category) in STATUSES.items()},
            categories=CATEGORIES,
            issue_types=list(ISSUE_TYPES),
            priorities=list(PRIORITIES),
        )

    # --- provider error format ------------------------------------------------------------

    def unauthorized_response(self, request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, json={"code": 401, "message": "Unauthorized"})

    def forbidden_response(
        self, request: httpx2.Request, missing: Sequence[str]
    ) -> httpx2.Response:
        return httpx2.Response(
            401, json={"code": 401, "message": "Unauthorized; scope does not match"}
        )

    def not_found_response(self, request: httpx2.Request) -> httpx2.Response:
        return _error(404, "null for uri: " + str(request.url))

    def server_error_response(self, request: httpx2.Request, status: int) -> httpx2.Response:
        return _error(status, "Internal server error")

    def rate_limit_response(
        self, request: httpx2.Request, retry_after: float | None
    ) -> httpx2.Response:
        response = httpx2.Response(429, text="Rate limit exceeded.")
        if retry_after is not None:
            response.headers["Retry-After"] = f"{retry_after:g}"
        return response

    def _missing_issue(self) -> httpx2.Response:
        return _error(404, "Issue does not exist or you do not have permission to see it.")

    # --- token endpoint -------------------------------------------------------------------

    def _token(self, r: FakeRequest) -> httpx2.Response:
        """refresh_token grant (JSON or form body); every refresh token works once."""
        body = r.json() if "json" in r.request.headers.get("content-type", "") else r.form()
        if self.oauth_clients.get(body.get("client_id", "")) != body.get("client_secret"):
            return _oauth_error(401, "access_denied", "Unauthorized")
        if body.get("grant_type") != "refresh_token":
            return _oauth_error(400, "unsupported_grant_type", "Only refresh_token is faked.")
        scopes = self.refresh_tokens.pop(body.get("refresh_token", ""), None)
        if scopes is None:
            return _oauth_error(403, "invalid_grant", "Unknown or invalid refresh token.")
        access, refresh = self.user_tokens(sorted(scopes))
        return httpx2.Response(
            200,
            json={
                "access_token": access,
                "refresh_token": refresh,
                "expires_in": 3600,
                "token_type": "Bearer",
                "scope": " ".join(sorted(scopes)),
            },
        )

    # --- routes ---------------------------------------------------------------------------

    @staticmethod
    def _select(issue: Mapping[str, Any], fields: str | None, default: str) -> dict[str, Any]:
        """The issue with only the requested fields (`*all`/`*navigable` = everything)."""
        wanted = [f.strip() for f in (fields or default).split(",") if f.strip()]
        out = {k: v for k, v in issue.items() if k != "fields"}
        if "*all" in wanted or "*navigable" in wanted:
            out["fields"] = issue["fields"]
        else:
            out["fields"] = {k: v for k, v in issue["fields"].items() if k in wanted}
        return out

    def _search(self, r: FakeRequest) -> httpx2.Response:
        try:
            query = jql.parse(r.query.get("jql", ""), self.context())
        except jql.JqlError as e:
            return _error(400, str(e))
        try:
            size = min(max(int(r.query.get(PAGE_SIZE_PARAM, "50")), 1), 5000)
            token = r.query.get(TOKEN_PARAM)
            offset = int(token.removeprefix("page-")) if token else 0
        except ValueError:
            return _error(400, "The nextPageToken or maxResults value is invalid.")
        hits = query.sort([i for i in self.issues.values() if query.matches(i)])
        chunk = hits[offset : offset + size]
        more = offset + size < len(hits)
        body: dict[str, Any] = {
            "issues": [self._select(i, r.query.get("fields"), "id") for i in chunk],
            "isLast": not more,
        }
        if more:
            body["nextPageToken"] = f"page-{offset + size}"
        return httpx2.Response(200, json=body)

    def _accessible_resources(self, r: FakeRequest) -> httpx2.Response:
        site = {"id": CLOUD_ID, "url": SITE_URL, "name": "bench-sandbox", "scopes": []}
        return httpx2.Response(200, json=[site])

    def _myself(self, r: FakeRequest) -> httpx2.Response:
        return httpx2.Response(200, json=BOT)

    def _get_project(self, r: FakeRequest) -> httpx2.Response:
        key = r.params["key"].upper()
        if key not in self.projects:
            return _error(404, f"No project could be found with key '{key}'.")
        return httpx2.Response(200, json={"key": key, "name": self.projects[key]})

    def _get_issue(self, r: FakeRequest) -> httpx2.Response:
        issue = self.find(r.params["key"])
        if issue is None:
            return self._missing_issue()
        return httpx2.Response(200, json=self._select(issue, r.query.get("fields"), "*all"))

    def _create_issue(self, r: FakeRequest) -> httpx2.Response:
        fields = (r.json() or {}).get("fields") or {}
        errors: dict[str, str] = {}
        project = (fields.get("project") or {}).get("key")
        if project not in self.projects:
            errors["project"] = "valid project is required"
        summary = fields.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            errors["summary"] = "You must specify a summary of the issue."
        elif len(summary) > MAX_SUMMARY:
            errors["summary"] = f"Summary must be less than {MAX_SUMMARY + 1} characters."
        issue_type = (fields.get("issuetype") or {}).get("name")
        if issue_type not in ISSUE_TYPES:
            errors["issuetype"] = "Specify a valid issue type"
        description = fields.get("description")
        if description is not None and not _is_adf(description):
            errors["description"] = "Operation value must be an Atlassian Document (see ADF)."
        labels = fields.get("labels") or []
        for label in labels:
            if " " in label:
                errors["labels"] = f"The label '{label}' contains spaces which is invalid."
        priority = (fields.get("priority") or {}).get("name", "Medium")
        if priority not in PRIORITIES:
            errors["priority"] = f"Specify the Priority (name) in the string format: {priority}"
        assignee = (fields.get("assignee") or {}).get("accountId")
        if assignee is not None and assignee not in self.users:
            errors["assignee"] = f"User '{assignee}' cannot be assigned issues."
        if errors:
            return _error(400, **errors)
        assert isinstance(summary, str)
        assert isinstance(project, str)
        assert isinstance(issue_type, str)
        issue = self.add_issue(
            summary,
            project=project,
            issue_type=issue_type,
            priority=priority,
            description=description,
            labels=labels,
            assignee=assignee,
        )
        return httpx2.Response(
            201, json={"id": issue["id"], "key": issue["key"], "self": issue["self"]}
        )

    def _add_comment(self, r: FakeRequest) -> httpx2.Response:
        issue = self.find(r.params["key"])
        if issue is None:
            return self._missing_issue()
        body = (r.json() or {}).get("body")
        if not _is_adf(body):
            return _error(400, comment="Comment body must be an Atlassian Document (see ADF).")
        if not (body.get("content") or []):
            return _error(400, comment="Comment body can not be empty!")
        return httpx2.Response(201, json=self.add_comment(issue["key"], body))

    def _available(self, issue: dict[str, Any]) -> list[dict[str, Any]]:
        current = issue["fields"]["status"]["name"]
        return [
            {"id": tid, "name": name, "to": self._status(target), "isAvailable": True}
            for tid, name, target in TRANSITIONS
            if target != current
        ]

    def _transitions(self, r: FakeRequest) -> httpx2.Response:
        issue = self.find(r.params["key"])
        if issue is None:
            return self._missing_issue()
        return httpx2.Response(200, json={"transitions": self._available(issue)})

    def _transition(self, r: FakeRequest) -> httpx2.Response:
        issue = self.find(r.params["key"])
        if issue is None:
            return self._missing_issue()
        wanted = str(((r.json() or {}).get("transition") or {}).get("id"))
        match = next((t for t in self._available(issue) if t["id"] == wanted), None)
        if match is None:
            return _error(400, f"Transition id '{wanted}' is not valid for this issue.")
        issue["fields"]["status"] = match["to"]
        issue["fields"]["updated"] = self.now()
        return httpx2.Response(204)
