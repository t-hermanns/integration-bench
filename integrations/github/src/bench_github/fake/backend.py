"""In-memory GitHub REST API for development and unit tests.

Stores issues, pull requests, comments and files as raw GitHub JSON, so the client's
models parse exactly what they would parse from the real API. Errors look like GitHub's:
401 "Bad credentials", 404 "Not Found" for anything hidden, 403 "Resource not accessible
by integration" with `X-Accepted-GitHub-Permissions`, and rate limits as 403 with
`x-ratelimit-remaining: 0` (primary) or `Retry-After` (secondary).

Tokens carry GitHub App permissions (`issues:write`); write implies read like on GitHub.
The fake also serves the two token endpoints: the OAuth refresh grant
(`/login/oauth/access_token`, rotating refresh tokens, errors as HTTP 200) and App
installation tokens (`/app/installations/{id}/access_tokens`, RS256 App JWT).
"""

import base64
import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx2
import jwt

from bench_core.clock import Clock
from bench_core.permissions import expand_levels
from bench_core.testing import FakeBackend, FakeClock, FakeRequest

API = "https://api.github.com"
WEB = "https://github.com"
DOCS = "https://docs.github.com/rest"
RATE_LIMIT = 5000


def _sha(data: bytes) -> str:
    return hashlib.sha1(data, usedforsecurity=False).hexdigest()


def timestamp(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def user(login: str) -> dict[str, Any]:
    kind = "Bot" if login.endswith("[bot]") else "User"
    return {
        "login": login,
        "id": sum(map(ord, login)),
        "type": kind,
        "html_url": f"{WEB}/{login}",
    }


@dataclass
class Repo:
    owner: str
    name: str
    issues: dict[int, dict[str, Any]] = field(default_factory=dict)
    """Issues and pull requests (GitHub numbers them together); PRs have `pull_request`."""
    pulls: dict[int, dict[str, Any]] = field(default_factory=dict)
    comments: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    files: dict[str, bytes] = field(default_factory=dict)
    users: set[str] = field(default_factory=set)
    """Logins that may be assigned (collaborators)."""

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def api_url(self) -> str:
        return f"{API}/repos/{self.full_name}"

    def next_number(self) -> int:
        return max(self.issues, default=0) + 1


@dataclass(frozen=True)
class App:
    public_key_pem: str
    installations: dict[int, dict[str, str]]


def _oauth_error(error: str, description: str) -> dict[str, str]:
    return {
        "error": error,
        "error_description": description,
        "error_uri": "https://docs.github.com/apps/managing-oauth-apps/troubleshooting-oauth-app-access-token-request-errors",
    }


@dataclass
class GitHubFake(FakeBackend):
    clock: Clock = field(default_factory=FakeClock)
    repos: dict[str, Repo] = field(default_factory=dict)
    known_users: set[str] = field(default_factory=set)
    oauth_clients: dict[str, str] = field(default_factory=dict)
    """client_id -> client_secret of OAuth/GitHub Apps that may refresh user tokens."""
    refresh_tokens: dict[str, frozenset[str]] = field(default_factory=dict)
    """Valid (unused) refresh tokens -> permissions of the access tokens they mint."""
    apps: dict[str, App] = field(default_factory=dict)
    _next_id: int = field(default=1_000_000, init=False, repr=False)
    _serial: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.route("POST", "/login/oauth/access_token", public=True)(self._oauth_token)
        self.route("POST", "/app/installations/{installation_id}/access_tokens", public=True)(
            self._installation_token
        )
        self.route("GET", "/search/issues")(self._search_issues)
        self.route("GET", "/repos/{owner}/{repo}/issues/{number}", scopes=["issues:read"])(
            self._get_issue
        )
        self.route("GET", "/repos/{owner}/{repo}/issues/{number}/comments", scopes=["issues:read"])(
            self._list_comments
        )
        self.route("POST", "/repos/{owner}/{repo}/issues", scopes=["issues:write"])(
            self._create_issue
        )
        self.route(
            "POST", "/repos/{owner}/{repo}/issues/{number}/comments", scopes=["issues:write"]
        )(self._create_comment)
        self.route("GET", "/repos/{owner}/{repo}/pulls", scopes=["pull_requests:read"])(
            self._list_pulls
        )
        self.route("GET", "/repos/{owner}/{repo}/contents", scopes=["contents:read"])(
            self._get_contents
        )
        self.route("GET", "/repos/{owner}/{repo}/contents/{path:path}", scopes=["contents:read"])(
            self._get_contents
        )

    # --- setup ----------------------------------------------------------------------------

    def issue_token(self, token: str, scopes: Sequence[str] = ()) -> str:
        return super().issue_token(token, sorted(expand_levels(scopes)))

    def _mint(self, prefix: str) -> str:
        self._serial += 1
        return f"fake-{prefix}-{self._serial}"

    def user_tokens(self, permissions: Sequence[str] = ()) -> tuple[str, str]:
        """A fresh access + refresh token pair, as after `bench auth login`."""
        access = self.issue_token(self._mint("user"), permissions)
        refresh = self._mint("refresh")
        self.refresh_tokens[refresh] = frozenset(permissions)
        return access, refresh

    def register_app(
        self, app_id: str, public_key_pem: str, installations: dict[int, dict[str, str]]
    ) -> None:
        """A GitHub App; `installations` maps installation id -> {permission: level}."""
        self.apps[app_id] = App(public_key_pem, installations)

    def new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def now(self) -> str:
        return timestamp(self.clock.time())

    def add_repo(self, owner: str, name: str, *, collaborators: Sequence[str] = ()) -> Repo:
        repo = Repo(owner, name, users={owner, *collaborators})
        self.repos[repo.full_name] = repo
        self.known_users |= repo.users
        return repo

    def add_issue(
        self,
        repo: Repo,
        title: str,
        *,
        author: str,
        body: str | None = None,
        labels: Sequence[str] = (),
        assignees: Sequence[str] = (),
        state: str = "open",
        created_at: str | None = None,
        updated_at: str | None = None,
        milestone: str | None = None,
    ) -> dict[str, Any]:
        number = repo.next_number()
        created = created_at or self.now()
        self.known_users.add(author)
        issue = {
            "url": f"{repo.api_url}/issues/{number}",
            "repository_url": repo.api_url,
            "html_url": f"{WEB}/{repo.full_name}/issues/{number}",
            "id": self.new_id(),
            "number": number,
            "title": title,
            "user": user(author),
            "labels": [{"id": sum(map(ord, n)), "name": n} for n in labels],
            "state": state,
            "state_reason": "completed" if state == "closed" else None,
            "assignees": [user(a) for a in assignees],
            "milestone": {"number": 1, "title": milestone} if milestone else None,
            "comments": 0,
            "created_at": created,
            "updated_at": updated_at or created,
            "closed_at": (updated_at or created) if state == "closed" else None,
            "author_association": "OWNER" if author == repo.owner else "CONTRIBUTOR",
            "body": body,
        }
        repo.issues[number] = issue
        repo.comments[number] = []
        return issue

    def add_comment(
        self, repo: Repo, number: int, body: str, *, author: str, created_at: str | None = None
    ) -> dict[str, Any]:
        comment_id = self.new_id()
        created = created_at or self.now()
        comment = {
            "id": comment_id,
            "url": f"{repo.api_url}/issues/comments/{comment_id}",
            "html_url": f"{WEB}/{repo.full_name}/issues/{number}#issuecomment-{comment_id}",
            "issue_url": f"{repo.api_url}/issues/{number}",
            "user": user(author),
            "body": body,
            "created_at": created,
            "updated_at": created,
            "author_association": "CONTRIBUTOR",
        }
        repo.comments[number].append(comment)
        issue = repo.issues[number]
        issue["comments"] += 1
        issue["updated_at"] = max(issue["updated_at"], created)
        return comment

    def add_pull(
        self,
        repo: Repo,
        title: str,
        *,
        author: str,
        head: str,
        base: str = "main",
        body: str | None = None,
        state: str = "open",
        draft: bool = False,
        merged: bool = False,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> dict[str, Any]:
        issue = self.add_issue(
            repo,
            title,
            author=author,
            body=body,
            state=state,
            created_at=created_at,
            updated_at=updated_at,
        )
        number = issue["number"]
        merged_at = issue["closed_at"] if merged else None
        issue["pull_request"] = {
            "url": f"{repo.api_url}/pulls/{number}",
            "html_url": f"{WEB}/{repo.full_name}/pull/{number}",
            "merged_at": merged_at,
        }
        issue["html_url"] = f"{WEB}/{repo.full_name}/pull/{number}"
        pull = {
            "url": f"{repo.api_url}/pulls/{number}",
            "id": self.new_id(),
            "html_url": issue["html_url"],
            "number": number,
            "state": state,
            "title": title,
            "user": issue["user"],
            "body": body,
            "labels": [],
            "draft": draft,
            "head": {"ref": head, "label": f"{repo.owner}:{head}", "sha": f"{number:040x}"},
            "base": {"ref": base, "label": f"{repo.owner}:{base}", "sha": "0" * 40},
            "created_at": issue["created_at"],
            "updated_at": issue["updated_at"],
            "closed_at": issue["closed_at"],
            "merged_at": merged_at,
        }
        repo.pulls[number] = pull
        return pull

    def add_file(self, repo: Repo, path: str, content: bytes | str) -> None:
        repo.files[path] = content.encode() if isinstance(content, str) else content

    # --- provider hooks -------------------------------------------------------------------

    def _error(
        self, status: int, message: str, headers: dict[str, str] | None = None
    ) -> httpx2.Response:
        body = {"message": message, "documentation_url": DOCS, "status": str(status)}
        return httpx2.Response(status, headers=headers, json=body)

    def unauthorized_response(self, request: httpx2.Request) -> httpx2.Response:
        return self._error(401, "Bad credentials")

    def not_found_response(self, request: httpx2.Request) -> httpx2.Response:
        return self._error(404, "Not Found")

    def forbidden_response(
        self, request: httpx2.Request, missing: Sequence[str]
    ) -> httpx2.Response:
        accepted = ",".join(m.replace(":", "=") for m in missing)
        return self._error(
            403,
            "Resource not accessible by integration",
            {"X-Accepted-GitHub-Permissions": accepted} if accepted else None,
        )

    def rate_limit_response(
        self, request: httpx2.Request, retry_after: float | None
    ) -> httpx2.Response:
        if retry_after is not None:
            return self._error(
                403,
                "You have exceeded a secondary rate limit. Please wait a few minutes "
                "before you try again.",
                {"Retry-After": f"{retry_after:g}"},
            )
        reset = int(self.clock.time()) + 60
        return self._error(
            403,
            "API rate limit exceeded for installation.",
            {
                "x-ratelimit-limit": str(RATE_LIMIT),
                "x-ratelimit-remaining": "0",
                "x-ratelimit-reset": str(reset),
                "x-ratelimit-resource": "core",
            },
        )

    def server_error_response(self, request: httpx2.Request, status: int) -> httpx2.Response:
        return self._error(status, "Server Error")

    # --- auth endpoints -------------------------------------------------------------------

    def _oauth_token(self, r: FakeRequest) -> Any:
        """github.com/login/oauth/access_token: errors come back as HTTP 200 with `error`."""
        form = r.form()
        if self.oauth_clients.get(form.get("client_id", "")) != form.get("client_secret"):
            return _oauth_error(
                "incorrect_client_credentials",
                "The client_id and/or client_secret passed are incorrect.",
            )
        if form.get("grant_type") != "refresh_token":
            return _oauth_error("unsupported_grant_type", "Only refresh_token is faked.")
        permissions = self.refresh_tokens.pop(form.get("refresh_token", ""), None)
        if permissions is None:
            return _oauth_error(
                "bad_refresh_token", "The refresh token passed is incorrect or expired."
            )
        access, refresh = self.user_tokens(sorted(permissions))
        return {
            "access_token": access,
            "expires_in": 28_800,
            "refresh_token": refresh,
            "refresh_token_expires_in": 15_897_600,
            "scope": "",
            "token_type": "bearer",
        }

    def _installation_token(self, r: FakeRequest) -> Any:
        try:
            claims = jwt.decode(r.token or "", options={"verify_signature": False})
            app = self.apps[str(claims["iss"])]
            jwt.decode(
                r.token or "",
                app.public_key_pem,
                algorithms=["RS256"],
                # Checked against the fake clock below, not the wall clock.
                options={"verify_exp": False, "verify_iat": False},
            )
        except (jwt.PyJWTError, KeyError):
            return self._error(401, "A JSON web token could not be decoded")
        now = self.clock.time()
        if not claims["iat"] <= now + 60 or not now < claims["exp"] <= now + 600 + 60:
            return self._error(401, "'Expiration time' claim ('exp') is too far in the future")
        permissions = app.installations.get(int(r.params["installation_id"]))
        if permissions is None:
            return self.not_found_response(r.request)
        token = self.issue_token(
            self._mint("installation"), [f"{k}:{v}" for k, v in permissions.items()]
        )
        body = {
            "token": token,
            "expires_at": timestamp(now + 3600),
            "permissions": permissions,
            "repository_selection": "selected",
        }
        return httpx2.Response(201, json=body)

    # --- helpers --------------------------------------------------------------------------

    def _repo(self, r: FakeRequest) -> Repo | None:
        return self.repos.get(f"{r.params['owner']}/{r.params['repo']}")

    def _ok(self, body: Any, *, status: int = 200, link: str | None = None) -> httpx2.Response:
        headers = {
            "x-github-api-version-selected": "2022-11-28",
            "x-ratelimit-limit": str(RATE_LIMIT),
            "x-ratelimit-remaining": str(RATE_LIMIT - len(self.requests)),
            "x-ratelimit-reset": str(int(self.clock.time()) + 3600),
            "x-ratelimit-resource": "core",
        }
        if link:
            headers["Link"] = link
        return httpx2.Response(status, headers=headers, json=body)

    def _paginate(self, r: FakeRequest, items: list[Any], *, default: int = 30) -> httpx2.Response:
        """Slice `items` by `page`/`per_page` and set a GitHub-style `Link` header."""
        try:
            per_page = min(max(int(r.query.get("per_page", default)), 1), 100)
            page = max(int(r.query.get("page", 1)), 1)
        except ValueError:
            return self._error(422, "Validation Failed")
        last = max((len(items) + per_page - 1) // per_page, 1)
        chunk = items[(page - 1) * per_page : page * per_page]
        return self._ok(chunk, link=self._link(r.request.url, page, last))

    @staticmethod
    def _link(url: httpx2.URL, page: int, last: int) -> str | None:
        def at(n: int) -> str:
            return str(url.copy_set_param("page", str(n)))

        rels = []
        if page < last:
            rels += [f'<{at(page + 1)}>; rel="next"', f'<{at(last)}>; rel="last"']
        if page > 1:
            rels += [f'<{at(1)}>; rel="first"', f'<{at(page - 1)}>; rel="prev"']
        return ", ".join(rels) or None

    # --- routes ---------------------------------------------------------------------------

    def _search_issues(self, r: FakeRequest) -> httpx2.Response:
        query = r.query.get("q", "")
        try:
            match = parse_search(query, self.known_users)
        except SearchError as exc:
            return self._validation_failed(
                {"message": str(exc), "resource": "Search", "field": "q", "code": "invalid"}
            )
        hits = [
            issue
            for repo in self.repos.values()
            for issue in repo.issues.values()
            if match(repo, issue)
        ]
        hits.sort(key=lambda i: (i["updated_at"], i["id"]), reverse=True)
        page = self._paginate(r, hits)
        if not page.is_success:
            return page
        body = {"total_count": len(hits), "incomplete_results": False, "items": page.json()}
        return httpx2.Response(200, headers=page.headers, json=body)

    def _get_issue(self, r: FakeRequest) -> Any:
        repo = self._repo(r)
        issue = repo and repo.issues.get(int(r.params["number"]))
        return self._ok(issue) if issue else self.not_found_response(r.request)

    def _list_comments(self, r: FakeRequest) -> Any:
        repo = self._repo(r)
        number = int(r.params["number"])
        if not repo or number not in repo.issues:
            return self.not_found_response(r.request)
        return self._paginate(r, repo.comments[number])

    def _create_issue(self, r: FakeRequest) -> Any:
        repo = self._repo(r)
        if not repo:
            return self.not_found_response(r.request)
        payload = r.json() or {}
        title = payload.get("title")
        if not isinstance(title, str) or not title.strip():
            return self._validation("Issue", "title", "missing_field")
        unknown = [a for a in payload.get("assignees") or [] if a not in repo.users]
        if unknown:
            return self._validation("Issue", "assignees", "invalid", value=unknown[0])
        issue = self.add_issue(
            repo,
            title,
            author=self._actor(r),
            body=payload.get("body"),
            labels=payload.get("labels") or [],
            assignees=payload.get("assignees") or [],
        )
        return self._ok(issue, status=201)

    def _create_comment(self, r: FakeRequest) -> Any:
        repo = self._repo(r)
        number = int(r.params["number"])
        if not repo or number not in repo.issues:
            return self.not_found_response(r.request)
        body = (r.json() or {}).get("body")
        if not isinstance(body, str) or not body:
            return self._validation("IssueComment", "body", "missing_field")
        return self._ok(self.add_comment(repo, number, body, author=self._actor(r)), status=201)

    def _list_pulls(self, r: FakeRequest) -> Any:
        repo = self._repo(r)
        if not repo:
            return self.not_found_response(r.request)
        state = r.query.get("state", "open")
        pulls = [p for p in repo.pulls.values() if state == "all" or p["state"] == state]
        key = "created_at" if r.query.get("sort", "created") == "created" else "updated_at"
        pulls.sort(key=lambda p: (p[key], p["number"]), reverse=r.query.get("direction") != "asc")
        return self._paginate(r, pulls)

    def _get_contents(self, r: FakeRequest) -> Any:
        repo = self._repo(r)
        if not repo:
            return self.not_found_response(r.request)
        path = r.params.get("path", "").strip("/")
        if path in repo.files:
            return self._ok(self._file_json(repo, path, r.query.get("ref", "main")))
        prefix = f"{path}/" if path else ""
        children = sorted(
            {p[len(prefix) :].split("/", 1)[0] for p in repo.files if p.startswith(prefix)}
        )
        if not children:
            return self.not_found_response(r.request)
        entries = []
        for child in children:
            full = prefix + child
            is_file = full in repo.files
            entries.append(
                {
                    "name": child,
                    "path": full,
                    "sha": _sha(full.encode()),
                    "size": len(repo.files[full]) if is_file else 0,
                    "type": "file" if is_file else "dir",
                }
            )
        return self._ok(entries)

    def _file_json(self, repo: Repo, path: str, ref: str) -> dict[str, Any]:
        content = repo.files[path]
        too_large = len(content) > 1_000_000
        encoded = base64.b64encode(content).decode()
        return {
            "type": "file",
            "name": path.rsplit("/", 1)[-1],
            "path": path,
            "sha": _sha(content),
            "size": len(content),
            "encoding": "none" if too_large else "base64",
            "content": ""
            if too_large
            else "\n".join(encoded[i : i + 60] for i in range(0, len(encoded), 60)) + "\n",
            "html_url": f"{WEB}/{repo.full_name}/blob/{ref}/{path}",
        }

    def _actor(self, r: FakeRequest) -> str:
        return "bench-bot[bot]"

    def _validation(
        self, resource: str, field_name: str, code: str, *, value: Any = None
    ) -> httpx2.Response:
        error: dict[str, Any] = {"resource": resource, "field": field_name, "code": code}
        if value is not None:
            error["value"] = value
        return self._validation_failed(error)

    def _validation_failed(self, *errors: dict[str, Any]) -> httpx2.Response:
        body = {
            "message": "Validation Failed",
            "errors": list(errors),
            "documentation_url": DOCS,
            "status": "422",
        }
        return httpx2.Response(422, json=body)


# --- Search query parsing -----------------------------------------------------------------


Predicate = Callable[[Repo, dict[str, Any]], bool]


class SearchError(ValueError):
    pass


def _negate(check: Predicate) -> Predicate:
    return lambda repo, issue: not check(repo, issue)


_TOKEN = re.compile(r'(-?)(\w+):("[^"]*"|\S+)|"([^"]*)"|(\S+)')


def parse_search(query: str, known_users: set[str]) -> Predicate:
    """Subset of GitHub's issue search: repo, is/type, state, label, author, assignee, text."""
    checks: list[Predicate] = []
    words: list[str] = []
    kinds: set[str] = set()
    for neg, key, value, quoted, word in _TOKEN.findall(query):
        if not key:
            words.append((quoted or word).lower())
            continue
        value = value.strip('"')
        key = key.lower()
        if key in ("is", "type") and value in ("issue", "pr", "pull-request"):
            kind = "issue" if value == "issue" else "pr"
            if neg:
                kind = "pr" if kind == "issue" else "issue"
            kinds.add(kind)
            continue
        if key in ("author", "assignee") and value not in known_users:
            raise SearchError(
                "The listed users and repositories cannot be searched either because the "
                "resources do not exist or you do not have permission to view them."
            )
        check = _qualifier(key, value)
        if check is None:
            continue  # unknown qualifiers are ignored, like GitHub ignores sort hints
        checks.append(_negate(check) if neg else check)
    kinds = kinds or {"issue", "pr"}

    def match(repo: Repo, issue: dict[str, Any]) -> bool:
        kind = "pr" if "pull_request" in issue else "issue"
        if kind not in kinds:
            return False
        text = f"{issue['title']}\n{issue.get('body') or ''}".lower()
        return all(w in text for w in words) and all(c(repo, issue) for c in checks)

    return match


def _qualifier(key: str, value: str) -> Predicate | None:
    match key:
        case "repo":
            return lambda repo, i: repo.full_name.lower() == value.lower()
        case "org" | "user":
            return lambda repo, i: repo.owner.lower() == value.lower()
        case "is" | "state" if value in ("open", "closed"):
            return lambda repo, i: i["state"] == value
        case "label":
            return lambda repo, i: value.lower() in {lb["name"].lower() for lb in i["labels"]}
        case "author":
            return lambda repo, i: (i["user"] or {}).get("login") == value
        case "assignee":
            return lambda repo, i: value in {a["login"] for a in i["assignees"]}
        case _:
            return None
