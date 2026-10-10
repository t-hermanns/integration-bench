"""`bench sandbox seed github`: the fake's seed repo content in the live test repo.

Two real authors stand in for the fake's two people: the maintainer is the signed-in
user (`bench auth login github`), the contributor is the App's bot (installation token).
Issues and pull requests share their numbers, so the order is fixed: files on `main`,
issues 1-12, then pull requests 13-16. Each pull request needs its own commit; only the
merged one lands on `main`, and it adds the one seed file held back for it, so `main`
ends up with exactly the fake's files.
"""

import base64
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx2

from bench_core.auth import TokenStore, default_store
from bench_core.http import ApiClient
from bench_core.sandbox import Echo, SandboxError, SmokeCall, require

from bench_github.auth import LOGIN, installation, user_token
from bench_github.client import make_api, raise_for_status, repo_path
from bench_github.doctor import DEFAULT_ORG, DEFAULT_REPO, private_key
from bench_github.fake.seed import (
    COMMENTS,
    CONTRIBUTOR,
    FILES,
    ISSUES,
    MAINTAINER,
    OWNER,
    PULLS,
    REPO,
    issue_assignees,
    issue_author,
    issue_milestone,
)

MERGED_FILE = "src/widgets/__init__.py"
"""Seed file that reaches `main` through the merged pull request."""


@dataclass(frozen=True)
class PullPlan:
    number: int
    title: str
    author: str
    head: str
    state: str
    draft: bool
    merged: bool
    path: str
    content: str | bytes


def pull_plan() -> list[PullPlan]:
    """The pull requests in creation order, each with the one file its branch adds."""
    plans: list[PullPlan] = []
    files = dict(FILES)
    for i, (title, author, head, state, draft, merged) in enumerate(PULLS):
        if merged:
            path, content = MERGED_FILE, files[MERGED_FILE]
        else:
            path = f"changes/{head.replace('/', '-')}.md"
            content = f"# {title}\n\nWork in progress.\n"
        plans.append(
            PullPlan(len(ISSUES) + 1 + i, title, author, head, state, draft, merged, path, content)
        )
    return plans


def main_files() -> list[tuple[str, str | bytes]]:
    return [(path, content) for path, content in FILES if path != MERGED_FILE]


def labels() -> list[str]:
    return sorted({label for _, issue_labels, _, _ in ISSUES for label in issue_labels})


async def seed(
    env: Mapping[str, str],
    *,
    echo: Echo = print,
    transport: httpx2.AsyncBaseTransport | None = None,
    store: TokenStore | None = None,
) -> None:
    require(env, "GITHUB_APP_ID", "GITHUB_INSTALLATION_ID", "GITHUB_CLIENT_ID")
    key = private_key(env)
    if key is None:
        raise SandboxError("set GITHUB_PRIVATE_KEY or GITHUB_PRIVATE_KEY_PATH")
    store = store or default_store(env)
    if store.load(LOGIN.account) is None:
        raise SandboxError("not signed in; run `bench auth login github` (the maintainer)")
    bot = installation(
        "github-sandbox",
        app_id=env["GITHUB_APP_ID"],
        private_key_pem=key,
        installation_id=int(env["GITHUB_INSTALLATION_ID"]),
        transport=transport,
    )
    user = user_token(
        LOGIN.account,
        store,
        client_id=env["GITHUB_CLIENT_ID"],
        client_secret=env.get("GITHUB_CLIENT_SECRET"),
        transport=transport,
    )
    owner = env.get("GITHUB_ORG") or DEFAULT_ORG
    name = env.get("GITHUB_REPO") or DEFAULT_REPO
    async with (
        make_api(credential=bot, transport=transport) as bot_api,
        make_api(credential=user, transport=transport) as user_api,
    ):
        await Seeder(repo_path(owner, name), bot_api, user_api, echo).run()


class Seeder:
    def __init__(self, repo: str, bot: ApiClient, user: ApiClient, echo: Echo) -> None:
        self.repo = repo
        self.apis = {CONTRIBUTOR: bot, MAINTAINER: user}
        self.bot = bot
        self.echo = echo

    async def call(
        self, api: ApiClient, method: str, path: str, what: str, **json: Any
    ) -> dict[str, Any]:
        response = await api.request(method, self.repo + path, json=json or None)
        raise_for_status(response, action=f"{method} {path}", what=what)
        return response.json() if response.content else {}

    async def run(self) -> None:
        first = await self.bot.get(f"{self.repo}/issues/1")
        if first.status_code == 200:
            await self._check_seeded()
            self.echo(f"github: {self.repo} already holds the seed data; nothing to do")
            return
        if first.status_code != 404:
            raise_for_status(first, action="read issue 1", what="Issue 1")
        maintainer = (await self.apis[MAINTAINER].get("/user")).json()["login"]
        await self._labels()
        milestone = await self._milestone()
        for path, content in main_files():
            await self._put_file(path, content, branch=None)
        for number, (title, issue_labels, state, body) in enumerate(ISSUES, 1):
            author = issue_author(number)
            assignees = [maintainer for _ in issue_assignees(issue_labels, state)]
            created = await self.call(
                self.apis[author], "POST", "/issues", f"Issue {number}",
                title=title, body=body, labels=issue_labels, assignees=assignees,
                milestone=milestone if issue_milestone(issue_labels) else None,
            )  # fmt: skip
            self._expect(created["number"], number, title)
            for comment_author, text in COMMENTS.get(number, []):
                await self.call(
                    self.apis[comment_author], "POST", f"/issues/{number}/comments",
                    f"Issue {number}", body=text,
                )  # fmt: skip
            if state == "closed":
                await self.call(
                    self.bot, "PATCH", f"/issues/{number}", f"Issue {number}", state="closed"
                )
            self.echo(f"#{number} [{state}] {title}")
        base = await self.call(self.bot, "GET", "/git/ref/heads/main", "Branch main")
        for pull in pull_plan():
            await self.call(
                self.bot, "POST", "/git/refs", f"Branch {pull.head}",
                ref=f"refs/heads/{pull.head}", sha=base["object"]["sha"],
            )  # fmt: skip
            await self._put_file(pull.path, pull.content, branch=pull.head)
            created = await self.call(
                self.apis[pull.author], "POST", "/pulls", f"Pull request {pull.number}",
                title=pull.title, head=pull.head, base="main", draft=pull.draft,
            )  # fmt: skip
            self._expect(created["number"], pull.number, pull.title)
            if pull.merged:
                await self.call(
                    self.bot, "PUT", f"/pulls/{pull.number}/merge", f"Pull {pull.number}"
                )
            elif pull.state == "closed":
                await self.call(
                    self.bot, "PATCH", f"/pulls/{pull.number}", f"Pull {pull.number}",
                    state="closed",
                )  # fmt: skip
            self.echo(f"#{pull.number} [{'merged' if pull.merged else pull.state}] {pull.title}")
        self.echo(f"github: seeded {self.repo} ({len(ISSUES)} issues, {len(PULLS)} pulls)")

    def _expect(self, got: int, expected: int, title: str) -> None:
        if got != expected:
            raise SandboxError(
                f"{title!r} got number {got}, expected {expected}: the repo had issues or "
                "pull requests before (numbers are never reused); seed into a fresh repo"
            )

    async def _check_seeded(self) -> None:
        last = pull_plan()[-1]
        response = await self.bot.get(f"{self.repo}/issues/{last.number}")
        title = response.json().get("title") if response.status_code == 200 else None
        if title != last.title:
            raise SandboxError(
                f"{self.repo} has issues but not the complete seed (#{last.number} is "
                f"{title!r}); seed into a fresh repo"
            )

    async def _labels(self) -> None:
        response = await self.bot.get(f"{self.repo}/labels", params={"per_page": 100})
        raise_for_status(response, action="list labels", what="Labels")
        existing = {label["name"] for label in response.json()}
        for label in labels():
            if label not in existing:
                await self.call(self.bot, "POST", "/labels", f"Label {label}", name=label)

    async def _milestone(self) -> int:
        response = await self.bot.get(f"{self.repo}/milestones", params={"state": "all"})
        raise_for_status(response, action="list milestones", what="Milestones")
        for milestone in response.json():
            if milestone["title"] == "v2.4":
                return milestone["number"]
        created = await self.call(self.bot, "POST", "/milestones", "Milestone", title="v2.4")
        return created["number"]

    async def _put_file(self, path: str, content: str | bytes, *, branch: str | None) -> None:
        url = f"/contents/{quote(path)}"
        params = {"ref": branch} if branch else None
        if (await self.bot.get(self.repo + url, params=params)).status_code == 200:
            return
        raw = content.encode() if isinstance(content, str) else content
        body: dict[str, Any] = {
            "message": f"Add {path}",
            "content": base64.b64encode(raw).decode("ascii"),
        }
        if branch:
            body["branch"] = branch
        await self.call(self.bot, "PUT", url, f"File {path}", **body)


@asynccontextmanager
async def live_api(env: Mapping[str, str]) -> AsyncIterator[ApiClient]:
    """`bench record github`: the App's installation token (the contributor bot)."""
    require(env, "GITHUB_APP_ID", "GITHUB_INSTALLATION_ID")
    key = private_key(env)
    if key is None:
        raise SandboxError("set GITHUB_PRIVATE_KEY or GITHUB_PRIVATE_KEY_PATH")
    if (env.get("GITHUB_ORG") or DEFAULT_ORG, env.get("GITHUB_REPO") or DEFAULT_REPO) != (
        OWNER,
        REPO,
    ):
        raise SandboxError(f"cassettes replay against {OWNER}/{REPO}; record there")
    bot = installation(
        "github-record",
        app_id=env["GITHUB_APP_ID"],
        private_key_pem=key,
        installation_id=int(env["GITHUB_INSTALLATION_ID"]),
    )
    await bot.authorization()  # fetch the installation token now, not inside a cassette
    async with make_api(credential=bot) as api:
        yield api


_REPO = {"owner": OWNER, "repo": REPO}
SMOKE = [
    SmokeCall("github_search_issues", {"query": f"repo:{OWNER}/{REPO} is:issue"}),
    SmokeCall("github_get_issue", lambda found: {**_REPO, "number": found["items"][0]["number"]}),
    SmokeCall("github_list_pull_requests", _REPO),
    SmokeCall("github_get_file_contents", {**_REPO, "path": "README.md"}),
]
