"""`bench sandbox seed github`: order, authors and numbering against a recording stub.

The fake has no write endpoints for files, branches or merges, so a small stateful
handler stands in: it numbers issues and pull requests like GitHub and records calls.
"""

import base64
import json
from typing import Any

import httpx2
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from bench_core.auth import MemoryTokenStore, TokenSet
from bench_core.sandbox import SandboxError

from bench_github.fake.seed import CONTRIBUTOR, FILES, ISSUES, MAINTAINER, PULLS
from bench_github.sandbox import MERGED_FILE, labels, main_files, pull_plan, seed

pytestmark = pytest.mark.anyio

REPO = "/repos/bench-sandbox/widgets"


@pytest.fixture(scope="module")
def pem() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


def env(pem: str) -> dict[str, str]:
    return {
        "GITHUB_APP_ID": "1",
        "GITHUB_INSTALLATION_ID": "7",
        "GITHUB_PRIVATE_KEY": pem,
        "GITHUB_CLIENT_ID": "cid",
        "GITHUB_CLIENT_SECRET": "secret",
    }


def store() -> MemoryTokenStore:
    return MemoryTokenStore({"github": TokenSet("user-token", "refresh", 2**40)})


class GitHubStub:
    def __init__(self, issues: dict[int, str] | None = None, first_number: int = 1) -> None:
        self.issues = dict(issues or {})
        self.next = first_number
        self.calls: list[tuple[str, str, str, dict[str, Any]]] = []
        """(method, path, who, json body)"""

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        if path.endswith("/access_tokens"):
            return httpx2.Response(
                201,
                json={
                    "token": "bot-token",
                    "expires_at": "2099-01-01T00:00:00Z",
                    "permissions": {},
                },
            )
        who = "bot" if request.headers["authorization"] == "Bearer bot-token" else "user"
        body = json.loads(request.content) if request.content else {}
        self.calls.append((request.method, path.removeprefix(REPO), who, body))
        rel = path.removeprefix(REPO)
        if path == "/user":
            return httpx2.Response(200, json={"login": "real-maintainer"})
        if request.method == "GET" and rel.startswith("/issues/"):
            title = self.issues.get(int(rel.split("/")[2]))
            return httpx2.Response(200, json={"title": title}) if title else httpx2.Response(404)
        if request.method == "GET" and rel.startswith("/contents/"):
            return httpx2.Response(404)
        if request.method == "GET" and rel in ("/labels", "/milestones"):
            return httpx2.Response(200, json=[{"name": "bug"}] if rel == "/labels" else [])
        if rel == "/milestones":
            return httpx2.Response(201, json={"number": 3})
        if rel == "/git/ref/heads/main":
            return httpx2.Response(200, json={"object": {"sha": "abc"}})
        if request.method == "POST" and rel in ("/issues", "/pulls"):
            self.issues[self.next] = body["title"]
            self.next += 1
            return httpx2.Response(201, json={"number": self.next - 1})
        return httpx2.Response(200, json={})

    def where(self, method: str, prefix: str) -> list[tuple[str, str, dict[str, Any]]]:
        return [(p, who, b) for m, p, who, b in self.calls if m == method and p.startswith(prefix)]


def test_plan_puts_every_seed_file_on_main_once() -> None:
    plans = pull_plan()
    assert [p.number for p in plans] == [13, 14, 15, 16]
    merged = [p for p in plans if p.merged]
    assert [(p.path, p.content) for p in merged] == [(MERGED_FILE, dict(FILES)[MERGED_FILE])]
    assert sorted([*main_files(), (merged[0].path, merged[0].content)]) == sorted(FILES)
    assert {"bug", "performance", "good first issue"} <= set(labels())


async def test_seed_order_authors_and_states(pem: str) -> None:
    stub = GitHubStub()
    lines: list[str] = []
    await seed(env(pem), echo=lines.append, transport=httpx2.MockTransport(stub), store=store())

    puts = [p for p, _, _ in stub.where("PUT", "/contents/")]
    assert puts[: len(main_files())] == [f"/contents/{p}" for p, _ in main_files()]
    issues = stub.where("POST", "/issues")
    created = [(who, b) for p, who, b in issues if p == "/issues"]
    assert [b["title"] for _, b in created] == [title for title, *_ in ISSUES]
    assert created[0][0] == "bot"
    assert created[1][0] == "user"
    assert created[0][1]["assignees"] == ["real-maintainer"]
    assert created[0][1]["milestone"] == 3
    assert created[1][1]["milestone"] is None
    closed = [p for p, _, b in stub.where("PATCH", "/issues/") if b == {"state": "closed"}]
    assert closed == [f"/issues/{n}" for n, i in enumerate(ISSUES, 1) if i[2] == "closed"]

    pulls = [(who, b) for _, who, b in stub.where("POST", "/pulls")]
    roles = {CONTRIBUTOR: "bot", MAINTAINER: "user"}
    assert [(who, b["head"], b["draft"]) for who, b in pulls] == [
        (roles[author], head, draft) for _, author, head, _, draft, _ in PULLS
    ]
    assert [p for p, _, _ in stub.where("PUT", "/pulls/")] == ["/pulls/15/merge"]
    assert [p for p, _, _ in stub.where("PATCH", "/pulls/")] == ["/pulls/16"]
    branch_file = [b for p, _, b in stub.where("PUT", "/contents/src/widgets/__init__")]
    assert branch_file[0]["branch"] == "fix/login-mobile"
    assert base64.b64decode(branch_file[0]["content"]).decode() == dict(FILES)[MERGED_FILE]
    assert lines[-1] == "github: seeded /repos/bench-sandbox/widgets (12 issues, 4 pulls)"


async def test_rerun_on_a_seeded_repo_changes_nothing(pem: str) -> None:
    stub = GitHubStub({1: ISSUES[0][0], 16: PULLS[-1][0]})
    lines: list[str] = []
    await seed(env(pem), echo=lines.append, transport=httpx2.MockTransport(stub), store=store())
    assert lines == [f"github: {REPO} already holds the seed data; nothing to do"]
    assert {m for m, *_ in stub.calls} == {"GET"}


async def test_partial_or_used_repos_are_refused(pem: str) -> None:
    partial = GitHubStub({1: ISSUES[0][0]})
    with pytest.raises(SandboxError, match="not the complete seed"):
        await seed(env(pem), transport=httpx2.MockTransport(partial), store=store())
    used = GitHubStub(first_number=5)
    with pytest.raises(SandboxError, match="got number 5, expected 1"):
        await seed(env(pem), transport=httpx2.MockTransport(used), store=store())


async def test_maintainer_must_be_signed_in(pem: str) -> None:
    with pytest.raises(SandboxError, match="bench auth login github"):
        await seed(env(pem), store=MemoryTokenStore())
