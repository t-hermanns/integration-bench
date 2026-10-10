"""`bench sandbox seed jira` against an empty fake: same issues as the fake's own seed."""

import pytest

from bench_core.sandbox import SandboxError
from bench_core.testing import FakeClock

from bench_jira.fake import JiraFake, seeded_fake
from bench_jira.fake.backend import BOT
from bench_jira.sandbox import seed
from bench_jira.scopes import READ, WRITE

pytestmark = pytest.mark.anyio

EMAIL = "bot@example.com"
TOKEN = "api-token"
ENV = {"JIRA_SITE": "bench-sandbox", "JIRA_EMAIL": EMAIL, "JIRA_API_TOKEN": TOKEN}


def with_token(fake: JiraFake) -> JiraFake:
    fake.issue_token(fake.basic_token(EMAIL, TOKEN), [READ, WRITE])
    return fake


def summary(fake: JiraFake, key: str) -> tuple[object, ...]:
    f = fake.issues[key]["fields"]
    assignee = (f.get("assignee") or {}).get("accountId")
    comments = [c["body"] for c in (f.get("comment") or {}).get("comments", [])]
    return (
        f["summary"], f["issuetype"]["name"], f["status"]["name"], f["priority"]["name"],
        sorted(f["labels"]), f.get("description"), assignee, comments,
    )  # fmt: skip


async def test_seed_matches_the_fake_seed_with_only_the_bot_assigned() -> None:
    fake = with_token(JiraFake(clock=FakeClock()))
    lines: list[str] = []
    await seed(ENV, echo=lines.append, transport=fake.transport())
    reference = seeded_fake()
    assert sorted(fake.issues) == sorted(reference.issues)
    for key in reference.issues:
        expected = list(summary(reference, key))
        if expected[6] != BOT["accountId"]:
            expected[6] = None  # Alex and Sam do not exist live
        assert summary(fake, key) == tuple(expected), key
    assert lines[0] == "BENCH-1 [To Do] Checkout crashes when the cart is empty"
    assert lines[-1] == "jira: created 12 issues in BENCH"


async def test_rerun_changes_nothing() -> None:
    fake = with_token(seeded_fake())
    lines: list[str] = []
    await seed(ENV, echo=lines.append, transport=fake.transport())
    assert lines == ["jira: BENCH already holds the 12 seed issues; nothing to do"]
    assert all(r.method == "GET" for r in fake.requests)


async def test_foreign_issues_are_refused() -> None:
    fake = with_token(JiraFake(clock=FakeClock()))
    fake.add_issue("Something else")
    with pytest.raises(SandboxError, match="do not match the seed"):
        await seed(ENV, transport=fake.transport())


async def test_missing_settings_name_the_variables() -> None:
    with pytest.raises(SandboxError, match="JIRA_EMAIL, JIRA_API_TOKEN"):
        await seed({"JIRA_SITE": "bench-sandbox"})
