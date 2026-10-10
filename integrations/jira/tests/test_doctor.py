"""`bench doctor jira` against the fake: API token, project visibility, login settings."""

import pytest

from bench_core.testing import FakeClock

from bench_jira.doctor import checks
from bench_jira.fake import JiraFake, seeded_fake
from bench_jira.scopes import READ, WRITE

pytestmark = pytest.mark.anyio

EMAIL = "bot@example.com"
TOKEN = "api-token"
ENV = {"JIRA_SITE": "bench-sandbox", "JIRA_EMAIL": EMAIL, "JIRA_API_TOKEN": TOKEN}


@pytest.fixture
def fake() -> JiraFake:
    fake = seeded_fake(FakeClock())
    fake.issue_token(fake.basic_token(EMAIL, TOKEN), [READ, WRITE])
    return fake


async def test_working_api_token(fake: JiraFake) -> None:
    token, login = await checks(ENV, transport=fake.transport())
    assert token.status == "ok", token.detail
    assert token.detail == "bench-sandbox.atlassian.net as Bench Bot; project BENCH visible"
    assert login.status == "skip"
    assert "JIRA_CLIENT_ID" in login.detail


async def test_revoked_token_fails(fake: JiraFake) -> None:
    token, _ = await checks({**ENV, "JIRA_API_TOKEN": "revoked"}, transport=fake.transport())
    assert token.status == "fail"
    assert "401" in token.detail


async def test_unknown_project_fails(fake: JiraFake) -> None:
    token, _ = await checks({**ENV, "JIRA_PROJECT": "NOPE"}, transport=fake.transport())
    assert token.status == "fail"
    assert token.detail.startswith("project NOPE: HTTP 404")


async def test_broken_site_name_fails(fake: JiraFake) -> None:
    token, _ = await checks({**ENV, "JIRA_SITE": "https://x"}, transport=fake.transport())
    assert token.status == "fail"
    assert "Invalid Atlassian site name" in token.detail


async def test_not_configured_makes_no_calls(fake: JiraFake) -> None:
    env = {"JIRA_CLIENT_ID": "id", "JIRA_CLIENT_SECRET": "secret"}
    token, login = await checks(env, transport=fake.transport())
    assert token.status == "skip"
    assert "JIRA_SITE, JIRA_EMAIL, JIRA_API_TOKEN" in token.detail
    assert "bench auth login jira" in login.detail
    assert fake.requests == []
