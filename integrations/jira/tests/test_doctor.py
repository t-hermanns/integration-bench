"""`bench doctor jira` against the fake: API token, project visibility, login settings."""

import pytest

from bench_core.auth import MemoryTokenStore, TokenSet
from bench_core.testing import FakeClock

from bench_jira.doctor import checks
from bench_jira.fake import JiraFake, seeded_fake
from bench_jira.fake.backend import CLOUD_ID
from bench_jira.scopes import LOGIN_SCOPES, READ, WRITE

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
    token, login = await checks(env, transport=fake.transport(), store=MemoryTokenStore())
    assert token.status == "skip"
    assert "JIRA_SITE, JIRA_EMAIL, JIRA_API_TOKEN" in token.detail
    assert login.detail == "not configured: set JIRA_SITE"
    assert fake.requests == []


LOGIN_ENV = {"JIRA_SITE": "bench-sandbox", "JIRA_CLIENT_ID": "id", "JIRA_CLIENT_SECRET": "secret"}


async def test_not_signed_in_skips(fake: JiraFake) -> None:
    _, login = await checks(LOGIN_ENV, transport=fake.transport(), store=MemoryTokenStore())
    assert login.status == "skip"
    assert login.detail == "not signed in; run `bench auth login jira`"


async def test_stored_login_finds_the_cloud_id_and_the_project(fake: JiraFake) -> None:
    access, refresh = fake.user_tokens(LOGIN_SCOPES)
    expires = fake.clock.time() + 3600
    tokens = TokenSet(access, refresh, expires_at=expires, scopes=frozenset(LOGIN_SCOPES))
    store = MemoryTokenStore({"jira": tokens})
    _, login = await checks(LOGIN_ENV, transport=fake.transport(), clock=fake.clock, store=store)
    assert login.status == "ok", login.detail
    assert f"cloud id {CLOUD_ID}; project BENCH visible" in login.detail
    assert "/ex/jira/" + CLOUD_ID + "/rest/api/3/project/BENCH" in str(fake.requests[-1].url)


async def test_login_for_another_site_fails(fake: JiraFake) -> None:
    access, refresh = fake.user_tokens(LOGIN_SCOPES)
    tokens = TokenSet(access, refresh, expires_at=fake.clock.time() + 3600)
    store = MemoryTokenStore({"jira": tokens})
    env = {**LOGIN_ENV, "JIRA_SITE": "elsewhere"}
    _, login = await checks(env, transport=fake.transport(), clock=fake.clock, store=store)
    assert login.status == "fail"
    assert "no access to elsewhere.atlassian.net" in login.detail
