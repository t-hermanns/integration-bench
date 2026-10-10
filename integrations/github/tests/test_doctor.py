"""`bench doctor github` against the fake: installation token, repo access, permissions."""

from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from bench_core.auth import MemoryTokenStore, TokenSet
from bench_core.testing import FakeClock

from bench_github.doctor import checks
from bench_github.fake import GitHubFake, seeded_fake

pytestmark = pytest.mark.anyio

APP_ID = "4242"
INSTALLATION_ID = 7
FULL = {"issues": "write", "pull_requests": "write", "contents": "write", "metadata": "read"}


@pytest.fixture(scope="module")
def app_key() -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public = (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    return private, public


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def fake_with(clock: FakeClock, public_key: str, permissions: dict[str, str]) -> GitHubFake:
    fake = seeded_fake(clock)
    fake.register_app(APP_ID, public_key, {INSTALLATION_ID: permissions})
    return fake


def app_env(private_key: str, **extra: str) -> dict[str, str]:
    return {
        "GITHUB_APP_ID": APP_ID,
        "GITHUB_INSTALLATION_ID": str(INSTALLATION_ID),
        "GITHUB_PRIVATE_KEY": private_key,
        **extra,
    }


async def test_working_installation(clock: FakeClock, app_key: tuple[str, str]) -> None:
    fake = fake_with(clock, app_key[1], FULL)
    app, login = await checks(app_env(app_key[0]), transport=fake.transport(), clock=clock)
    assert app.status == "ok", app.detail
    assert "sees bench-sandbox/widgets" in app.detail
    assert "issues:write" in app.detail
    assert "6/6 tools" in app.detail
    assert login.status == "skip"
    assert "GITHUB_CLIENT_ID" in login.detail


async def test_private_key_from_file(
    clock: FakeClock, app_key: tuple[str, str], tmp_path: Path
) -> None:
    fake = fake_with(clock, app_key[1], FULL)
    pem = tmp_path / "app.pem"
    pem.write_text(app_key[0], encoding="utf-8")
    env = app_env("", GITHUB_PRIVATE_KEY_PATH=str(pem))
    app, _ = await checks(env, transport=fake.transport(), clock=clock)
    assert app.status == "ok", app.detail

    env["GITHUB_PRIVATE_KEY_PATH"] = str(tmp_path / "absent.pem")
    app, _ = await checks(env, transport=fake.transport(), clock=clock)
    assert app.status == "fail"
    assert "cannot read the private key" in app.detail


async def test_not_configured_makes_no_calls(clock: FakeClock, app_key: tuple[str, str]) -> None:
    fake = fake_with(clock, app_key[1], FULL)
    app, login = await checks(
        {"GITHUB_CLIENT_ID": "id", "GITHUB_CLIENT_SECRET": "secret"},
        transport=fake.transport(),
        store=MemoryTokenStore(),
    )
    assert app.status == "skip"
    assert "GITHUB_APP_ID" in app.detail
    assert "GITHUB_PRIVATE_KEY or GITHUB_PRIVATE_KEY_PATH" in app.detail
    assert "bench auth login github" in login.detail
    assert fake.requests == []


async def test_missing_permissions_fail_and_name_the_hidden_tools(
    clock: FakeClock, app_key: tuple[str, str]
) -> None:
    fake = fake_with(clock, app_key[1], {"issues": "read", "metadata": "read"})
    app, _ = await checks(app_env(app_key[0]), transport=fake.transport(), clock=clock)
    assert app.status == "fail"
    assert "2/6 tools" in app.detail
    assert "github_create_issue" in app.detail
    assert "github_list_pull_requests" in app.detail


async def test_wrong_installation_or_repo(clock: FakeClock, app_key: tuple[str, str]) -> None:
    fake = fake_with(clock, app_key[1], FULL)
    env = app_env(app_key[0], GITHUB_INSTALLATION_ID="99")
    app, _ = await checks(env, transport=fake.transport(), clock=clock)
    assert app.status == "fail"
    assert "404" in app.detail

    env = app_env(app_key[0], GITHUB_REPO="nope")
    app, _ = await checks(env, transport=fake.transport(), clock=clock)
    assert app.status == "fail"
    assert app.detail.startswith("repo bench-sandbox/nope: HTTP 404")


async def test_broken_settings_fail_instead_of_crashing(
    clock: FakeClock, app_key: tuple[str, str]
) -> None:
    fake = fake_with(clock, app_key[1], FULL)
    for env in (
        app_env("not a pem"),
        app_env(app_key[0], GITHUB_INSTALLATION_ID="seven"),
        app_env(app_key[0], GITHUB_ORG="-bad-"),
    ):
        app, _ = await checks(env, transport=fake.transport(), clock=clock)
        assert app.status == "fail", env


async def test_stored_login_is_used_and_refreshed(
    clock: FakeClock, app_key: tuple[str, str]
) -> None:
    fake = fake_with(clock, app_key[1], FULL)
    fake.oauth_clients["id"] = "secret"
    _, refresh = fake.user_tokens(["issues:write"])
    store = MemoryTokenStore({"github": TokenSet(access_token="expired", refresh_token=refresh)})
    env = {"GITHUB_CLIENT_ID": "id", "GITHUB_CLIENT_SECRET": "secret"}
    _, login = await checks(env, transport=fake.transport(), clock=clock, store=store)
    assert login.status == "ok", login.detail
    assert login.detail == "signed in as bench-tester; sees bench-sandbox/widgets"
    stored = store.load("github")
    assert stored is not None
    assert stored.refresh_token != refresh


async def test_revoked_login_fails_with_a_hint(clock: FakeClock, app_key: tuple[str, str]) -> None:
    fake = fake_with(clock, app_key[1], FULL)
    fake.oauth_clients["id"] = "secret"
    store = MemoryTokenStore({"github": TokenSet(access_token="x", refresh_token="used")})
    env = {"GITHUB_CLIENT_ID": "id", "GITHUB_CLIENT_SECRET": "secret"}
    _, login = await checks(env, transport=fake.transport(), clock=clock, store=store)
    assert login.status == "fail"
    assert "bench auth login" in login.detail
