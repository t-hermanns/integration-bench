"""Browser sign-in: PKCE, the authorize URL, the loopback receiver and the code exchange."""

import base64
import hashlib
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import httpx2
import pytest

from bench_core.auth import (
    CallbackReceiver,
    FileTokenStore,
    KeyringTokenStore,
    LoginError,
    LoginSpec,
    OAuthError,
    TokenEndpoint,
    authorization_url,
    default_store,
    login,
    pkce_pair,
)
from bench_core.testing import FakeClock

pytestmark = pytest.mark.anyio

SPEC = LoginSpec(
    account="demo",
    authorize_url="https://auth.example.com/authorize",
    token_url="https://auth.example.com/token",
    client_id_var="DEMO_CLIENT_ID",
    client_secret_var="DEMO_CLIENT_SECRET",
    scopes=("read:things", "offline_access"),
    params={"audience": "api.example.com"},
)


def free_redirect_uri() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://localhost:{port}/callback"


def visit(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url.replace("localhost", "127.0.0.1"), timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, ""


def query(url: str) -> dict[str, str]:
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))


def test_pkce_challenge_is_the_s256_of_the_verifier() -> None:
    verifier, challenge = pkce_pair()
    assert 43 <= len(verifier) <= 128
    digest = hashlib.sha256(verifier.encode()).digest()
    assert challenge == base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    assert pkce_pair()[0] != verifier


def test_authorization_url_carries_scopes_extras_and_pkce() -> None:
    url = authorization_url(SPEC, client_id="cid", state="st", code_challenge="ch")
    assert url.startswith("https://auth.example.com/authorize?")
    assert query(url) == {
        "client_id": "cid",
        "redirect_uri": "http://localhost:8765/callback",
        "response_type": "code",
        "state": "st",
        "audience": "api.example.com",
        "scope": "read:things offline_access",
        "code_challenge": "ch",
        "code_challenge_method": "S256",
    }
    assert "%20" in url  # spaces as %20, not "+": some providers reject "+"


def test_receiver_ignores_other_paths_and_takes_one_callback() -> None:
    redirect = free_redirect_uri()
    with CallbackReceiver(redirect) as receiver:
        assert visit(redirect.replace("/callback", "/favicon.ico"))[0] == 404
        status, page = visit(redirect + "?code=abc&state=xyz")
        assert status == 200
        assert "Signed in" in page
        assert visit(redirect + "?code=second")[0] == 404
        assert receiver.wait(1) == {"code": "abc", "state": "xyz"}


def test_receiver_rejects_non_loopback_redirects() -> None:
    with pytest.raises(ValueError, match="loopback"):
        CallbackReceiver("https://example.com/callback")


def test_busy_port_is_a_login_error() -> None:
    redirect = free_redirect_uri()
    with CallbackReceiver(redirect), pytest.raises(LoginError, match="cannot listen"):
        CallbackReceiver(redirect).__enter__()


def token_endpoint(seen: list[dict[str, str]]) -> TokenEndpoint:
    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(dict(urllib.parse.parse_qsl(request.content.decode())))
        return httpx2.Response(
            200,
            json={
                "access_token": "at",
                "refresh_token": "rt",
                "expires_in": 3600,
                "scope": "read:things offline_access",
            },
        )

    return TokenEndpoint(
        SPEC.token_url,
        client_id="cid",
        client_secret="secret",
        transport=httpx2.MockTransport(handler),
    )


def browser(params: dict[str, str] | None = None):
    """A stand-in browser: follows the authorize URL straight to the redirect."""

    def open_browser(url: str) -> None:
        sent = query(url)
        answer = params if params is not None else {"code": "the-code", "state": sent["state"]}
        visit(f"{sent['redirect_uri']}?{urllib.parse.urlencode(answer)}")

    return open_browser


async def test_login_exchanges_the_code_with_the_pkce_verifier() -> None:
    seen: list[dict[str, str]] = []
    clock = FakeClock()
    tokens = await login(
        SPEC,
        client_id="cid",
        client_secret="secret",
        redirect_uri=free_redirect_uri(),
        open_browser=browser(),
        endpoint=token_endpoint(seen),
        clock=clock,
    )
    assert (tokens.access_token, tokens.refresh_token) == ("at", "rt")
    assert tokens.expires_at == clock.time() + 3600
    assert tokens.scopes == {"read:things", "offline_access"}
    form = seen[0]
    assert form["grant_type"] == "authorization_code"
    assert form["code"] == "the-code"
    assert form["redirect_uri"].endswith("/callback")
    assert len(form["code_verifier"]) >= 43
    assert (form["client_id"], form["client_secret"]) == ("cid", "secret")


async def test_login_without_pkce_sends_no_verifier() -> None:
    seen: list[dict[str, str]] = []
    spec = LoginSpec("demo", SPEC.authorize_url, SPEC.token_url, "X", pkce=False)
    await login(
        spec,
        client_id="cid",
        client_secret="secret",
        redirect_uri=free_redirect_uri(),
        open_browser=browser(),
        endpoint=token_endpoint(seen),
    )
    assert "code_verifier" not in seen[0]


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        ({"code": "c", "state": "forged"}, LoginError, "different state"),
        ({"error": "access_denied", "error_description": "user said no"}, OAuthError, "no"),
    ],
)
async def test_bad_redirects_exchange_nothing(
    params: dict[str, str], error: type[Exception], match: str
) -> None:
    seen: list[dict[str, str]] = []
    with pytest.raises(error, match=match):
        await login(
            SPEC,
            client_id="cid",
            client_secret=None,
            redirect_uri=free_redirect_uri(),
            open_browser=browser(params),
            endpoint=token_endpoint(seen),
        )
    assert seen == []


async def test_no_sign_in_times_out() -> None:
    with pytest.raises(LoginError, match="no sign-in"):
        await login(
            SPEC,
            client_id="cid",
            client_secret=None,
            redirect_uri=free_redirect_uri(),
            open_browser=lambda url: None,
            wait_seconds=0.2,
        )


def test_default_store_uses_the_token_dir_when_set(tmp_path: Path) -> None:
    store = default_store({"BENCH_TOKEN_DIR": str(tmp_path)})
    assert isinstance(store, FileTokenStore)
    assert store.directory == tmp_path
    assert isinstance(default_store({}), KeyringTokenStore)
