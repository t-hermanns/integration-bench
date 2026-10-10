"""`bench auth login`: OAuth 2 authorization code flow through the browser.

The user signs in themselves in their own browser; a one-shot HTTP server on the
loopback interface receives the redirect with the code, which is then exchanged for a
token set and stored. `state` guards against forged redirects; PKCE (S256) where the
provider supports it. The redirect URI is fixed because providers compare it exactly
with the one registered in the app.
"""

import base64
import hashlib
import html
import secrets
import socket
import threading
import urllib.parse
import webbrowser
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import TracebackType
from typing import Self

import anyio

from bench_core.auth.oauth import OAuthError, TokenEndpoint
from bench_core.auth.store import TokenSet
from bench_core.clock import Clock, SystemClock

REDIRECT_URI = "http://localhost:8765/callback"
"""Registered in every app (Atlassian, GitHub, Google, Slack); must match exactly."""


@dataclass(frozen=True)
class LoginSpec:
    """How one provider's browser sign-in works; `bench_<name>.auth.LOGIN`."""

    account: str
    """Token store key, e.g. "jira"."""
    authorize_url: str
    token_url: str
    client_id_var: str
    client_secret_var: str | None = None
    scopes: tuple[str, ...] = ()
    params: Mapping[str, str] = field(default_factory=dict)
    """Extra authorize parameters, e.g. Atlassian's `audience` and `prompt`."""
    pkce: bool = True


class LoginError(Exception):
    pass


def pkce_pair() -> tuple[str, str]:
    """(verifier, S256 challenge) per RFC 7636."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def authorization_url(
    spec: LoginSpec,
    *,
    client_id: str,
    state: str,
    redirect_uri: str = REDIRECT_URI,
    code_challenge: str | None = None,
) -> str:
    query: dict[str, str] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "state": state,
        **spec.params,
    }
    if spec.scopes:
        query["scope"] = " ".join(spec.scopes)
    if code_challenge is not None:
        query["code_challenge"] = code_challenge
        query["code_challenge_method"] = "S256"
    return f"{spec.authorize_url}?{urllib.parse.urlencode(query, quote_via=urllib.parse.quote)}"


_PAGE = (
    "<!doctype html><meta charset=utf-8><title>integration-bench</title>"
    "<body style='font-family:sans-serif;margin:3em'><h1>{title}</h1><p>{text}</p>"
)


class _LoopbackServer(HTTPServer):
    """Binds exclusively. On Windows, `SO_REUSEADDR` (the `HTTPServer` default) would let
    a second process bind the same port and receive the code; elsewhere it only allows
    rebinding over TIME_WAIT, which a quick second login needs."""

    def server_bind(self) -> None:
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            self.allow_reuse_address = False
            self.socket.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        super().server_bind()


class CallbackReceiver:
    """Serves the redirect URI on the loopback interface until one callback arrives.

    `localhost` is bound as 127.0.0.1 (browsers resolve it there). Only the exact path
    counts; anything else (favicon, probes) gets a 404 and is ignored.
    """

    def __init__(self, redirect_uri: str = REDIRECT_URI) -> None:
        url = urllib.parse.urlsplit(redirect_uri)
        if url.scheme != "http" or url.hostname not in ("localhost", "127.0.0.1"):
            raise ValueError(f"redirect URI must be http on the loopback host: {redirect_uri}")
        self.path = url.path or "/"
        self.address = ("127.0.0.1", url.port or 80)
        self.params: dict[str, str] | None = None
        self._done = threading.Event()
        self._server: HTTPServer | None = None

    def __enter__(self) -> Self:
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                url = urllib.parse.urlsplit(self.path)
                if url.path != receiver.path or receiver._done.is_set():
                    self.send_error(404)
                    return
                params = dict(urllib.parse.parse_qsl(url.query))
                if "error" in params:
                    title, text = "Sign-in failed", params.get("error_description", params["error"])
                else:
                    title, text = "Signed in", "You can close this tab and return to the terminal."
                body = _PAGE.format(title=title, text=html.escape(text)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                receiver.params = params
                receiver._done.set()

            def log_message(self, format: str, *args: object) -> None:
                pass  # the query holds the code; keep it out of the terminal

        try:
            self._server = _LoopbackServer(self.address, Handler)
        except OSError as exc:
            raise LoginError(
                f"cannot listen on {self.address[0]}:{self.address[1]} ({exc}); "
                "is another login still running?"
            ) from None
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def wait(self, timeout: float) -> dict[str, str]:
        if not self._done.wait(timeout):
            raise LoginError(f"no sign-in within {timeout:.0f} s")
        assert self.params is not None
        return self.params

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


async def login(
    spec: LoginSpec,
    *,
    client_id: str,
    client_secret: str | None,
    redirect_uri: str = REDIRECT_URI,
    open_browser: Callable[[str], object] = webbrowser.open,
    wait_seconds: float = 300.0,
    endpoint: TokenEndpoint | None = None,
    clock: Clock | None = None,
) -> TokenSet:
    """Open the browser, wait for the redirect, exchange the code; returns the tokens."""
    state = secrets.token_urlsafe(24)
    verifier, challenge = pkce_pair() if spec.pkce else (None, None)
    url = authorization_url(
        spec, client_id=client_id, state=state, redirect_uri=redirect_uri, code_challenge=challenge
    )
    with CallbackReceiver(redirect_uri) as receiver:
        open_browser(url)
        params = await anyio.to_thread.run_sync(receiver.wait, wait_seconds)
    if "error" in params:
        raise OAuthError(params["error"], params.get("error_description"))
    if not secrets.compare_digest(params.get("state", ""), state):
        raise LoginError("the redirect carried a different state; start the login again")
    code = params.get("code")
    if not code:
        raise LoginError("the redirect carried no authorization code")
    endpoint = endpoint or TokenEndpoint(
        spec.token_url, client_id=client_id, client_secret=client_secret
    )
    return await endpoint.authorization_code(
        code, redirect_uri=redirect_uri, code_verifier=verifier, clock=clock or SystemClock()
    )
