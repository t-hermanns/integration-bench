"""In-memory fake of a provider API for unit tests, with fault injection.

A `FakeBackend` is an httpx2 `MockTransport` handler. Routes map `METHOD /path/{param}`
to handlers over in-memory state, bearer tokens carry scopes, and faults (rate limits,
server errors, expired tokens, permission errors, network errors) can be queued for
matching requests. Integrations subclass it and override the `*_response` hooks so
errors look like the real provider's (e.g. GitHub's 403 with `x-ratelimit-remaining: 0`).

Note: vcrpy also patches `MockTransport`, so do not run a fake inside an active cassette
unless the test is about recording.
"""

import base64
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import parse_qsl

import httpx2

FaultKind = Literal["rate_limit", "server_error", "unauthorized", "forbidden", "network"]


@dataclass(frozen=True)
class FakeRequest:
    """What a route handler sees: the raw request, path parameters and the caller's token."""

    request: httpx2.Request
    params: dict[str, str]
    token: str | None

    @property
    def method(self) -> str:
        return self.request.method

    @property
    def path(self) -> str:
        return self.request.url.path

    @property
    def query(self) -> dict[str, str]:
        return dict(self.request.url.params)

    def json(self) -> Any:
        return json.loads(self.request.content) if self.request.content else None

    def form(self) -> dict[str, str]:
        return dict(parse_qsl(self.request.content.decode(), keep_blank_values=True))


Handler = Callable[[FakeRequest], Any]


@dataclass(frozen=True)
class Fault:
    """A failure to return instead of the normal response. Build one with the classmethods."""

    kind: FaultKind
    retry_after: float | None = None
    status: int = 0
    missing: tuple[str, ...] = ()

    @classmethod
    def rate_limit(cls, retry_after: float | None = None) -> "Fault":
        return cls("rate_limit", retry_after=retry_after)

    @classmethod
    def server_error(cls, status: int = 503) -> "Fault":
        return cls("server_error", status=status)

    @classmethod
    def unauthorized(cls) -> "Fault":
        """A 401 as if the access token had expired; the token stays valid afterwards."""
        return cls("unauthorized")

    @classmethod
    def forbidden(cls, *missing: str) -> "Fault":
        return cls("forbidden", missing=missing)

    @classmethod
    def network(cls) -> "Fault":
        return cls("network")


def _compile(template: str) -> re.Pattern[str]:
    parts = re.split(r"\{(\w+)\}", template)
    regex = "".join(
        f"(?P<{part}>[^/]+)" if i % 2 else re.escape(part) for i, part in enumerate(parts)
    )
    return re.compile(f"^{regex}$")


@dataclass
class _Route:
    method: str
    pattern: re.Pattern[str]
    handler: Handler
    scopes: tuple[str, ...]


@dataclass
class _Injection:
    fault: Fault
    method: str | None
    pattern: re.Pattern[str] | None
    remaining: int | None  # None = until cleared

    def matches(self, request: httpx2.Request) -> bool:
        if self.method is not None and request.method != self.method:
            return False
        return self.pattern is None or self.pattern.match(request.url.path) is not None


@dataclass
class FakeBackend:
    """Base fake. Set `require_auth=False` for APIs or tests that do not check tokens."""

    require_auth: bool = True
    tokens: dict[str, frozenset[str]] = field(default_factory=dict)
    requests: list[httpx2.Request] = field(default_factory=list)
    _routes: list[_Route] = field(default_factory=list, init=False, repr=False)
    _injections: list[_Injection] = field(default_factory=list, init=False, repr=False)

    # --- setup ----------------------------------------------------------------------------

    def route(
        self, method: str, template: str, *, scopes: Sequence[str] = ()
    ) -> Callable[[Handler], Handler]:
        """Decorator: `@backend.route("GET", "/repos/{owner}/{repo}", scopes=["repo"])`."""

        def register(handler: Handler) -> Handler:
            self.add_route(method, template, handler, scopes=scopes)
            return handler

        return register

    def add_route(
        self, method: str, template: str, handler: Handler, *, scopes: Sequence[str] = ()
    ) -> None:
        self._routes.append(_Route(method.upper(), _compile(template), handler, tuple(scopes)))

    def issue_token(self, token: str, scopes: Sequence[str] = ()) -> str:
        self.tokens[token] = frozenset(scopes)
        return token

    def expire_token(self, token: str) -> None:
        """Every later request with this token gets `unauthorized_response`."""
        self.tokens.pop(token, None)

    def inject(
        self,
        fault: Fault,
        *,
        method: str | None = None,
        path: str | None = None,
        times: int | None = 1,
    ) -> None:
        """Queue `fault` for the next `times` matching requests (`None` = until cleared).

        `path` is a route template like `/repos/{owner}/{repo}/issues`. Injections are
        checked in the order they were added; the first match wins.
        """
        self._injections.append(
            _Injection(
                fault,
                method.upper() if method else None,
                _compile(path) if path else None,
                times,
            )
        )

    def clear_faults(self) -> None:
        self._injections.clear()

    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self)

    def calls(self, method: str | None = None, path: str | None = None) -> list[httpx2.Request]:
        pattern = _compile(path) if path else None
        return [
            r
            for r in self.requests
            if (method is None or r.method == method.upper())
            and (pattern is None or pattern.match(r.url.path))
        ]

    @staticmethod
    def basic_token(user: str, secret: str) -> str:
        """Token value as `extract_token` sees it for HTTP Basic auth (Jira API tokens)."""
        return base64.b64encode(f"{user}:{secret}".encode()).decode()

    # --- provider hooks (override in subclasses) --------------------------------------------

    def extract_token(self, request: httpx2.Request) -> str | None:
        _, _, value = request.headers.get("authorization", "").partition(" ")
        return value.strip() or None

    def rate_limit_response(
        self, request: httpx2.Request, retry_after: float | None
    ) -> httpx2.Response:
        headers = {"Retry-After": f"{retry_after:g}"} if retry_after is not None else {}
        return httpx2.Response(429, headers=headers, json={"message": "rate limit exceeded"})

    def server_error_response(self, request: httpx2.Request, status: int) -> httpx2.Response:
        return httpx2.Response(status, json={"message": "internal error"})

    def unauthorized_response(self, request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, json={"message": "invalid or expired token"})

    def forbidden_response(
        self, request: httpx2.Request, missing: Sequence[str]
    ) -> httpx2.Response:
        return httpx2.Response(
            403, json={"message": "insufficient permissions", "missing_scopes": list(missing)}
        )

    def not_found_response(self, request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(404, json={"message": "not found"})

    # --- dispatch -----------------------------------------------------------------------------

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        fault = self._take_fault(request)
        if fault is not None:
            return self._render(fault, request)

        token = self.extract_token(request)
        if self.require_auth and (token is None or token not in self.tokens):
            return self.unauthorized_response(request)

        for route in self._routes:
            if route.method != request.method:
                continue
            match = route.pattern.match(request.url.path)
            if match is None:
                continue
            if self.require_auth and token is not None:
                missing = [s for s in route.scopes if s not in self.tokens[token]]
                if missing:
                    return self.forbidden_response(request, missing)
            result = route.handler(FakeRequest(request, match.groupdict(), token))
            if isinstance(result, httpx2.Response):
                return result
            return httpx2.Response(200, json=result)
        return self.not_found_response(request)

    def _take_fault(self, request: httpx2.Request) -> Fault | None:
        for injection in self._injections:
            if not injection.matches(request):
                continue
            if injection.remaining is not None:
                injection.remaining -= 1
                if injection.remaining <= 0:
                    self._injections.remove(injection)
            return injection.fault
        return None

    def _render(self, fault: Fault, request: httpx2.Request) -> httpx2.Response:
        match fault.kind:
            case "rate_limit":
                return self.rate_limit_response(request, fault.retry_after)
            case "server_error":
                return self.server_error_response(request, fault.status)
            case "unauthorized":
                return self.unauthorized_response(request)
            case "forbidden":
                return self.forbidden_response(request, fault.missing)
            case "network":
                raise httpx2.ConnectError("injected network error", request=request)
