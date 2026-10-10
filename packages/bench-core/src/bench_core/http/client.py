"""`ApiClient`: an httpx2 async client with provider retry policies and a wait budget.

The wait budget caps how long one MCP tool call may spend sleeping on retries. Tools open
a budget with `api.wait_budget(seconds)`; every request inside (e.g. all pages of a
paginated call) draws from it. Without an open budget, each request gets the client's
default budget. A wait that does not fit raises `RateLimited`, which the model sees as
an actionable tool error instead of the call hanging.
"""

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import TracebackType
from typing import Any, Self

import httpx2
from mcp.server.mcpserver.exceptions import ToolError

from bench_core.clock import Clock, SystemClock
from bench_core.http.retry import IDEMPOTENT_METHODS, RetryPolicy, parse_retry_after


class RateLimited(ToolError):
    """The provider asked us to wait longer than the remaining wait budget allows."""

    def __init__(self, provider: str, retry_after: float) -> None:
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(
            f"{provider} rate limited, retry after {math.ceil(retry_after)} s. "
            "Do not retry immediately; tell the user or try again later."
        )


class WaitBudget:
    def __init__(self, seconds: float) -> None:
        self.remaining = seconds

    def fits(self, seconds: float) -> bool:
        return seconds <= self.remaining

    def spend(self, seconds: float) -> None:
        self.remaining -= seconds


_active_budget: ContextVar[WaitBudget | None] = ContextVar("bench_wait_budget", default=None)


class ApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        policy: RetryPolicy,
        clock: Clock | None = None,
        transport: httpx2.AsyncBaseTransport | None = None,
        headers: Mapping[str, str] | None = None,
        auth: httpx2.Auth | None = None,
        timeout: float = 30.0,
        default_wait_budget: float = 30.0,
    ) -> None:
        self.policy = policy
        self.clock: Clock = clock or SystemClock()
        self.default_wait_budget = default_wait_budget
        self._http = httpx2.AsyncClient(
            base_url=base_url,
            transport=transport,
            headers=headers,
            auth=auth,
            timeout=timeout,
        )

    @contextmanager
    def wait_budget(self, seconds: float) -> Iterator[WaitBudget]:
        budget = WaitBudget(seconds)
        token = _active_budget.set(budget)
        try:
            yield budget
        finally:
            _active_budget.reset(token)

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx2.Response:
        """Send a request, retrying per policy. Returns the last response (any status)."""
        method = method.upper()
        idempotent = method in IDEMPOTENT_METHODS
        budget = _active_budget.get() or WaitBudget(self.default_wait_budget)
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._http.request(method, url, **kwargs)
            except httpx2.TransportError:
                delay = self.policy.backoff(None, attempt, self.clock)
                if not idempotent or attempt >= self.policy.max_attempts or not budget.fits(delay):
                    raise
                await self._wait(budget, delay)
                continue

            if not self.policy.should_retry(response):
                return response
            rate_limited = self.policy.is_rate_limited(response)
            if not idempotent and (self.policy.idempotent_only or not rate_limited):
                return response
            delay = parse_retry_after(response, self.clock)
            if delay is None:
                delay = self.policy.backoff(response, attempt, self.clock)
            if attempt >= self.policy.max_attempts or not budget.fits(delay):
                if rate_limited:
                    raise RateLimited(self.policy.name, delay)
                return response
            await self._wait(budget, delay)

    async def _wait(self, budget: WaitBudget, seconds: float) -> None:
        budget.spend(seconds)
        await self.clock.sleep(seconds)

    async def get(self, url: str, **kwargs: Any) -> httpx2.Response:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx2.Response:
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs: Any) -> httpx2.Response:
        return await self.request("PUT", url, **kwargs)

    async def patch(self, url: str, **kwargs: Any) -> httpx2.Response:
        return await self.request("PATCH", url, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> httpx2.Response:
        return await self.request("DELETE", url, **kwargs)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
