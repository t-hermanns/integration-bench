"""Per-provider retry policies.

A policy only answers questions about a single response; the retry loop, `Retry-After`
precedence, idempotency and the wait budget live in `ApiClient`.
"""

from dataclasses import dataclass
from email.utils import parsedate_to_datetime

import httpx2

from bench_core.clock import Clock

IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})


def parse_retry_after(response: httpx2.Response, clock: Clock) -> float | None:
    """Seconds to wait per `Retry-After` (delta-seconds or HTTP-date), or None if absent/invalid."""
    value = response.headers.get("retry-after")
    if value is None:
        return None
    value = value.strip()
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    try:
        return max(parsedate_to_datetime(value).timestamp() - clock.time(), 0.0)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class RetryPolicy:
    """Base policy: never retries. Subclasses override the three hooks."""

    name: str = "generic"
    """Provider name in messages to the model; integrations pass their own, e.g. "GitHub"."""
    max_attempts: int = 1
    # Retry only idempotent methods, even on rate limits (Jira's documented guidance).
    idempotent_only: bool = False

    def should_retry(self, response: httpx2.Response) -> bool:
        return False

    def is_rate_limited(self, response: httpx2.Response) -> bool:
        """A rate-limited request was not executed, so retrying a POST is safe."""
        return response.status_code == 429

    def backoff(self, response: httpx2.Response | None, attempt: int, clock: Clock) -> float:
        """Delay before the next attempt when the response carries no `Retry-After`.

        `attempt` is the number of attempts made so far (1 after the first failure);
        `response` is None after a transport error.
        """
        return float(2 ** (attempt - 1))


@dataclass(frozen=True)
class JiraRetryPolicy(RetryPolicy):
    """2-5 s initial delay, doubling up to 30 s, +/-30 % jitter, idempotent calls only."""

    name: str = "jira"
    max_attempts: int = 4
    idempotent_only: bool = True

    def should_retry(self, response: httpx2.Response) -> bool:
        return response.status_code in (429, 503)

    def is_rate_limited(self, response: httpx2.Response) -> bool:
        return response.status_code in (429, 503)

    def backoff(self, response: httpx2.Response | None, attempt: int, clock: Clock) -> float:
        initial = 2.0 + 3.0 * clock.random()
        capped = min(initial * 2 ** (attempt - 1), 30.0)
        return capped * (0.7 + 0.6 * clock.random())


@dataclass(frozen=True)
class GitHubRetryPolicy(RetryPolicy):
    """Primary limit: wait for `x-ratelimit-reset`. Secondary limit: at least 60 s, doubling."""

    name: str = "github"
    max_attempts: int = 3

    def should_retry(self, response: httpx2.Response) -> bool:
        return self.is_rate_limited(response)

    def is_rate_limited(self, response: httpx2.Response) -> bool:
        if response.status_code == 429:
            return True
        if response.status_code != 403:
            return False
        # A 403 is a permission error unless GitHub marks it as a (secondary) rate limit.
        return (
            response.headers.get("x-ratelimit-remaining") == "0"
            or "retry-after" in response.headers
            or "rate limit" in response.text.lower()
        )

    def backoff(self, response: httpx2.Response | None, attempt: int, clock: Clock) -> float:
        if response is None:
            return super().backoff(response, attempt, clock)
        reset = response.headers.get("x-ratelimit-reset")
        if response.headers.get("x-ratelimit-remaining") == "0" and reset is not None:
            return max(float(reset) - clock.time(), 0.0) + 1.0
        return 60.0 * 2 ** (attempt - 1)


_DRIVE_RATE_LIMIT_REASONS = ("ratelimitexceeded", "userratelimitexceeded", "rate_limit_exceeded")


@dataclass(frozen=True)
class DriveRetryPolicy(RetryPolicy):
    """`min(2^n + random, 64 s)` on 429, 5xx and 403 with a rate-limit `reason`."""

    name: str = "gdrive"
    max_attempts: int = 5

    def should_retry(self, response: httpx2.Response) -> bool:
        return self.is_rate_limited(response) or response.status_code in (500, 502, 503, 504)

    def is_rate_limited(self, response: httpx2.Response) -> bool:
        if response.status_code == 429:
            return True
        if response.status_code != 403:
            return False
        body = response.text.lower()
        return any(reason in body for reason in _DRIVE_RATE_LIMIT_REASONS)

    def backoff(self, response: httpx2.Response | None, attempt: int, clock: Clock) -> float:
        return min(2 ** (attempt - 1) + clock.random(), 64.0)


@dataclass(frozen=True)
class SlackRetryPolicy(RetryPolicy):
    """Slack signals rate limits with 429 + `Retry-After`; other errors are 200 + `ok: false`."""

    name: str = "slack"
    max_attempts: int = 3

    def should_retry(self, response: httpx2.Response) -> bool:
        return response.status_code == 429

    def backoff(self, response: httpx2.Response | None, attempt: int, clock: Clock) -> float:
        return min(float(2 ** (attempt - 1)), 30.0)
