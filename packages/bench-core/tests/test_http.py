import httpx2
import pytest

from bench_core.http import (
    ApiClient,
    DriveRetryPolicy,
    GitHubRetryPolicy,
    JiraRetryPolicy,
    RateLimited,
    RetryPolicy,
    SlackRetryPolicy,
    parse_retry_after,
)
from bench_core.testing import FakeClock

pytestmark = pytest.mark.anyio

Step = httpx2.Response | Exception


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class Scripted:
    """MockTransport handler that replays a fixed sequence of responses/exceptions."""

    def __init__(self, *steps: Step) -> None:
        self.steps = list(steps)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def make_client(
    policy: RetryPolicy, *steps: Step, clock: FakeClock | None = None, budget: float = 30.0
) -> tuple[ApiClient, Scripted, FakeClock]:
    clock = clock or FakeClock()
    script = Scripted(*steps)
    api = ApiClient(
        "https://api.example.invalid",
        policy=policy,
        clock=clock,
        transport=httpx2.MockTransport(script),
        default_wait_budget=budget,
    )
    return api, script, clock


def ok() -> httpx2.Response:
    return httpx2.Response(200, json={"ok": True})


def r429(retry_after: str | None = None) -> httpx2.Response:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return httpx2.Response(429, headers=headers)


# --- Retry-After parsing ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("7", 7.0),
        ("0", 0.0),
        ("-3", 0.0),
        # FakeClock starts at 1_800_000_000 = Fri, 15 Jan 2027 08:00:00 GMT
        ("Fri, 15 Jan 2027 08:00:42 GMT", 42.0),
        ("Fri, 15 Jan 2027 07:00:00 GMT", 0.0),
        ("soon", None),
        (None, None),
    ],
)
def test_parse_retry_after(value: str | None, expected: float | None) -> None:
    headers = {"Retry-After": value} if value is not None else {}
    assert parse_retry_after(httpx2.Response(429, headers=headers), FakeClock()) == expected


# --- Generic behaviour --------------------------------------------------------------------


async def test_success_is_returned_without_waiting() -> None:
    api, script, clock = make_client(JiraRetryPolicy(), ok())
    async with api:
        assert (await api.get("/x")).status_code == 200
    assert len(script.requests) == 1
    assert clock.sleeps == []


async def test_base_policy_never_retries() -> None:
    api, _, clock = make_client(RetryPolicy(), r429("1"))
    assert (await api.get("/x")).status_code == 429
    assert clock.sleeps == []


async def test_retry_after_takes_precedence_over_backoff() -> None:
    api, script, clock = make_client(JiraRetryPolicy(), r429("7"), ok())
    assert (await api.get("/x")).status_code == 200
    assert clock.sleeps == [7.0]
    assert len(script.requests) == 2


async def test_transport_error_retried_for_idempotent_requests() -> None:
    api, _, clock = make_client(SlackRetryPolicy(), httpx2.ConnectError("boom"), ok())
    assert (await api.get("/x")).status_code == 200
    assert clock.sleeps == [1.0]


async def test_transport_error_not_retried_for_post() -> None:
    api, script, _ = make_client(SlackRetryPolicy(), httpx2.ConnectError("boom"), ok())
    with pytest.raises(httpx2.ConnectError):
        await api.post("/x")
    assert len(script.requests) == 1


# --- Wait budget --------------------------------------------------------------------------


async def test_wait_longer_than_budget_raises_rate_limited_without_sleeping() -> None:
    api, script, clock = make_client(SlackRetryPolicy(), r429("45"), ok(), budget=30)
    with pytest.raises(RateLimited, match=r"slack rate limited, retry after 45 s") as info:
        await api.get("/x")
    assert info.value.retry_after == 45.0
    assert clock.sleeps == []
    assert len(script.requests) == 1


async def test_budget_is_shared_across_requests_of_one_tool_call() -> None:
    api, _, clock = make_client(SlackRetryPolicy(), r429("7"), ok(), r429("7"), ok())
    with api.wait_budget(10) as budget:
        await api.get("/page1")
        assert budget.remaining == 3.0
        with pytest.raises(RateLimited):
            await api.get("/page2")
    assert clock.sleeps == [7.0]


async def test_without_open_budget_each_request_gets_the_default() -> None:
    api, _, clock = make_client(SlackRetryPolicy(), r429("20"), ok(), r429("20"), ok(), budget=25)
    await api.get("/a")
    await api.get("/b")
    assert clock.sleeps == [20.0, 20.0]


async def test_rate_limited_is_a_tool_error() -> None:
    from mcp.server.mcpserver.exceptions import ToolError

    assert issubclass(RateLimited, ToolError)


# --- Jira ---------------------------------------------------------------------------------


async def test_jira_backoff_doubles_with_jitter_and_caps_attempts() -> None:
    # rand=0.5 -> initial 3.5 s, jitter factor 1.0
    api, script, clock = make_client(JiraRetryPolicy(), *[r429()] * 4, budget=1000)
    with pytest.raises(RateLimited, match="retry after 28 s"):
        await api.get("/rest/api/3/search/jql")
    assert clock.sleeps == [3.5, 7.0, 14.0]
    assert len(script.requests) == 4


@pytest.mark.parametrize(("rand", "expected"), [(0.0, 1.4), (0.999, 6.5)])
def test_jira_backoff_jitter_bounds(rand: float, expected: float) -> None:
    delay = JiraRetryPolicy().backoff(None, 1, FakeClock(rand=rand))
    assert delay == pytest.approx(expected, abs=0.01)


def test_jira_backoff_caps_at_30_s_before_jitter() -> None:
    assert JiraRetryPolicy().backoff(None, 10, FakeClock(rand=0.5)) == pytest.approx(30.0)


async def test_jira_503_is_retried() -> None:
    api, _, clock = make_client(JiraRetryPolicy(), httpx2.Response(503), ok())
    assert (await api.get("/x")).status_code == 200
    assert clock.sleeps == [3.5]


async def test_jira_post_is_never_retried() -> None:
    api, script, clock = make_client(JiraRetryPolicy(), r429("1"), ok())
    assert (await api.post("/rest/api/3/issue", json={})).status_code == 429
    assert len(script.requests) == 1
    assert clock.sleeps == []


# --- GitHub -------------------------------------------------------------------------------


async def test_github_primary_limit_waits_until_reset() -> None:
    clock = FakeClock()
    exhausted = httpx2.Response(
        403,
        headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(clock.now) + 10)},
        json={"message": "API rate limit exceeded"},
    )
    api, _, _ = make_client(GitHubRetryPolicy(), exhausted, ok(), clock=clock)
    assert (await api.get("/repos/o/r/issues")).status_code == 200
    assert clock.sleeps == [11.0]


async def test_github_secondary_limit_waits_at_least_60_s() -> None:
    secondary = httpx2.Response(403, json={"message": "You have exceeded a secondary rate limit."})
    api, _, clock = make_client(GitHubRetryPolicy(), secondary, ok(), budget=30)
    with pytest.raises(RateLimited, match="github rate limited, retry after 60 s"):
        await api.get("/search/issues")
    assert clock.sleeps == []

    api, _, clock = make_client(GitHubRetryPolicy(), secondary, ok(), budget=120)
    assert (await api.get("/search/issues")).status_code == 200
    assert clock.sleeps == [60.0]


async def test_github_permission_403_is_not_retried() -> None:
    denied = httpx2.Response(403, json={"message": "Resource not accessible by integration"})
    api, script, _ = make_client(GitHubRetryPolicy(), denied)
    assert (await api.get("/repos/o/r")).status_code == 403
    assert len(script.requests) == 1


async def test_github_post_retried_on_rate_limit() -> None:
    api, script, _ = make_client(GitHubRetryPolicy(), r429("2"), httpx2.Response(201))
    assert (await api.post("/repos/o/r/issues", json={"title": "t"})).status_code == 201
    assert len(script.requests) == 2


# --- Drive --------------------------------------------------------------------------------


def drive_403(reason: str) -> httpx2.Response:
    return httpx2.Response(
        403, json={"error": {"code": 403, "errors": [{"reason": reason}], "message": reason}}
    )


async def test_drive_rate_limit_403_backs_off_exponentially() -> None:
    api, _, clock = make_client(
        DriveRetryPolicy(), drive_403("userRateLimitExceeded"), drive_403("rateLimitExceeded"), ok()
    )
    assert (await api.get("/drive/v3/files")).status_code == 200
    assert clock.sleeps == [1.5, 2.5]


def test_drive_backoff_caps_at_64_s() -> None:
    assert DriveRetryPolicy().backoff(None, 10, FakeClock()) == 64.0


async def test_drive_permission_403_is_not_retried() -> None:
    api, script, _ = make_client(DriveRetryPolicy(), drive_403("insufficientFilePermissions"))
    assert (await api.get("/drive/v3/files/x")).status_code == 403
    assert len(script.requests) == 1


async def test_drive_5xx_retried_for_get_but_not_for_post() -> None:
    api, _, clock = make_client(DriveRetryPolicy(), httpx2.Response(503), ok())
    assert (await api.get("/drive/v3/files")).status_code == 200
    assert clock.sleeps == [1.5]

    api, script, _ = make_client(DriveRetryPolicy(), httpx2.Response(503), ok())
    assert (await api.post("/drive/v3/files", json={})).status_code == 503
    assert len(script.requests) == 1


async def test_drive_5xx_exhausted_returns_last_response() -> None:
    api, script, _ = make_client(DriveRetryPolicy(), *[httpx2.Response(500)] * 5, budget=1000)
    assert (await api.get("/drive/v3/files")).status_code == 500
    assert len(script.requests) == 5


# --- Slack --------------------------------------------------------------------------------


async def test_slack_429_honours_retry_after() -> None:
    api, _, clock = make_client(SlackRetryPolicy(), r429("2"), ok())
    assert (await api.post("/api/chat.postMessage", json={})).status_code == 200
    assert clock.sleeps == [2.0]


async def test_slack_ok_false_is_not_retried() -> None:
    error = httpx2.Response(200, json={"ok": False, "error": "channel_not_found"})
    api, script, _ = make_client(SlackRetryPolicy(), error)
    assert (await api.get("/api/conversations.history")).json()["error"] == "channel_not_found"
    assert len(script.requests) == 1
