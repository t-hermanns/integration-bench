import json
from pathlib import Path
from typing import Any, cast

import httpx2
import pytest
import yaml
from vcr.cassette import Cassette
from vcr.errors import CannotOverwriteExistingCassetteException
from vcr.record_mode import RecordMode
from vcr.request import Request

from bench_core.http import ApiClient, GitHubRetryPolicy, RetryPolicy, SlackRetryPolicy
from bench_core.testing import (
    REDACTED,
    REDACTED_EMAIL,
    FakeBackend,
    FakeClock,
    FakeRequest,
    Fault,
    find_secrets,
    make_vcr,
    redact_request,
    redact_response,
    scrub,
)

BASE = "https://api.example.invalid"
GH_TOKEN = "ghu_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4"
SLACK_TOKEN = "xoxb-" + "1234567890-abcdefghijkl"


def issues_backend() -> FakeBackend:
    backend = FakeBackend()
    backend.issue_token("t-read", ["issues:read"])
    backend.issue_token("t-write", ["issues:read", "issues:write"])
    issues: dict[str, dict[str, Any]] = {"1": {"number": 1, "title": "first"}}

    @backend.route("GET", "/repos/{owner}/{repo}/issues/{number}", scopes=["issues:read"])
    def get_issue(req: FakeRequest) -> Any:
        issue = issues.get(req.params["number"])
        return issue if issue else httpx2.Response(404, json={"message": "Not Found"})

    @backend.route("POST", "/repos/{owner}/{repo}/issues", scopes=["issues:write"])
    def create_issue(req: FakeRequest) -> httpx2.Response:
        number = str(len(issues) + 1)
        issues[number] = {"number": int(number), "title": req.json()["title"]}
        return httpx2.Response(201, json=issues[number])

    return backend


def client(
    backend: FakeBackend, token: str = "t-write", policy: RetryPolicy | None = None
) -> tuple[ApiClient, FakeClock]:
    clock = FakeClock()
    api = ApiClient(
        BASE,
        policy=policy or RetryPolicy(),
        clock=clock,
        transport=backend.transport(),
        headers={"Authorization": f"Bearer {token}"},
    )
    return api, clock


# --- FakeBackend: routing and auth -----------------------------------------------------------


@pytest.mark.anyio
async def test_routes_with_path_params_and_json_body() -> None:
    backend = issues_backend()
    api, _ = client(backend)
    assert (await api.get("/repos/o/r/issues/1")).json() == {"number": 1, "title": "first"}
    created = await api.post("/repos/o/r/issues", json={"title": "second"})
    assert created.status_code == 201
    assert (await api.get("/repos/o/r/issues/2")).json()["title"] == "second"
    assert (await api.get("/repos/o/r/issues/9")).status_code == 404  # handler's own 404
    assert (await api.get("/nope")).status_code == 404  # no route
    assert len(backend.calls("GET", "/repos/{owner}/{repo}/issues/{number}")) == 3
    assert len(backend.calls("POST")) == 1


@pytest.mark.anyio
async def test_unknown_missing_and_expired_tokens_get_401() -> None:
    backend = issues_backend()
    assert (await client(backend, token="bogus")[0].get("/repos/o/r/issues/1")).status_code == 401

    no_auth = ApiClient(BASE, policy=RetryPolicy(), transport=backend.transport())
    assert (await no_auth.get("/repos/o/r/issues/1")).status_code == 401

    api, _ = client(backend, token="t-read")
    assert (await api.get("/repos/o/r/issues/1")).status_code == 200
    backend.expire_token("t-read")
    assert (await api.get("/repos/o/r/issues/1")).status_code == 401


@pytest.mark.anyio
async def test_missing_scope_gets_403_with_missing_scopes() -> None:
    api, _ = client(issues_backend(), token="t-read")
    response = await api.post("/repos/o/r/issues", json={"title": "x"})
    assert response.status_code == 403
    assert response.json()["missing_scopes"] == ["issues:write"]


@pytest.mark.anyio
async def test_require_auth_false_skips_token_checks() -> None:
    backend = FakeBackend(require_auth=False)
    backend.add_route("GET", "/ping", lambda req: {"token": req.token})
    api = ApiClient(BASE, policy=RetryPolicy(), transport=backend.transport())
    assert (await api.get("/ping")).json() == {"token": None}


@pytest.mark.anyio
async def test_basic_auth_token_is_accepted() -> None:
    backend = FakeBackend()
    backend.issue_token(FakeBackend.basic_token("me@example.invalid", "api-token"))
    backend.add_route("GET", "/myself", lambda _: {"ok": True})
    api = ApiClient(
        BASE,
        policy=RetryPolicy(),
        transport=backend.transport(),
        auth=httpx2.BasicAuth("me@example.invalid", "api-token"),
    )
    assert (await api.get("/myself")).status_code == 200


# --- FakeBackend: fault injection ------------------------------------------------------------


@pytest.mark.anyio
async def test_rate_limit_fault_is_retried_after_retry_after() -> None:
    backend = issues_backend()
    backend.inject(Fault.rate_limit(retry_after=5), times=2)
    api, clock = client(backend, policy=SlackRetryPolicy())
    assert (await api.get("/repos/o/r/issues/1")).status_code == 200
    assert clock.sleeps == [5.0, 5.0]
    assert len(backend.requests) == 3


@pytest.mark.anyio
async def test_fault_filters_by_method_and_path() -> None:
    backend = issues_backend()
    backend.inject(Fault.server_error(502), method="POST", path="/repos/{owner}/{repo}/issues")
    api, _ = client(backend)
    assert (await api.get("/repos/o/r/issues/1")).status_code == 200
    assert (await api.post("/repos/o/r/issues", json={"title": "x"})).status_code == 502
    assert (await api.post("/repos/o/r/issues", json={"title": "x"})).status_code == 201


@pytest.mark.anyio
async def test_persistent_fault_until_cleared() -> None:
    backend = issues_backend()
    backend.inject(Fault.forbidden("admin:org"), times=None)
    api, _ = client(backend)
    for _ in range(3):
        response = await api.get("/repos/o/r/issues/1")
        assert response.status_code == 403
        assert response.json()["missing_scopes"] == ["admin:org"]
    backend.clear_faults()
    assert (await api.get("/repos/o/r/issues/1")).status_code == 200


@pytest.mark.anyio
async def test_unauthorized_fault_leaves_token_valid() -> None:
    backend = issues_backend()
    backend.inject(Fault.unauthorized())
    api, _ = client(backend)
    assert (await api.get("/repos/o/r/issues/1")).status_code == 401
    assert (await api.get("/repos/o/r/issues/1")).status_code == 200


@pytest.mark.anyio
async def test_network_fault_raises_and_idempotent_get_is_retried() -> None:
    backend = issues_backend()
    backend.inject(Fault.network())
    api, clock = client(backend, policy=SlackRetryPolicy())
    assert (await api.get("/repos/o/r/issues/1")).status_code == 200
    assert clock.sleeps == [1.0]

    backend.inject(Fault.network())
    with pytest.raises(httpx2.ConnectError):
        await api.post("/repos/o/r/issues", json={"title": "x"})


class GitHubishBackend(FakeBackend):
    def rate_limit_response(
        self, request: httpx2.Request, retry_after: float | None
    ) -> httpx2.Response:
        return httpx2.Response(
            403,
            headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1800000030"},
            json={"message": "API rate limit exceeded"},
        )


@pytest.mark.anyio
async def test_subclass_hook_shapes_provider_specific_rate_limit() -> None:
    backend = GitHubishBackend(require_auth=False)
    backend.add_route("GET", "/rate", lambda _: {"ok": True})
    backend.inject(Fault.rate_limit())
    clock = FakeClock()
    api = ApiClient(
        BASE,
        policy=GitHubRetryPolicy(),
        clock=clock,
        transport=backend.transport(),
        default_wait_budget=60,
    )
    assert (await api.get("/rate")).status_code == 200
    assert clock.sleeps == [31.0]  # until reset + 1 s


def test_fake_clock_advance_does_not_record_a_sleep() -> None:
    clock = FakeClock()
    start = clock.time()
    clock.advance(3600)
    assert clock.time() == start + 3600
    assert clock.sleeps == []


# --- redaction -------------------------------------------------------------------------------


def test_scrub_replaces_token_formats_and_emails() -> None:
    text = (
        f"gh={GH_TOKEN} slack={SLACK_TOKEN} jwt=eyJhbGciOiJI.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF2Q "
        "google=ya29.a0AfH6SMBx3abcdefghijklmnopq mail=Jane.Doe+x@corp.example.com"
    )
    clean = scrub(text)
    assert find_secrets(text) == [
        "github_token",
        "slack_token",
        "google_access_token",
        "jwt",
        "email",
    ]
    assert find_secrets(clean) == []
    assert REDACTED_EMAIL in clean
    assert scrub(clean) == clean


def test_find_secrets_detects_private_keys() -> None:
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----"
    assert find_secrets(pem) == ["private_key"]
    assert scrub(pem) == REDACTED


def test_scrub_replaces_credentials_in_urls_but_not_cursors() -> None:
    body = (
        '{"download_url": "https://raw.githubusercontent.com/o/r/main/a.md?token=CRG3EJ7VOF",'
        ' "next": "https://x.test/search?nextPageToken=EAEY&access_token=at&page=2"}'
    )
    assert find_secrets(body) == ["url_credential"]
    clean = scrub(body)
    assert "a.md?token=REDACTED" in clean
    assert "?nextPageToken=EAEY&access_token=REDACTED&page=2" in clean
    assert find_secrets(clean) == []


def test_redact_request_headers_query_and_form_body() -> None:
    request = Request(
        "POST",
        f"{BASE}/oauth/token?access_token=abc&page=2",
        b"grant_type=refresh_token&refresh_token=rt-secret&client_secret=sec&client_id=cid",
        {
            "Authorization": "Bearer at-secret",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "bench",
        },
    )
    redacted = redact_request(request)
    assert redacted.headers["Authorization"] == REDACTED
    assert redacted.headers["User-Agent"] == "bench"
    assert "access_token=REDACTED" in redacted.uri
    assert "page=2" in redacted.uri
    assert redacted.body == (
        b"grant_type=refresh_token&refresh_token=REDACTED&client_secret=REDACTED&client_id=cid"
    )
    assert request.headers["Authorization"] == "Bearer at-secret"  # original untouched
    assert redact_request(redacted).body == redacted.body  # idempotent


def test_redact_request_json_body_keeps_jira_keys() -> None:
    body = {"key": "BENCH-1", "token": "t", "reporter": {"email": "a.b@corp.example.com"}}
    request = Request("POST", f"{BASE}/x", json.dumps(body).encode(), {})
    assert json.loads(cast(bytes, redact_request(request).body)) == {
        "key": "BENCH-1",
        "token": REDACTED,
        "reporter": {"email": REDACTED_EMAIL},
    }


def _response(body: bytes, content_type: str = "application/json") -> dict[str, Any]:
    return {
        "status": {"code": 200, "message": "OK"},
        "headers": {
            "Content-Type": [content_type],
            "Content-Length": [str(len(body))],
            "Set-Cookie": ["session=abc", "other=def"],
        },
        "body": {"string": body},
    }


def test_redact_response_json_tokens_cookies_and_length() -> None:
    body = json.dumps(
        {
            "ok": True,
            "access_token": SLACK_TOKEN,
            "authed_user": {"access_token": "xoxp-short", "refresh_token": "xoxe-1-abc"},
            "scope": "chat:write",
        }
    ).encode()
    redacted = redact_response(_response(body))
    data = json.loads(redacted["body"]["string"])
    assert data["access_token"] == REDACTED
    assert data["authed_user"] == {"access_token": REDACTED, "refresh_token": REDACTED}
    assert data["scope"] == "chat:write"
    assert redacted["headers"]["Set-Cookie"] == [REDACTED, REDACTED]
    assert redacted["headers"]["Content-Length"] == [str(len(redacted["body"]["string"]))]


def test_redact_response_leaves_binary_and_clean_bodies_alone() -> None:
    binary = b"\x89PNG\r\n\x1a\n\xff\xfe"
    assert redact_response(_response(binary, "image/png"))["body"]["string"] == binary
    clean = b'{"items": [1, 2]}'  # original formatting kept when nothing changes
    assert redact_response(_response(clean))["body"]["string"] == clean


# --- cassettes end to end --------------------------------------------------------------------


@pytest.mark.anyio
async def test_recorded_cassette_is_redacted_and_replays_with_other_token(
    tmp_path: Path,
) -> None:
    cassette = tmp_path / "issues.yaml"
    backend = issues_backend()
    backend.issue_token(GH_TOKEN, ["issues:read", "issues:write"])
    api, _ = client(backend, token=GH_TOKEN)
    with make_vcr(record_mode=RecordMode.ONCE).use_cassette(str(cassette)):
        assert (await api.get("/repos/o/r/issues/1")).status_code == 200
        created = await api.post(
            "/repos/o/r/issues", json={"title": "new", "body": "by dev@corp.example.com"}
        )
        assert created.status_code == 201

    text = cassette.read_text()
    assert GH_TOKEN not in text
    assert find_secrets(text) == []
    recorded = yaml.safe_load(text)["interactions"]
    assert recorded[0]["request"]["headers"]["authorization"] == [REDACTED]

    # Replay with a different real-looking token, a fake that would 401 everything, and
    # the JSON keys in another order: the redacted request still matches.
    replay_api, _ = client(FakeBackend(), token="ghu_" + "Z" * 30)
    with make_vcr(record_mode=RecordMode.NONE).use_cassette(str(cassette)) as cass:
        assert (await replay_api.get("/repos/o/r/issues/1")).json()["title"] == "first"
        replayed = await replay_api.post(
            "/repos/o/r/issues",
            content=json.dumps({"body": "by dev@corp.example.com", "title": "new"}),
            headers={"Content-Type": "application/json"},
        )
        assert replayed.status_code == 201
        assert cass.all_played


@pytest.mark.anyio
async def test_replay_rejects_different_json_body(tmp_path: Path) -> None:
    cassette = tmp_path / "post.yaml"
    api, _ = client(issues_backend())
    with make_vcr(record_mode=RecordMode.ONCE).use_cassette(str(cassette)):
        await api.post("/repos/o/r/issues", json={"title": "a"})
    with (
        make_vcr(record_mode=RecordMode.NONE).use_cassette(str(cassette)),
        pytest.raises(CannotOverwriteExistingCassetteException),
    ):
        await api.post("/repos/o/r/issues", json={"title": "b"})


@pytest.fixture
def vcr_cassette_dir(tmp_path: Path) -> str:
    return str(tmp_path)


@pytest.fixture
def record_mode() -> str:
    return "once"


@pytest.mark.anyio
@pytest.mark.vcr
async def test_pytest_recording_uses_repo_config(vcr: Cassette) -> None:
    api, _ = client(issues_backend())
    assert (await api.get("/repos/o/r/issues/1")).status_code == 200
    assert vcr.requests[0].headers["Authorization"] == REDACTED
