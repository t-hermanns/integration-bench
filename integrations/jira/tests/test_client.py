"""Client pieces that cassettes do not cover: validation, URLs and error translation."""

import httpx2
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from bench_core.permissions import NotFoundOrNoAccess, PermissionDenied

from bench_jira.client import (
    error_text,
    gateway_api_url,
    issue_key,
    project_key,
    raise_for_status,
    site_api_url,
)

CLOUD_ID = "11223344-a1b2-3b33-c444-def123456789"


def response(status: int, body: object = None, *, text: str | None = None) -> httpx2.Response:
    if text is not None:
        return httpx2.Response(status, text=text)
    return httpx2.Response(status, json=body)


def test_base_urls() -> None:
    assert site_api_url("bench-sandbox") == "https://bench-sandbox.atlassian.net/rest/api/3"
    assert gateway_api_url(CLOUD_ID) == f"https://api.atlassian.com/ex/jira/{CLOUD_ID}/rest/api/3"
    for bad in ("evil.com/x", "Bench", ""):
        with pytest.raises(ValueError, match="site name"):
            site_api_url(bad)
    with pytest.raises(ValueError, match="cloud id"):
        gateway_api_url("../../admin")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("BENCH-12", "BENCH-12"), (" bench-3 ", "BENCH-3"), ("10042", "10042"), ("AB_2-1", "AB_2-1")],
)
def test_issue_keys(value: str, expected: str) -> None:
    assert issue_key(value) == expected


@pytest.mark.parametrize("value", ["BENCH", "BENCH-0", "BENCH-1/comment", "../x", "B-1", "1-2"])
def test_invalid_issue_keys(value: str) -> None:
    with pytest.raises(ToolError, match="Invalid issue key"):
        issue_key(value)


def test_project_keys() -> None:
    assert project_key("bench") == "BENCH"
    with pytest.raises(ToolError, match="Invalid project key"):
        project_key("BENCH-1")


def test_error_text_formats() -> None:
    jira = response(400, {"errorMessages": ["Bad JQL"], "errors": {"summary": "required"}})
    assert error_text(jira) == "Bad JQL; summary: required"
    gateway = response(401, {"code": 401, "message": "Unauthorized"})
    assert error_text(gateway) == "Unauthorized"
    assert error_text(response(502, text="<html>Bad gateway</html>")) == "Bad Gateway"


def test_status_translation() -> None:
    def check(r: httpx2.Response) -> None:
        raise_for_status(r, action="read the issue", what="Issue BENCH-1")

    check(response(200, {}))
    check(response(204, text=""))
    scope = response(401, {"code": 401, "message": "Unauthorized; scope does not match"})
    with pytest.raises(PermissionDenied, match="OAuth scope"):
        check(scope)
    with pytest.raises(ToolError, match=r"rejected the credentials.*sign in again"):
        check(response(401, {"code": 401, "message": "Unauthorized"}))
    forbidden = response(403, {"errorMessages": ["You cannot edit this issue"], "errors": {}})
    with pytest.raises(PermissionDenied, match="You cannot edit this issue"):
        check(forbidden)
    with pytest.raises(NotFoundOrNoAccess, match="Issue BENCH-1 was not found"):
        check(response(404, {"errorMessages": ["Issue does not exist"], "errors": {}}))
    with pytest.raises(ToolError, match=r"could not read the issue: busy\."):
        check(response(409, {"errorMessages": ["busy"]}))
    with pytest.raises(ToolError, match=r"server error \(500\).*Try again later"):
        check(response(500, {}))
    with pytest.raises(ToolError, match="returned 413"):
        check(response(413, {"errorMessages": ["too big"]}))
