"""`bench_core.doctor`: settings files and turning errors into failed checks."""

from pathlib import Path

import httpx2
import pytest

from bench_core.auth import AuthError, OAuthError
from bench_core.doctor import (
    Check,
    http_failure,
    load_env,
    missing,
    ok,
    parse_env_file,
    probe,
)
from bench_core.http import RateLimited

pytestmark = pytest.mark.anyio


def test_parse_env_file() -> None:
    text = """
# comment
JIRA_SITE=bench-sandbox          # the part before .atlassian.net
export GITHUB_ORG = bench-sandbox
QUOTED="value # kept"
SINGLE='x'
EMPTY=
URL=https://example.com/a#fragment
"""
    assert parse_env_file(text) == {
        "JIRA_SITE": "bench-sandbox",
        "GITHUB_ORG": "bench-sandbox",
        "QUOTED": "value # kept",
        "SINGLE": "x",
        "EMPTY": "",
        "URL": "https://example.com/a#fragment",
    }


@pytest.mark.parametrize("line", ["just text", "=value", "BAD KEY=1"])
def test_parse_env_file_rejects_other_lines(line: str) -> None:
    with pytest.raises(ValueError, match="KEY=value"):
        parse_env_file(line)


def test_process_environment_wins_over_the_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("A=file\nB=file\nC=file\n", encoding="utf-8")
    env = load_env(env_file, {"A": "process", "C": ""})
    assert env == {"A": "process", "B": "file", "C": "file"}
    assert load_env(tmp_path / "absent", {"A": "x"}) == {"A": "x"}


def test_missing_counts_empty_values() -> None:
    assert missing({"A": "1", "B": ""}, "A", "B", "C") == ["B", "C"]


async def _raise(exc: Exception) -> Check:
    raise exc


@pytest.mark.parametrize(
    ("exc", "detail"),
    [
        (AuthError("acct", "the server rejected the static token (401)", "Check it."), "401"),
        (OAuthError("invalid_client", "bad secret"), "invalid_client"),
        (RateLimited("Jira", 120.0), "Jira"),
        (httpx2.ConnectError("no route"), "ConnectError: no route"),
        (ValueError("bad PEM"), "invalid setting: bad PEM"),
    ],
)
async def test_probe_turns_known_errors_into_failures(exc: Exception, detail: str) -> None:
    check = await probe("target", _raise(exc))
    assert check.status == "fail"
    assert detail in check.detail


async def test_probe_passes_results_and_unexpected_errors_through() -> None:
    async def fine() -> Check:
        return ok("target", "fine")

    assert (await probe("target", fine())).status == "ok"
    with pytest.raises(KeyError):
        await probe("target", _raise(KeyError("bug")))


def test_http_failure_shows_status_and_start_of_body() -> None:
    response = httpx2.Response(403, text="  Forbidden:\n  " + "x" * 500)
    check = http_failure("t", response, "repo a/b")
    assert check.status == "fail"
    assert check.detail.startswith("repo a/b: HTTP 403 Forbidden: xxx")
    assert len(check.detail) < 240
