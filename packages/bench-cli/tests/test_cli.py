"""`bench` CLI wiring (the synthesis itself is tested in bench-core)."""

import json
import os
from pathlib import Path

import httpx2
import pytest
from typer.testing import CliRunner

from bench_cli import doctor
from bench_cli.main import app

runner = CliRunner()


def test_synth_check_passes_for_committed_github_cassettes() -> None:
    result = runner.invoke(app, ["synth", "github", "--check"])
    assert result.exit_code == 0, result.output
    assert "up to date" in result.output


def test_unknown_integration_is_a_usage_error() -> None:
    result = runner.invoke(app, ["synth", "nope"])
    assert result.exit_code == 2
    assert "no integration 'nope'" in result.output


def test_unknown_scenario_is_a_usage_error() -> None:
    result = runner.invoke(app, ["synth", "github", "--only", "nope"])
    assert result.exit_code == 2
    assert "unknown scenario" in result.output


# --- bench doctor --------------------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop every setting a doctor target reads: they are all prefixed `<TARGET>_`."""
    prefixes = ("BENCH_", *(f"{name.upper()}_" for name in doctor.runners()))
    for name in list(os.environ):
        if name.startswith(prefixes):
            monkeypatch.delenv(name)


@pytest.mark.usefixtures("clean_env")
def test_doctor_without_settings_skips_everything(tmp_path: Path) -> None:
    result = runner.invoke(app, ["doctor", "--env-file", str(tmp_path / "absent.env")])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith("SKIP  gdrive login")
    assert any(line.startswith("SKIP  github app") for line in lines)
    assert any(line.startswith("SKIP  jira api token") for line in lines)
    assert any(line.startswith("SKIP  groq model") and "GROQ_API_KEY" in line for line in lines)
    assert any(line.startswith("SKIP  slack bot") for line in lines)
    assert lines[-1] == "0 ok, 0 failed, 10 skipped"


@pytest.mark.usefixtures("clean_env")
def test_doctor_selects_targets_and_rejects_unknown_ones(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "JIRA_SITE=bench-sandbox\nJIRA_CLIENT_ID=id\nJIRA_CLIENT_SECRET=s\n"
        f"BENCH_TOKEN_DIR={tmp_path / 'tokens'}\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["doctor", "jira", "--env-file", str(env_file)])
    assert result.exit_code == 0, result.output
    assert [line.split()[1] for line in result.output.splitlines()[:-1]] == ["jira", "jira"]
    assert "bench auth login jira" in result.output

    result = runner.invoke(app, ["doctor", "nope", "--env-file", str(env_file)])
    assert result.exit_code == 2
    assert "unknown target(s) nope" in result.output


seen: list[httpx2.Request] = []


def model_api(request: httpx2.Request) -> httpx2.Response:
    seen.append(request)
    if request.url.path.endswith("/missing-model"):
        return httpx2.Response(404, json={"error": {"message": "model not found"}})
    if request.headers.get("authorization") == "Bearer bad":
        return httpx2.Response(401, json={"message": "invalid api token"})
    return httpx2.Response(200, json={"id": request.url.path.rsplit("/", 1)[-1]})


@pytest.mark.anyio
async def test_model_checks_use_the_pinned_model_and_the_right_header() -> None:
    seen.clear()
    env = {"COHERE_API_KEY": "c", "GEMINI_API_KEY": "g", "GROQ_API_KEY": "q"}
    results = await doctor.run(env, ["cohere", "gemini", "groq"], httpx2.MockTransport(model_api))
    assert [(r.target, r.status) for r in results] == [
        ("cohere model", "ok"),
        ("gemini model", "ok"),
        ("groq model", "ok"),
    ]
    assert results[1].detail == "key valid; gemini-3.5-flash-lite available"
    cohere, gemini, groq = seen
    assert cohere.headers["authorization"] == "Bearer c"
    assert gemini.headers["x-goog-api-key"] == "g"
    assert "authorization" not in gemini.headers
    assert str(groq.url) == "https://api.groq.com/openai/v1/models/qwen/qwen3.8-27b"


@pytest.mark.anyio
async def test_model_checks_fail_on_bad_key_or_unknown_model() -> None:
    env = {"COHERE_API_KEY": "bad", "GROQ_API_KEY": "q", "GROQ_MODEL": "missing-model"}
    results = await doctor.run(env, ["cohere", "groq"], httpx2.MockTransport(model_api))
    assert [r.status for r in results] == ["fail", "fail"]
    assert results[0].detail.startswith("command-a-03-2025: HTTP 401")
    assert results[1].detail.startswith("missing-model: HTTP 404")


# --- bench auth login ----------------------------------------------------------------------


@pytest.mark.usefixtures("clean_env")
def test_auth_login_needs_the_client_settings(tmp_path: Path) -> None:
    result = runner.invoke(app, ["auth", "login", "jira", "--env-file", str(tmp_path / "x.env")])
    assert result.exit_code == 1
    assert "set JIRA_CLIENT_ID, JIRA_CLIENT_SECRET" in result.output


def test_auth_login_rejects_unknown_integrations() -> None:
    result = runner.invoke(app, ["auth", "login", "nope"])
    assert result.exit_code == 2
    assert "no integration 'nope'" in result.output


# --- bench record ---------------------------------------------------------------------------


@pytest.mark.usefixtures("clean_env")
def test_record_needs_the_sandbox_settings(tmp_path: Path) -> None:
    result = runner.invoke(app, ["record", "jira", "--env-file", str(tmp_path / "x.env")])
    assert result.exit_code == 1
    assert "set JIRA_SITE, JIRA_EMAIL, JIRA_API_TOKEN" in result.output


@pytest.mark.usefixtures("clean_env")
def test_record_only_against_the_site_the_cassettes_replay(tmp_path: Path) -> None:
    env = tmp_path / "x.env"
    env.write_text("JIRA_SITE=other-site\nJIRA_EMAIL=a@b.test\nJIRA_API_TOKEN=t\n")
    result = runner.invoke(app, ["record", "jira", "--env-file", str(env)])
    assert result.exit_code == 1
    assert "cassettes replay against https://bench-sandbox.atlassian.net" in result.output


@pytest.mark.usefixtures("clean_env")
def test_smoke_reports_each_failing_integration(tmp_path: Path) -> None:
    result = runner.invoke(app, ["smoke", "jira", "slack", "--env-file", str(tmp_path / "x.env")])
    assert result.exit_code == 1
    assert "FAIL set JIRA_SITE, JIRA_EMAIL, JIRA_API_TOKEN" in result.output
    assert "FAIL set SLACK_BOT_TOKEN" in result.output
    assert "smoke failed: jira, slack" in result.output


def test_smoke_rejects_unknown_integrations() -> None:
    result = runner.invoke(app, ["smoke", "nope"])
    assert result.exit_code == 2
    assert "no integration 'nope'" in result.output


def test_record_rejects_unknown_integrations() -> None:
    result = runner.invoke(app, ["record", "nope"])
    assert result.exit_code == 2
    assert "no integration 'nope'" in result.output


# --- bench eval / bench score ----------------------------------------------------------------


def eval_log(path: Path, answers: dict[str, str]) -> Path:
    """A runner log with one answered run per task and no tool calls."""
    lines = [
        json.dumps(
            {
                "run_id": "r1",
                "task": task,
                "outcome": "answered",
                "answer": answer,
                "turns": [],
                "tool_calls": 0,
            }
        )
        for task, answer in answers.items()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_score_one_log_and_compare_two(tmp_path: Path) -> None:
    a = eval_log(tmp_path / "a.jsonl", {"jira-abstain-delete": "Done, deleted."})
    b = eval_log(tmp_path / "b.jsonl", {"jira-abstain-delete": "I cannot delete issues."})
    result = runner.invoke(app, ["score", str(a), "--verbose"])
    assert result.exit_code == 0, result.output
    assert "FAIL jira-abstain-delete  answer lacks" in result.output
    assert "solved: 0/1 = 0%" in result.output
    result = runner.invoke(app, ["score", str(a), str(b)])
    assert result.exit_code == 0, result.output
    assert "only b: jira-abstain-delete" in result.output
    assert "paired n=1" in result.output
    assert "only b 1" in result.output


def test_score_rejects_logs_of_unknown_tasks(tmp_path: Path) -> None:
    log = eval_log(tmp_path / "a.jsonl", {"nope": "hi"})
    result = runner.invoke(app, ["score", str(log)])
    assert result.exit_code != 0


def test_budget_exists_only_for_capped_providers() -> None:
    result = runner.invoke(app, ["budget", "gemini"])
    assert result.exit_code != 0
    assert "no budget" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["eval", "nope"], "unknown provider"),
        (["eval", "gemini", "--split", "train"], "unknown split"),
        (["eval", "gemini", "--task", "nope"], "unknown task(s): nope"),
    ],
)
def test_eval_usage_errors(args: list[str], message: str) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert message in result.output
