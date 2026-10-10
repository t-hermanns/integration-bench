import json
from collections.abc import Sequence

import httpx2
import pytest

from bench_core.testing import FakeClock

from bench_evals.models import (
    Assistant,
    ModelError,
    Reply,
    Request,
    ScriptedModel,
    ToolCall,
    ToolResult,
    Usage,
)
from bench_evals.runner import Limit, Limiter, Task, open_toolbox, run_task, run_tasks
from bench_github.server import create_server as github_server
from bench_jira.fake import credential, seeded_fake
from bench_jira.scopes import READ, WRITE
from bench_jira.server import create_server as jira_server

pytestmark = pytest.mark.anyio

TOKEN = "runner-token"
TASK = Task("jira-read-1", ("jira",), "What is the summary of BENCH-1?")


def servers():
    fake = seeded_fake(FakeClock())
    fake.issue_token(TOKEN, [READ, WRITE])
    jira = jira_server(
        credential=credential(TOKEN),
        transport=fake.transport(),
        clock=FakeClock(),
        variant="v1",
        read_only=False,
    )
    # Only listed, never called: a task for Jira must not see GitHub's tools.
    github = github_server(
        transport=httpx2.MockTransport(lambda r: httpx2.Response(500)), read_only=False
    )
    return {"jira": jira, "github": github}


def call(name: str, **arguments) -> Assistant:
    return Assistant(tool_calls=(ToolCall(f"c-{name}", name, arguments),))


def last_result(request: Request) -> ToolResult:
    result = request.messages[-1]
    assert isinstance(result, ToolResult)
    return result


async def test_limiter_spaces_requests_by_rpm():
    clock = FakeClock()
    limiter = Limiter(Limit(rpm=2), clock)
    assert await limiter.wait(0) == 0
    clock.advance(10)
    assert await limiter.wait(0) == 0
    assert await limiter.wait(0) == 50  # until the first request leaves the window
    assert clock.sleeps == [50]


async def test_limiter_waits_for_enough_tokens_using_the_settled_usage():
    clock = FakeClock()
    limiter = Limiter(Limit(rpm=100, tpm=1000), clock)
    await limiter.wait(100)
    limiter.settle(600)  # the estimate was low
    clock.advance(5)
    await limiter.wait(300)
    clock.advance(5)
    assert limiter.delay(100) == 0  # 600 + 300 + 100 fits exactly
    assert limiter.delay(101) == 50  # the 600 must expire first
    assert await limiter.wait(500) == 50
    assert [t for _, t in limiter.window] == [300, 500]


async def test_limiter_lets_an_oversized_request_through_on_an_empty_window():
    clock = FakeClock()
    limiter = Limiter(Limit(rpm=100, tpm=1000), clock)
    await limiter.wait(10)
    assert await limiter.wait(5000) == 60
    assert limiter.delay(0) == 60  # and now fills the window itself


async def test_run_tasks_sends_only_the_tasks_tools_and_logs_one_line(tmp_path):
    def answer(request: Request) -> Assistant:
        issue = json.loads(last_result(request).content)
        return Assistant(text=f"BENCH-1: {issue['summary']}")

    model = ScriptedModel([call("jira_get_issue", issue_key="BENCH-1"), answer])
    out = tmp_path / "runs" / "run.jsonl"
    records = await run_tasks(model, [TASK], servers(), out, run_id="r1")

    tools = {t.name for t in model.requests[0].tools}
    assert tools
    assert all(name.startswith("jira_") for name in tools)
    assert "jira" in model.requests[0].system
    (record,) = records
    assert record["run_id"] == "r1"
    assert record["outcome"] == "answered"
    assert record["answer"].startswith("BENCH-1: ")
    assert record["tool_calls"] == 1
    assert record["turns"][0]["results"][0]["is_error"] is False
    assert json.loads(out.read_bytes().decode().strip()) == record


async def test_tool_errors_and_foreign_tools_go_back_to_the_model():
    model = ScriptedModel(
        [
            call("github_search_issues", query="x"),
            call("jira_get_issue", issue_key="NOPE-999"),
            Assistant(text="not found"),
        ]
    )
    async with open_toolbox(servers()) as toolbox:
        record = await run_task(model, TASK, toolbox)
    unknown, missing = (t["results"][0] for t in record["turns"][:2])
    assert unknown == {
        "name": "github_search_issues",
        "is_error": True,
        "content": "Unknown tool: github_search_issues",
    }
    assert missing["is_error"] is True
    assert missing["content"]
    assert last_result(model.requests[1]).is_error
    assert record["outcome"] == "answered"


async def test_a_model_that_never_stops_calling_tools_is_cut_off():
    model = ScriptedModel([call("jira_get_issue", issue_key="BENCH-1")] * 3)
    async with open_toolbox(servers()) as toolbox:
        record = await run_task(model, TASK, toolbox, max_turns=3)
    assert record["outcome"] == "max_turns"
    assert record["tool_calls"] == 3
    assert record["answer"] == ""


class Flaky:
    """Fails with the given errors first, then answers."""

    provider = "flaky"
    model = "flaky-1"

    def __init__(self, errors: Sequence[ModelError]):
        self.errors = list(errors)

    async def complete(self, system, messages, tools=(), tool_choice="auto") -> Reply:
        if self.errors:
            raise self.errors.pop(0)
        return Reply(Assistant(text="done"), "flaky-1-0409", Usage(10, 2))


async def test_rate_limits_are_waited_out_and_logged():
    clock = FakeClock()
    model = Flaky([ModelError("flaky", 429, "slow down", 7.0), ModelError("flaky", 429, "again")])
    async with open_toolbox(servers()) as toolbox:
        record = await run_task(model, TASK, toolbox, clock=clock)
    assert clock.sleeps == [7.0, 60.0]
    assert record["outcome"] == "answered"
    assert record["model"] == "flaky-1-0409"
    assert record["waited_s"] == 67.0
    assert record["usage"] == {"input_tokens": 10, "output_tokens": 2}


async def test_other_provider_errors_end_the_task_as_model_error():
    model = Flaky([ModelError("flaky", 500, "boom")])
    async with open_toolbox(servers()) as toolbox:
        record = await run_task(model, TASK, toolbox, clock=FakeClock())
    assert record["outcome"] == "model_error"
    assert "HTTP 500" in record["error"]
    assert record["turns"] == []


async def test_unknown_integration_fails_before_any_model_call():
    model = ScriptedModel([])
    async with open_toolbox(servers()) as toolbox:
        with pytest.raises(KeyError, match="slack"):
            await run_task(model, Task("x", ("slack",), "hi"), toolbox)
