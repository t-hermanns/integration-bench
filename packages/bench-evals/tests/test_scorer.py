import json

import pytest

from bench_core.testing import FakeClock

from bench_evals.models import Assistant, Request, ScriptedModel, ToolCall, ToolResult
from bench_evals.runner import run_tasks
from bench_evals.scorer import Call, Score, calls_of, score, score_run, unmatched_calls
from bench_evals.tasks import EvalTask, ExpectedCall, parse_tasks
from bench_jira.fake import credential, seeded_fake
from bench_jira.scopes import READ as READ_SCOPE
from bench_jira.server import create_server as jira_server

TASKS = parse_tasks(
    """
integrations: [jira]
tasks:
  - id: read
    category: read
    split: dev
    prompt: Summary of BENCH-1?
    expected_calls:
      - {tool: jira_get_issue, args: {issue_key: BENCH-1}}
    forbidden_tools: [jira_create_issue]
    answer_contains: [BENCH-1, [open, "to do"]]
  - id: comment-twice
    category: multi_step_write
    split: test
    prompt: "Comment 'a {run_id}' and 'b {run_id}' on BENCH-1."
    expected_calls:
      - {tool: jira_add_comment, args: {issue_key: BENCH-1, body: "a {run_id}"}}
      - {tool: jira_add_comment, args: {issue_key: BENCH-1, body: "b {run_id}"}}
  - id: denied
    category: permission
    split: dev
    prompt: Move BENCH-1 to Done.
    expected_calls:
      - {tool: jira_transition_issue, error: true}
    answer_contains: [[permission, not allowed]]
"""
)
READ, TWICE, DENIED = TASKS


def turn(*calls: tuple[str, dict, bool], text: str = "") -> dict:
    return {
        "text": text,
        "tool_calls": [
            {"id": f"c{i}", "name": n, "arguments": a} for i, (n, a, _) in enumerate(calls)
        ],
        "results": [{"name": n, "is_error": e, "content": ""} for n, _, e in calls],
    }


def record(task: EvalTask, *turns: dict, answer: str = "", outcome: str = "answered") -> dict:
    return {
        "run_id": "r1",
        "task": task.id,
        "outcome": outcome,
        "answer": answer,
        "error": None,
        "turns": list(turns),
        "tool_calls": sum(len(t["tool_calls"]) for t in turns),
    }


def test_solved():
    rec = record(
        READ,
        turn(("jira_get_issue", {"issue_key": "bench-1", "response_format": "detailed"}, False)),
        answer="BENCH-1 is still To Do.",
    )
    assert score(READ, rec) == Score("read", True, True, True, 1, ())


def test_wrong_arguments_are_the_right_tool_but_not_solved():
    rec = record(
        READ,
        turn(("jira_get_issue", {"issue_key": "BENCH-2"}, False)),
        turn(("jira_search_issues", {"jql": "key = BENCH-1"}, False)),
        answer="BENCH-1 is open",
    )
    s = score(READ, rec)
    assert (s.solved, s.right_tool, s.calls) == (False, True, 2)
    assert s.reasons == (
        "no successful jira_get_issue call with arguments matching {'issue_key': 'BENCH-1'}",
    )


def test_a_failed_call_does_not_count_unless_a_retry_succeeds():
    failed = turn(("jira_get_issue", {"issue_key": "BENCH-1"}, True))
    assert score(READ, record(READ, failed, answer="BENCH-1 open")).solved is False
    retried = record(READ, failed, failed, answer="BENCH-1 open")
    retried["turns"][1]["results"][0]["is_error"] = False
    assert score(READ, retried).solved is True


def test_forbidden_tool_and_missing_answer():
    rec = record(
        READ,
        turn(("jira_get_issue", {"issue_key": "BENCH-1"}, False)),
        turn(("jira_create_issue", {"project_key": "BENCH", "summary": "x"}, True)),
        answer="Done: BENCH-1",
    )
    s = score(READ, rec)
    assert (s.solved, s.right_tool) == (False, False)
    assert s.reasons == ("called forbidden jira_create_issue", "answer lacks open | to do")


def test_never_called_is_reported_once():
    s = score(READ, record(READ, answer="BENCH-1 open"))
    assert (s.solved, s.right_tool, s.calls) == (False, False, 0)
    assert s.reasons == ("never called jira_get_issue",)


def test_each_expected_call_needs_its_own_call_and_run_id_is_filled():
    a = ("jira_add_comment", {"issue_key": "BENCH-1", "body": "A r1"}, False)
    b = ("jira_add_comment", {"issue_key": "BENCH-1", "body": "b r1"}, False)
    assert score(TWICE, record(TWICE, turn(a, b), answer="ok")).solved is True
    s = score(TWICE, record(TWICE, turn(a, a), answer="ok"))
    assert (s.solved, s.right_tool) == (False, True)
    assert s.reasons == (
        "no successful jira_add_comment call with arguments matching "
        "{'issue_key': 'BENCH-1', 'body': 'b r1'}",
    )
    other_run = ("jira_add_comment", {"issue_key": "BENCH-1", "body": "b r0"}, False)
    assert score(TWICE, record(TWICE, turn(a, other_run), answer="ok")).solved is False


def test_unmatched_calls_finds_the_best_assignment():
    # A greedy pass would give the loose expectation the only call the strict one fits.
    loose = ExpectedCall("t", {"q": {"$present": True}})
    strict = ExpectedCall("t", {"q": "x"})
    calls = [Call("t", {"q": "x"}, False), Call("t", {"q": "y"}, False)]
    assert unmatched_calls([loose, strict], calls) == []
    assert unmatched_calls([strict, strict], calls) == [strict]
    assert unmatched_calls([], calls) == []


def test_expected_error():
    denied = turn(("jira_transition_issue", {"issue_key": "BENCH-1", "status": "Done"}, True))
    rec = record(DENIED, denied, answer="I am not allowed to change it.")
    assert score(DENIED, rec).solved is True
    worked = record(DENIED, turn(("jira_transition_issue", {}, False)), answer="not allowed")
    assert score(DENIED, worked).solved is False


def test_max_turns_is_scored_but_unsolved():
    calls = [turn(("jira_get_issue", {"issue_key": "BENCH-1"}, False))] * 8
    s = score(READ, record(READ, *calls, outcome="max_turns"))
    assert (s.valid, s.solved, s.right_tool, s.calls) == (True, False, True, 8)
    assert s.reasons[-1] == "no final answer (max_turns)"


def test_model_error_is_invalid():
    rec = record(READ, outcome="model_error")
    rec["error"] = "groq 500"
    assert score(READ, rec) == Score("read", False, None, None, None, ("model error: groq 500",))


def test_score_run_rejects_unknown_and_mismatched_tasks():
    with pytest.raises(KeyError, match="no task gone"):
        score_run(TASKS, [{"task": "gone"}])
    with pytest.raises(ValueError, match="record of task read scored against denied"):
        score(DENIED, record(READ))


def test_calls_of_requires_one_result_per_call():
    broken = turn(("jira_get_issue", {}, False))
    broken["results"] = []
    with pytest.raises(ValueError, match="shorter"):
        calls_of(record(READ, broken))


@pytest.mark.anyio
async def test_scores_a_real_runner_record(tmp_path):
    def answer(request: Request) -> Assistant:
        result = request.messages[-1]
        assert isinstance(result, ToolResult)
        issue = json.loads(result.content)
        return Assistant(text=f"BENCH-1 ({issue['status']}): {issue['summary']}")

    model = ScriptedModel(
        [
            Assistant(tool_calls=(ToolCall("c1", "jira_get_issue", {"issue_key": "BENCH-1"}),)),
            answer,
        ]
    )
    fake = seeded_fake(FakeClock())
    fake.issue_token("t", [READ_SCOPE])
    jira = jira_server(credential=credential("t"), transport=fake.transport(), clock=FakeClock())
    records = await run_tasks(
        model, [READ.for_run("r9")], {"jira": jira}, tmp_path / "run.jsonl", run_id="r9"
    )
    assert records[0]["answer"].startswith("BENCH-1 (To Do): ")
    assert score_run(TASKS, records) == [Score("read", True, True, True, 1, ())]
