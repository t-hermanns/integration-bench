import textwrap

import httpx2
import pytest
from mcp import Client

from bench_evals.tasks import (
    EvalTask,
    ExpectedCall,
    TaskFormatError,
    check_tools,
    load_tasks,
    parse_tasks,
    task_files,
)
from bench_jira.server import create_server as jira_server

FILE = """
integrations: [jira]
tasks:
  - id: jira-read-summary
    category: read
    split: dev
    prompt: What is the summary of BENCH-1?
    expected_calls:
      - tool: jira_get_issue
        args: {issue_key: BENCH-1}
    forbidden_tools: [jira_create_issue]
    answer_contains: [login, [done, closed]]
  - id: jira-create-bug
    integrations: [jira, github]
    category: multi_step_write
    split: test
    prompt: "Create a bug titled 'crash {run_id}' in BENCH."
    expected_calls:
      - tool: jira_create_issue
        args:
          summary: {$contains: "{run_id}"}
          issue_type: Bug
      - tool: jira_get_issue
        error: true
"""

TASK = """
tasks:
  - id: t1
    integrations: [jira]
    category: read
    split: dev
    prompt: p
    answer_contains: [x]
"""


def test_parse_tasks():
    read, write = parse_tasks(FILE)
    assert read == EvalTask(
        "jira-read-summary",
        ("jira",),
        "read",
        "dev",
        "What is the summary of BENCH-1?",
        (ExpectedCall("jira_get_issue", {"issue_key": "BENCH-1"}),),
        ("jira_create_issue",),
        (("login",), ("done", "closed")),
    )
    assert write.integrations == ("jira", "github")
    assert write.expected_calls[1] == ExpectedCall("jira_get_issue", {}, error=True)
    task = write.for_run("r42")
    assert (task.id, task.integrations) == ("jira-create-bug", ("jira", "github"))
    assert task.prompt == "Create a bug titled 'crash r42' in BENCH."


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("prompt: p", "prompt: ''"), "`prompt`"),
        (("id: t1", "id: T_1"), "`id`"),
        (("category: read", "category: easy"), "`category`"),
        (("split: dev", "split: train"), "`split`"),
        (("integrations: [jira]", "integrations: []"), "no `integrations`"),
        (("answer_contains: [x]", "answer_contains: [[]]"), "empty alternative"),
        (("answer_contains: [x]", "answer_contains: x"), "must be a list"),
        (("answer_contains: [x]", "answer: [x]"), "unknown keys"),
        (("answer_contains: [x]", "forbidden_tools: []"), "nothing to check"),
        (
            ("answer_contains: [x]", "expected_calls: [{tool: a, args: {q: {$like: x}}}]"),
            "a.q: unknown operators",
        ),
        (("answer_contains: [x]", "expected_calls: [{tool: a, error: 1}]"), "`error`"),
        (("answer_contains: [x]", "expected_calls: [{args: {}}]"), "needs a `tool`"),
        (
            ("answer_contains: [x]", "expected_calls: [{tool: a}]\n    forbidden_tools: [a]"),
            "both expected and forbidden",
        ),
    ],
)
def test_parse_tasks_rejects(change, message):
    with pytest.raises(TaskFormatError, match=r"task \S+: .*" + message.replace("[", r"\[")):
        parse_tasks(TASK.replace(*change))


def test_parse_tasks_rejects_bad_files():
    with pytest.raises(TaskFormatError, match="`tasks` list"):
        parse_tasks("- id: t1")
    with pytest.raises(TaskFormatError, match="unknown keys"):
        parse_tasks(TASK + "version: 2\n")


def test_load_tasks_finds_files_and_rejects_duplicate_ids(tmp_path):
    for path in ("integrations/jira/evals/tasks.yaml", "evals/tasks.yaml"):
        (tmp_path / path).parent.mkdir(parents=True)
        (tmp_path / path).write_text(TASK, encoding="utf-8")
    files = task_files(tmp_path)
    assert files == [
        tmp_path / "integrations/jira/evals/tasks.yaml",
        tmp_path / "evals/tasks.yaml",
    ]
    assert [t.id for t in load_tasks(files[:1])] == ["t1"]
    with pytest.raises(TaskFormatError, match="duplicate task id t1"):
        load_tasks(files)


@pytest.mark.anyio
async def test_check_tools_reports_unknown_tools_and_arguments():
    server = jira_server(
        transport=httpx2.MockTransport(lambda r: httpx2.Response(500)), read_only=False
    )
    async with Client(server) as client:
        tools = {"jira": (await client.list_tools()).tools}
    bad = textwrap.dedent(
        """
        integrations: [jira]
        tasks:
          - {id: ok, category: read, split: dev, prompt: p,
             expected_calls: [{tool: jira_get_issue, args: {issue_key: BENCH-1}}]}
          - {id: typo, category: read, split: dev, prompt: p,
             expected_calls: [{tool: jira_get_issue, args: {key: BENCH-1}}, {tool: jira_delete}],
             forbidden_tools: [github_create_issue]}
          - {id: other, integrations: [slack], category: read, split: dev, prompt: p,
             answer_contains: [x]}
        """
    )
    assert check_tools(parse_tasks(bad), tools) == [
        "typo: forbidden tool github_create_issue is not one of the task's tools",
        "typo: jira_get_issue has no argument key",
        "typo: expected tool jira_delete is not one of the task's tools",
        "other: unknown integrations ['slack']",
    ]
