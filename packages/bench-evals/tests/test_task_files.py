"""The repo's eval task files: well-formed, about the real tools, solvable on the seed data.

Each task gets a worked solution below: the calls a good model would make. It is played
back against the seeded fakes (which mirror the live sandbox), so a task whose expected
calls, matchers or answers drift from the seed data fails here instead of in a live run.
The final answer is the text of the tool results, so `answer_contains` must be backed by
what the tools return; only tasks whose answer is a refusal or a question add own words.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from mcp.client import Client

from bench_core.auth import StaticToken
from bench_core.testing import FakeClock

from bench_evals.models import Assistant, Request, ScriptedModel, ToolCall, ToolResult
from bench_evals.runner import open_toolbox, run_task
from bench_evals.scorer import score
from bench_evals.tasks import CATEGORIES, EvalTask, check_tools, load_tasks, task_files
from bench_github.fake import seeded_fake as github_fake
from bench_github.server import create_server as github_server
from bench_jira.fake import credential
from bench_jira.fake import seeded_fake as jira_fake
from bench_jira.scopes import READ, WRITE
from bench_jira.server import create_server as jira_server

pytestmark = pytest.mark.anyio

ROOT = Path(__file__).resolve().parents[3]
TOKEN = "eval-task-token"
RUN_ID = "r0test"
CURSOR = "<next_cursor>"
"""Stands for the `next_cursor` of the previous tool result."""

GH = {"owner": "bench-sandbox", "repo": "widgets"}

# task id -> (calls, own words added to the answer)
SOLUTIONS: dict[str, tuple[list[tuple[str, dict[str, Any]]], str]] = {
    "github-read-issue-error": ([("github_get_issue", {**GH, "number": 1})], ""),
    "github-read-comment": (
        [("github_get_issue", {**GH, "number": 7, "response_format": "detailed"})],
        "",
    ),
    "github-read-readme-install": (
        [("github_get_file_contents", {**GH, "path": "README.md"})],
        "",
    ),
    "github-args-label-performance": (
        [
            (
                "github_search_issues",
                {"query": "repo:bench-sandbox/widgets is:open label:performance"},
            )
        ],
        "",
    ),
    "github-args-path-with-space": (
        [("github_get_file_contents", {**GH, "path": "docs/user guide.md"})],
        "",
    ),
    "github-args-label-with-spaces": (
        [
            (
                "github_search_issues",
                {"query": 'repo:bench-sandbox/widgets is:open label:"good first issue"'},
            )
        ],
        "",
    ),
    "github-write-find-and-comment": (
        [
            ("github_search_issues", {"query": "repo:bench-sandbox/widgets is:open rate limit"}),
            ("github_add_issue_comment", {**GH, "number": 9, "body": f"Retry ({RUN_ID})."}),
        ],
        "",
    ),
    "github-write-follow-up-issue": (
        [
            (
                "github_create_issue",
                {
                    **GH,
                    "title": f"Add a test for Unicode paths ({RUN_ID})",
                    "labels": ["bug"],
                    "body": "Follow-up to #12.",
                },
            )
        ],
        "Created it.",
    ),
    "github-page-open-issues": (
        [
            (
                "github_search_issues",
                {"query": "repo:bench-sandbox/widgets is:issue is:open", "per_page": 5},
            ),
            (
                "github_search_issues",
                {
                    "query": "repo:bench-sandbox/widgets is:issue is:open",
                    "per_page": 5,
                    "cursor": CURSOR,
                },
            ),
        ],
        "",
    ),
    "github-page-all-pulls": (
        [
            ("github_list_pull_requests", {**GH, "state": "all", "per_page": 2}),
            ("github_list_pull_requests", {**GH, "state": "all", "per_page": 2, "cursor": CURSOR}),
        ],
        "",
    ),
    "github-perm-unknown-repo": (
        [
            (
                "github_get_file_contents",
                {"owner": "bench-sandbox", "repo": "billing", "path": "README.md"},
            )
        ],
        "",
    ),
    "github-abstain-merge": ([], "I can't merge pull requests with the tools I have."),
    "github-abstain-missing-details": ([], "Which repository, and what should the title be?"),
    "github-confuse-open-pulls": ([("github_list_pull_requests", GH)], ""),
    "github-confuse-note-on-issue": (
        [("github_add_issue_comment", {**GH, "number": 4, "body": f"Draft ({RUN_ID})."})],
        "",
    ),
    "jira-read-error": ([("jira_get_issue", {"issue_key": "BENCH-1"})], ""),
    "jira-read-latest-comment": (
        [("jira_get_issue", {"issue_key": "BENCH-3", "response_format": "detailed"})],
        "",
    ),
    "jira-read-status-and-platform": ([("jira_get_issue", {"issue_key": "BENCH-12"})], ""),
    "jira-args-high-open-bugs": (
        [
            (
                "jira_search_issues",
                {"jql": "project = BENCH AND type = Bug AND priority = High AND status != Done"},
            )
        ],
        "",
    ),
    "jira-args-assigned-to-me": (
        [("jira_search_issues", {"jql": "assignee = currentUser() AND status != Done"})],
        "",
    ),
    "jira-args-open-crashes": (
        [("jira_search_issues", {"jql": "project = BENCH AND labels = crash AND status != Done"})],
        "",
    ),
    "jira-args-in-progress-detailed": (
        [
            (
                "jira_search_issues",
                {
                    "jql": 'project = BENCH AND status = "In Progress"',
                    "response_format": "detailed",
                },
            )
        ],
        "",
    ),
    "jira-write-find-and-comment": (
        [
            ("jira_search_issues", {"jql": 'project = BENCH AND text ~ "network"'}),
            ("jira_add_comment", {"issue_key": "BENCH-7", "body": f"Offline ({RUN_ID})."}),
        ],
        "",
    ),
    "jira-write-follow-up-bug": (
        [
            (
                "jira_create_issue",
                {
                    "project_key": "BENCH",
                    "summary": f"Add a regression test for report totals ({RUN_ID})",
                    "issue_type": "Bug",
                    "labels": ["reports"],
                    "description": "Follow-up to BENCH-10.",
                },
            )
        ],
        "",
    ),
    "jira-write-create-and-start": (
        [
            (
                "jira_create_issue",
                {"project_key": "BENCH", "summary": f"Review ({RUN_ID})", "issue_type": "Task"},
            ),
            ("jira_transition_issue", {"issue_key": "BENCH-13", "status": "In Progress"}),
        ],
        "",
    ),
    "jira-page-all-keys": (
        [
            ("jira_search_issues", {"jql": "project = BENCH ORDER BY key"}),
            ("jira_search_issues", {"jql": "project = BENCH ORDER BY key", "cursor": CURSOR}),
        ],
        "",
    ),
    "jira-page-done": (
        [
            ("jira_search_issues", {"jql": "project = BENCH", "page_size": 4}),
            ("jira_search_issues", {"jql": "project = BENCH", "page_size": 4, "cursor": CURSOR}),
            ("jira_search_issues", {"jql": "project = BENCH", "page_size": 4, "cursor": CURSOR}),
        ],
        "",
    ),
    "jira-perm-other-project": (
        [
            (
                "jira_create_issue",
                {"project_key": "OPS", "summary": f"Rotate ({RUN_ID})", "issue_type": "Task"},
            )
        ],
        "",
    ),
    "jira-abstain-delete": ([], "I can't delete issues with the tools I have."),
    "jira-confuse-read-comments": (
        [("jira_get_issue", {"issue_key": "BENCH-12", "response_format": "detailed"})],
        "",
    ),
}


def tasks() -> list[EvalTask]:
    return load_tasks(task_files(ROOT))


def servers() -> dict[str, Any]:
    jira = jira_fake(FakeClock())
    jira.issue_token(TOKEN, [READ, WRITE])
    github = github_fake(FakeClock())
    github.issue_token(TOKEN, ["issues:write", "pull_requests:read", "contents:read"])
    common = {"clock": FakeClock(), "read_only": False, "variant": "v1"}
    return {
        "jira": jira_server(credential=credential(TOKEN), transport=jira.transport(), **common),
        "github": github_server(
            credential=StaticToken("github", TOKEN), transport=github.transport(), **common
        ),
    }


def test_task_files_are_balanced():
    found = tasks()
    by_integration = Counter(t.integrations for t in found)
    for integrations, n in by_integration.items():
        assert 13 <= n <= 17, f"{integrations}: {n} tasks"
    for integrations in by_integration:
        own = [t for t in found if t.integrations == integrations]
        test = sum(t.split == "test" for t in own)
        assert 0.3 <= test / len(own) <= 0.45, f"{integrations}: {test} of {len(own)} in test"
        assert {t.category for t in own} == set(CATEGORIES), integrations


async def test_task_files_name_real_tools_and_arguments():
    tools = {}
    for name, server in servers().items():
        async with Client(server) as client:
            tools[name] = (await client.list_tools()).tools
    assert check_tools(tasks(), tools) == []


def test_every_task_has_a_solution():
    assert sorted(SOLUTIONS) == sorted(t.id for t in tasks())


def script(calls: list[tuple[str, dict[str, Any]]], words: str) -> list[Any]:
    def step(n: int, name: str, arguments: dict[str, Any]):
        def reply(request: Request) -> Assistant:
            args = dict(arguments)
            if args.get("cursor") == CURSOR:
                last = request.messages[-1]
                assert isinstance(last, ToolResult)
                assert not last.is_error, last.content
                args["cursor"] = json.loads(last.content)["next_cursor"]
                assert args["cursor"], f"{name}: no next page"
            return Assistant(tool_calls=(ToolCall(f"c{n}", name, args),))

        return reply

    def answer(request: Request) -> Assistant:
        results = [m.content for m in request.messages if isinstance(m, ToolResult)]
        return Assistant(text="\n".join([*results, words]))

    return [step(n, name, args) for n, (name, args) in enumerate(calls)] + [answer]


@pytest.mark.parametrize("task", tasks(), ids=lambda t: t.id)
async def test_solution_solves_the_task_on_the_seed_data(task: EvalTask):
    calls, words = SOLUTIONS[task.id]
    model = ScriptedModel(script(calls, words))
    async with open_toolbox(servers()) as toolbox:
        record = await run_task(model, task.for_run(RUN_ID), toolbox, clock=FakeClock())
    record["run_id"] = RUN_ID
    result = score(task, record)
    assert result.solved, result.reasons
