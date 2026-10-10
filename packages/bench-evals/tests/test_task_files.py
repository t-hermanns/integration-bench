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
from bench_gdrive.fake import credential as gdrive_credential
from bench_gdrive.fake import seeded_fake as gdrive_fake
from bench_gdrive.scopes import DRIVE_FILE
from bench_gdrive.server import create_server as gdrive_server
from bench_github.fake import seeded_fake as github_fake
from bench_github.server import create_server as github_server
from bench_jira.fake import credential
from bench_jira.fake import seeded_fake as jira_fake
from bench_jira.scopes import READ, WRITE
from bench_jira.server import create_server as jira_server
from bench_slack.fake import RELEASE
from bench_slack.fake import credential as slack_credential
from bench_slack.fake import seeded_fake as slack_fake
from bench_slack.scopes import ALL as SLACK_SCOPES
from bench_slack.server import create_server as slack_server

pytestmark = pytest.mark.anyio

ROOT = Path(__file__).resolve().parents[3]
TOKEN = "eval-task-token"
RUN_ID = "r0test"
CURSOR = "<next_cursor>"
"""Stands for the `next_cursor` of the previous tool result."""

GH = {"owner": "bench-sandbox", "repo": "widgets"}
FLAKY = "Who owns the flaky export test?"


def ref(field: str, key: str) -> str:
    """`field` of the newest result item whose name or text is `key` (ids, ts)."""
    return f"<{field}|{key}>"


def find(name: str) -> tuple[str, dict[str, Any]]:
    return ("gdrive_search_files", {"query": name})


def read(name: str, **extra: Any) -> list[tuple[str, dict[str, Any]]]:
    return [find(name), ("gdrive_read_file_content", {"file_id": ref("id", name), **extra})]


CHANNELS = ("slack_list_channels", {})
SANDBOX = {"channel_id": ref("id", "bench-sandbox")}
HISTORY = ("slack_read_channel_history", SANDBOX)


def thread(text: str) -> dict[str, Any]:
    return {**SANDBOX, "thread_ts": ref("ts", text)}


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
    "gdrive-read-roadmap-q3": (read("Roadmap 2026"), ""),
    "gdrive-read-glossary": (read("glossary.txt"), ""),
    "gdrive-read-release-step": (read("Release checklist"), ""),
    "gdrive-args-search-in-folder": (
        [
            find("Meeting notes"),
            (
                "gdrive_search_files",
                {"query": "dark mode", "folder_id": ref("id", "Meeting notes")},
            ),
        ],
        "",
    ),
    "gdrive-args-max-chars": (read("Roadmap 2026", max_chars=100), ""),
    "gdrive-args-mime-type": (
        [
            find("glossary.txt"),
            (
                "gdrive_get_file_metadata",
                {"file_id": ref("id", "glossary.txt"), "response_format": "detailed"},
            ),
        ],
        "",
    ),
    "gdrive-write-agenda-in-folder": (
        [
            find("Meeting notes"),
            (
                "gdrive_create_document",
                {
                    "name": f"Agenda 2026-09-28 ({RUN_ID})",
                    "content": "Release 2.4 status",
                    "folder_id": ref("id", "Meeting notes"),
                },
            ),
        ],
        "Created it.",
    ),
    "gdrive-write-follow-up-doc": (
        [
            *read("Weekly sync 2026-09-14"),
            find("Projects"),
            (
                "gdrive_create_document",
                {
                    "name": f"Export follow-up ({RUN_ID})",
                    "content": "Lee investigates the export timeouts.",
                    "folder_id": ref("id", "Projects"),
                },
            ),
        ],
        "",
    ),
    "gdrive-page-sandbox-folder": (
        [
            find("bench-sandbox"),
            ("gdrive_list_folder", {"folder_id": ref("id", "bench-sandbox"), "page_size": 2}),
            (
                "gdrive_list_folder",
                {"folder_id": ref("id", "bench-sandbox"), "page_size": 2, "cursor": CURSOR},
            ),
        ],
        "",
    ),
    "gdrive-page-weekly-syncs": (
        [
            ("gdrive_search_files", {"query": "Weekly sync", "page_size": 2}),
            ("gdrive_search_files", {"query": "Weekly sync", "page_size": 2, "cursor": CURSOR}),
        ],
        "",
    ),
    "gdrive-perm-foreign-file": (
        [("gdrive_read_file_content", {"file_id": "1AbCdEfGhIjKlMnOpQrStUvWxYz"})],
        "",
    ),
    "gdrive-abstain-delete": ([], "I can't delete files with the tools I have."),
    "gdrive-abstain-share": ([], "I can't share files with the tools I have."),
    "gdrive-confuse-folder-contents": (
        [find("Projects"), ("gdrive_list_folder", {"folder_id": ref("id", "Projects")})],
        "",
    ),
    "gdrive-confuse-owner-in-text": (read("Weekly sync 2026-09-07"), ""),
    "slack-read-staging-db": ([CHANNELS, HISTORY], ""),
    "slack-read-welcome": ([CHANNELS, HISTORY], ""),
    "slack-read-standup": ([CHANNELS, HISTORY], ""),
    "slack-args-thread-flaky-test": (
        [CHANNELS, HISTORY, ("slack_read_thread", thread(FLAKY))],
        "",
    ),
    "slack-args-thread-release-qa": (
        [CHANNELS, HISTORY, ("slack_read_thread", thread(RELEASE))],
        "",
    ),
    "slack-args-history-page-size": (
        [CHANNELS, ("slack_read_channel_history", {**SANDBOX, "page_size": 50})],
        "",
    ),
    "slack-write-reply-release": (
        [
            CHANNELS,
            HISTORY,
            ("slack_reply_in_thread", {**thread(RELEASE), "text": f"QA is done ({RUN_ID})."}),
        ],
        "Replied.",
    ),
    "slack-write-post-owner": (
        [
            CHANNELS,
            HISTORY,
            ("slack_read_thread", thread(FLAKY)),
            (
                "slack_post_message",
                {**SANDBOX, "text": f"The flaky export test fix is in review ({RUN_ID})."},
            ),
        ],
        "Posted.",
    ),
    "slack-page-channels": (
        [
            ("slack_list_channels", {"page_size": 2}),
            ("slack_list_channels", {"page_size": 2, "cursor": CURSOR}),
        ],
        "",
    ),
    "slack-page-history": (
        [
            CHANNELS,
            ("slack_read_channel_history", {**SANDBOX, "page_size": 2}),
            ("slack_read_channel_history", {**SANDBOX, "page_size": 2, "cursor": CURSOR}),
        ],
        "",
    ),
    "slack-perm-not-member": (
        [CHANNELS, ("slack_read_channel_history", {"channel_id": ref("id", "sozial")})],
        "",
    ),
    "slack-abstain-direct-message": ([], "I can't send direct messages with the tools I have."),
    "slack-abstain-missing-details": ([], "Which channel, and what should the update say?"),
    "slack-confuse-announcement": (
        [
            CHANNELS,
            ("slack_post_message", {**SANDBOX, "text": f"Code freeze starts Friday ({RUN_ID})."}),
        ],
        "",
    ),
    "slack-confuse-answer-question": (
        [
            CHANNELS,
            HISTORY,
            ("slack_reply_in_thread", {**thread(FLAKY), "text": f"Lee takes over ({RUN_ID})."}),
        ],
        "",
    ),
    "cross-checklist-step-to-slack": (
        [
            *read("Release checklist"),
            CHANNELS,
            ("slack_post_message", {**SANDBOX, "text": f"3. Update the changelog ({RUN_ID})"}),
        ],
        "Posted.",
    ),
    "cross-issue-to-drive-doc": (
        [
            ("github_get_issue", {**GH, "number": 12}),
            (
                "gdrive_create_document",
                {
                    "name": f"Issue 12 notes ({RUN_ID})",
                    "content": "Crash on startup with Unicode paths",
                },
            ),
        ],
        "Created it.",
    ),
    "cross-notes-owner-to-thread": (
        [
            *read("Weekly sync 2026-09-14"),
            CHANNELS,
            HISTORY,
            ("slack_reply_in_thread", {**thread(RELEASE), "text": f"Lee ({RUN_ID})"}),
        ],
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
    gdrive = gdrive_fake(FakeClock())
    gdrive.issue_token(TOKEN, [DRIVE_FILE])
    slack = slack_fake(FakeClock())
    slack.issue_token(TOKEN, sorted(SLACK_SCOPES))
    return {
        "jira": jira_server(credential=credential(TOKEN), transport=jira.transport(), **common),
        "github": github_server(
            credential=StaticToken("github", TOKEN), transport=github.transport(), **common
        ),
        "gdrive": gdrive_server(
            credential=gdrive_credential(TOKEN), transport=gdrive.transport(), **common
        ),
        "slack": slack_server(
            credential=slack_credential(TOKEN), transport=slack.transport(), **common
        ),
    }


def test_task_files_are_balanced():
    found = tasks()
    by_integration = Counter(t.integrations for t in found if len(t.integrations) == 1)
    assert 0 < len(found) - by_integration.total() <= 6, "a few tasks span integrations"
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


def test_prompts_are_whole_sentences():
    """An unquoted ` #12` starts a YAML comment and silently cuts the prompt off."""
    cut = [t.id for t in tasks() if not t.prompt.rstrip().endswith((".", "?", "!", '"'))]
    assert cut == []


def test_every_task_has_a_solution():
    assert sorted(SOLUTIONS) == sorted(t.id for t in tasks())


def resolve(request: Request, value: Any) -> Any:
    """Replaces a `ref` placeholder with the value from the newest matching result."""
    if not (isinstance(value, str) and value.startswith("<") and "|" in value):
        return value
    field, key = value[1:-1].split("|", 1)
    for message in reversed(request.messages):
        if isinstance(message, ToolResult) and not message.is_error:
            for item in items(json.loads(message.content)):
                if key in (item.get("name"), item.get("text")):
                    return item[field]
    raise AssertionError(f"no result item named {key!r}")


def items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [found for entry in data for found in items(entry)]
    if isinstance(data, dict):
        return [data, *items(list(data.values()))]
    return []


def script(calls: list[tuple[str, dict[str, Any]]], words: str) -> list[Any]:
    def step(n: int, name: str, arguments: dict[str, Any]):
        def reply(request: Request) -> Assistant:
            args = {key: resolve(request, value) for key, value in arguments.items()}
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
