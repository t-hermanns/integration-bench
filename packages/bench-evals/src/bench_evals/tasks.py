"""Eval tasks: the YAML format, loading and checks against the servers' tools.

One file per integration (`integrations/<name>/evals/tasks.yaml`) plus one for tasks
that span integrations (`evals/tasks.yaml`):

    integrations: [jira]            # default for every task in the file
    tasks:
      - id: jira-read-summary
        category: read
        split: dev
        prompt: What is the summary of BENCH-1?
        expected_calls:
          - tool: jira_get_issue
            args: {issue_key: BENCH-1}
        forbidden_tools: [jira_create_issue]
        answer_contains: [login]     # an item may be a list of alternatives

Argument matchers are described in `bench_evals.matchers`. `{run_id}` in a prompt,
matcher or expected answer is replaced by the run's id, so writes can be found and
cleaned up afterwards.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from mcp.types import Tool as McpTool

from bench_evals.matchers import check_matcher
from bench_evals.runner import Task

CATEGORIES = (
    "read",
    "arguments",
    "multi_step_write",
    "pagination",
    "permission",
    "abstention",
    "confusable",
)
SPLITS = ("dev", "test")
RUN_ID = "{run_id}"

_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_TASK_KEYS = {
    "id",
    "integrations",
    "category",
    "split",
    "prompt",
    "expected_calls",
    "forbidden_tools",
    "answer_contains",
}
_CALL_KEYS = {"tool", "args", "error"}


class TaskFormatError(ValueError):
    pass


@dataclass(frozen=True)
class ExpectedCall:
    tool: str
    args: Mapping[str, Any]
    """Argument name -> matcher; arguments not listed here are not checked."""
    error: bool = False
    """True if the call is expected to fail (missing permission), not to succeed."""


@dataclass(frozen=True)
class EvalTask:
    id: str
    integrations: tuple[str, ...]
    category: str
    split: str
    prompt: str
    expected_calls: tuple[ExpectedCall, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    answer_contains: tuple[tuple[str, ...], ...] = ()
    """All items must appear in the answer; each item is a tuple of alternatives."""

    def for_run(self, run_id: str) -> Task:
        """What the runner gets: the prompt with `{run_id}` filled in."""
        return Task(self.id, self.integrations, self.prompt.replace(RUN_ID, run_id))


def task_files(root: Path) -> list[Path]:
    files = sorted(root.glob("integrations/*/evals/tasks.yaml"))
    shared = root / "evals" / "tasks.yaml"
    return [*files, shared] if shared.exists() else files


def load_tasks(paths: Iterable[Path]) -> list[EvalTask]:
    tasks: list[EvalTask] = []
    seen: dict[str, Path] = {}
    for path in paths:
        for task in parse_tasks(path.read_text(encoding="utf-8"), str(path)):
            if task.id in seen:
                raise TaskFormatError(
                    f"{path}: duplicate task id {task.id} (also in {seen[task.id]})"
                )
            seen[task.id] = path
            tasks.append(task)
    return tasks


def parse_tasks(text: str, source: str = "<tasks>") -> list[EvalTask]:
    doc = yaml.safe_load(text) or {}
    if not isinstance(doc, dict) or not isinstance(doc.get("tasks"), list):
        raise TaskFormatError(f"{source}: expected a mapping with a `tasks` list")
    unknown = set(doc) - {"integrations", "tasks"}
    if unknown:
        raise TaskFormatError(f"{source}: unknown keys {sorted(unknown)}")
    default = doc.get("integrations")
    tasks = []
    for i, raw in enumerate(doc["tasks"]):
        where = f"{source}: task {raw.get('id', i) if isinstance(raw, dict) else i}"
        try:
            tasks.append(_task(raw, default))
        except TaskFormatError as exc:
            raise TaskFormatError(f"{where}: {exc}") from None
    return tasks


def _task(raw: Any, default_integrations: Any) -> EvalTask:
    if not isinstance(raw, dict):
        raise TaskFormatError("not a mapping")
    unknown = set(raw) - _TASK_KEYS
    if unknown:
        raise TaskFormatError(f"unknown keys {sorted(unknown)}")
    for key in ("id", "category", "split", "prompt"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise TaskFormatError(f"`{key}` must be a non-empty string")
    if not _ID.match(raw["id"]):
        raise TaskFormatError("`id` must be lowercase letters, digits and dashes")
    if raw["category"] not in CATEGORIES:
        raise TaskFormatError(f"`category` must be one of {', '.join(CATEGORIES)}")
    if raw["split"] not in SPLITS:
        raise TaskFormatError(f"`split` must be one of {', '.join(SPLITS)}")
    integrations = _strings(raw.get("integrations", default_integrations), "integrations")
    if not integrations:
        raise TaskFormatError("no `integrations` (neither on the task nor for the file)")
    calls = tuple(_call(c) for c in _list(raw.get("expected_calls"), "expected_calls"))
    forbidden = _strings(raw.get("forbidden_tools"), "forbidden_tools")
    answer = tuple(
        _strings(item if isinstance(item, list) else [item], "answer_contains")
        for item in _list(raw.get("answer_contains"), "answer_contains")
    )
    if not (calls or forbidden or answer):
        raise TaskFormatError(
            "nothing to check: no expected_calls, forbidden_tools or answer_contains"
        )
    clash = {c.tool for c in calls} & set(forbidden)
    if clash:
        raise TaskFormatError(f"tools both expected and forbidden: {sorted(clash)}")
    if any(not alternatives for alternatives in answer):
        raise TaskFormatError("empty alternative list in `answer_contains`")
    return EvalTask(
        raw["id"],
        integrations,
        raw["category"],
        raw["split"],
        raw["prompt"],
        calls,
        forbidden,
        answer,
    )


def _call(raw: Any) -> ExpectedCall:
    if not isinstance(raw, dict) or not isinstance(raw.get("tool"), str):
        raise TaskFormatError("an expected call needs a `tool` name")
    unknown = set(raw) - _CALL_KEYS
    if unknown:
        raise TaskFormatError(f"unknown keys in expected call: {sorted(unknown)}")
    args = raw.get("args") or {}
    if not isinstance(args, dict):
        raise TaskFormatError(f"{raw['tool']}: `args` must be a mapping")
    for name, matcher in args.items():
        try:
            check_matcher(matcher)
        except ValueError as exc:
            raise TaskFormatError(f"{raw['tool']}.{name}: {exc}") from None
    error = raw.get("error", False)
    if not isinstance(error, bool):
        raise TaskFormatError(f"{raw['tool']}: `error` must be true or false")
    return ExpectedCall(raw["tool"], args, error)


def _list(value: Any, key: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TaskFormatError(f"`{key}` must be a list")
    return value


def _strings(value: Any, key: str) -> tuple[str, ...]:
    items = _list(value, key)
    if not all(isinstance(s, str) and s.strip() for s in items):
        raise TaskFormatError(f"`{key}` must be a list of non-empty strings")
    return tuple(items)


def check_tools(tasks: Sequence[EvalTask], tools: Mapping[str, Sequence[McpTool]]) -> list[str]:
    """Problems with tool and argument names, given each integration's `list_tools()`.

    Catches typos that would otherwise only show up as tasks nobody can solve.
    """
    problems = []
    for task in tasks:
        missing = [i for i in task.integrations if i not in tools]
        if missing:
            problems.append(f"{task.id}: unknown integrations {missing}")
            continue
        own = {t.name: t for i in task.integrations for t in tools[i]}
        for name in task.forbidden_tools:
            if name not in own:
                problems.append(f"{task.id}: forbidden tool {name} is not one of the task's tools")
        for call in task.expected_calls:
            tool = own.get(call.tool)
            if tool is None:
                problems.append(
                    f"{task.id}: expected tool {call.tool} is not one of the task's tools"
                )
                continue
            params = set(tool.input_schema.get("properties", {}))
            for arg in call.args:
                if arg not in params:
                    problems.append(f"{task.id}: {call.tool} has no argument {arg}")
    return problems
