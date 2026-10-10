"""Scores one runner record against its task: solved, right tool, number of calls.

Deterministic and offline: everything comes from the record the runner wrote (calls,
their arguments, whether each result was an error, the final answer).

- **solved**: the model gave a final answer (not cut off after `MAX_TURNS`), every
  expected call has its own matching call (arguments match, and the result succeeded,
  or failed for `error: true`), no forbidden tool was called, and the answer contains
  every `answer_contains` item (one of its alternatives, ignoring case).
- **right_tool**: every expected tool was called at least once, whatever the
  arguments, and no forbidden tool was called.
- **calls**: tool calls in the task, failed ones included.

A record whose provider failed (`model_error`) is not valid: it says nothing about the
model, so its metrics are None and it is left out of every rate.

**Headline metric, fixed before the first measurement (step 36):** `solved`. The
description experiment compares v1 and v2 on the test split, per task; with k > 1 a
task counts as solved if it was solved in more than half of its valid runs. The
paired comparison is the exact McNemar test (step 37). `right_tool` and `calls` are
reported alongside, but are not tested for significance.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from bench_evals.matchers import args_match, fill
from bench_evals.tasks import RUN_ID, EvalTask, ExpectedCall

HEADLINE = "solved"


@dataclass(frozen=True)
class Call:
    name: str
    arguments: dict[str, Any]
    is_error: bool


@dataclass(frozen=True)
class Score:
    task: str
    valid: bool
    solved: bool | None
    right_tool: bool | None
    calls: int | None
    reasons: tuple[str, ...] = ()
    """Why the task is not solved (empty if it is); for reading failures, not for metrics."""


def calls_of(record: dict[str, Any]) -> list[Call]:
    """Every tool call of the record with whether its result was an error."""
    calls = []
    for turn in record["turns"]:
        for call, result in zip(turn["tool_calls"], turn["results"], strict=True):
            calls.append(Call(call["name"], call["arguments"], result["is_error"]))
    return calls


def score(task: EvalTask, record: dict[str, Any]) -> Score:
    if record["task"] != task.id:
        raise ValueError(f"record of task {record['task']} scored against {task.id}")
    if record["outcome"] == "model_error":
        return Score(task.id, False, None, None, None, (f"model error: {record['error']}",))
    run_id = record.get("run_id", "")
    calls = calls_of(record)
    called = {c.name for c in calls}
    reasons = []

    forbidden = sorted(set(task.forbidden_tools) & called)
    if forbidden:
        reasons.append(f"called forbidden {', '.join(forbidden)}")
    never = sorted({e.tool for e in task.expected_calls} - called)
    if never:
        reasons.append(f"never called {', '.join(never)}")
    right_tool = not forbidden and not never

    expected = [
        ExpectedCall(e.tool, fill(dict(e.args), RUN_ID, run_id), e.error)
        for e in task.expected_calls
    ]
    unmatched = unmatched_calls(expected, calls)
    for e in unmatched:
        if e.tool not in never:
            kind = "failing" if e.error else "successful"
            reasons.append(f"no {kind} {e.tool} call with arguments matching {e.args}")

    answer = (record.get("answer") or "").casefold()
    missing = [
        alternatives
        for alternatives in task.answer_contains
        if not any(a.replace(RUN_ID, run_id).casefold() in answer for a in alternatives)
    ]
    for alternatives in missing:
        reasons.append(f"answer lacks {' | '.join(alternatives)}")

    answered = record["outcome"] == "answered"
    if not answered:
        reasons.append(f"no final answer ({record['outcome']})")
    solved = answered and not forbidden and not unmatched and not missing
    return Score(task.id, True, solved, right_tool, record["tool_calls"], tuple(reasons))


def unmatched_calls(expected: Sequence[ExpectedCall], calls: Sequence[Call]) -> list[ExpectedCall]:
    """Expected calls left over by the best assignment of distinct actual calls.

    Two expected comments on one issue need two calls, so each actual call serves at
    most one expectation. Backtracking finds the assignment that leaves the fewest
    expectations open; tasks have a handful of calls, so this stays cheap.
    """

    def fits(e: ExpectedCall, c: Call) -> bool:
        return c.name == e.tool and c.is_error == e.error and args_match(dict(e.args), c.arguments)

    best: list[ExpectedCall] = list(expected)

    def search(i: int, used: frozenset[int], open_: list[ExpectedCall]) -> None:
        nonlocal best
        if len(open_) >= len(best):
            return
        if i == len(expected):
            best = open_
            return
        for j, c in enumerate(calls):
            if j not in used and fits(expected[i], c):
                search(i + 1, used | {j}, open_)
                if not best:
                    return
        search(i + 1, used, [*open_, expected[i]])

    if expected:
        search(0, frozenset(), [])
    return best


def score_run(tasks: Iterable[EvalTask], records: Iterable[dict[str, Any]]) -> list[Score]:
    """One score per record, in record order; a record of an unknown task is an error."""
    by_id = {t.id: t for t in tasks}
    scores = []
    for record in records:
        task = by_id.get(record["task"])
        if task is None:
            raise KeyError(f"no task {record['task']}")
        scores.append(score(task, record))
    return scores
