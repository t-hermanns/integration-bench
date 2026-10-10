"""Nightly regression rule and the GitHub issues it files.

Rule (fixed before the first nightly run): the current run is a regression against the
baseline if, on the tasks with an outcome in both,

1. the solved rate dropped by at least `MIN_DROP` (5 percentage points), **and**
2. the exact McNemar test of the paired outcomes gives p < `ALPHA`.

A regression is then re-checked with a confirmation round of only the lost tasks
(`confirm`): a lost task that is solved in the rerun counts as flaky and goes back to
solved before the rule is applied again. No trend analysis: one baseline, one run.

Issues: one open issue per fingerprint. The fingerprint names the provider, the
description variant and the affected integrations, so a regression that persists over
several nights gets a comment each night instead of a new issue. A run that is mostly
invalid (model errors, missing access) is not judged at all; it files an `eval-infra`
issue instead.
"""

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx2

from bench_evals.results_branch import git
from bench_evals.scorer import Score, calls_of
from bench_evals.stats import Paired, compare
from bench_evals.tasks import EvalTask

MIN_DROP = 0.05
ALPHA = 0.05
MAX_INVALID_SHARE = 0.5
REGRESSION_LABEL = "eval-regression"
INFRA_LABEL = "eval-infra"
CODE_PATHS = ("integrations/", "packages/", "evals/")
"""Changes under these paths can move eval results; anything else (docs, CI) cannot."""
MAX_EXCERPTS = 5
EXCERPT_CHARS = 300
MARKER = "<!-- bench-fingerprint: {} -->"


@dataclass(frozen=True)
class Verdict:
    paired: Paired
    lost: tuple[str, ...]
    """Solved in the baseline, not solved now."""
    gained: tuple[str, ...]
    regressed: bool


def check(
    baseline: Mapping[str, bool | None],
    current: Mapping[str, bool | None],
    min_drop: float = MIN_DROP,
    alpha: float = ALPHA,
) -> Verdict:
    """Apply the regression rule to per-task outcomes (`stats.task_outcomes`)."""
    paired = compare(baseline, current)
    # Compare counts, not floats: a drop of exactly min_drop must count.
    dropped = paired.n > 0 and paired.only_a - paired.only_b >= min_drop * paired.n - 1e-9
    return Verdict(
        paired=paired,
        lost=tuple(sorted(t for t, v in current.items() if v is False and baseline.get(t))),
        gained=tuple(sorted(t for t, v in current.items() if v and baseline.get(t) is False)),
        regressed=dropped and paired.p_value < alpha,
    )


def confirm(
    baseline: Mapping[str, bool | None],
    current: Mapping[str, bool | None],
    rerun: Mapping[str, bool | None],
    min_drop: float = MIN_DROP,
    alpha: float = ALPHA,
) -> Verdict:
    """Apply the rule again after a confirmation round of the lost tasks.

    Only a pass in the rerun changes anything (the task was flaky, it counts as solved);
    a failure or an invalid rerun leaves the first result as it was.
    """
    merged = dict(current)
    for task, solved in rerun.items():
        if solved and task in merged:
            merged[task] = True
    return check(baseline, merged, min_drop, alpha)


def fingerprint(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:12]


def regression_fingerprint(
    provider: str, variant: str, lost: Iterable[str], tasks: Mapping[str, EvalTask]
) -> str:
    integrations = sorted({i for t in lost if t in tasks for i in tasks[t].integrations})
    return fingerprint("regression", provider, variant, *integrations)


def infra_problem(scores: Sequence[Score], max_invalid: float = MAX_INVALID_SHARE) -> str | None:
    """Why the run cannot be judged, or None if it can."""
    if not scores:
        return "the run produced no results"
    invalid = sum(1 for s in scores if not s.valid)
    if invalid / len(scores) > max_invalid:
        return f"{invalid} of {len(scores)} runs are invalid (model errors)"
    return None


def changed_files(root: Path, base: str, head: str) -> list[str]:
    return git("diff", "--name-only", f"{base}..{head}", cwd=root).split()


def classify(changed: Iterable[str], baseline_model: str, model: str) -> str:
    """'code changed' if the commit range touches code or tasks, else 'model drift'."""
    if any(f.startswith(CODE_PATHS) for f in changed):
        return "code changed"
    if baseline_model != model:
        return f"model drift ({baseline_model} -> {model})"
    return "model drift"


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def excerpt(record: Mapping[str, Any]) -> str:
    calls = calls_of(dict(record))
    lines = [f"- `{c.name}`{' (error)' if c.is_error else ''}" for c in calls] or ["- no calls"]
    answer = (record.get("answer") or record.get("error") or "").strip()
    if len(answer) > EXCERPT_CHARS:
        answer = answer[:EXCERPT_CHARS] + " ..."
    quoted = "\n".join(f"> {line}" for line in answer.splitlines()) or "> (no answer)"
    return "\n".join([f"outcome `{record.get('outcome')}`, calls:", *lines, "", quoted])


def regression_body(
    verdict: Verdict,
    *,
    fingerprint_: str,
    provider: str,
    variant: str,
    cause: str,
    compare_url: str,
    scores: Sequence[Score],
    records: Sequence[Mapping[str, Any]],
) -> str:
    """Issue body: summary, table of lost tasks, transcript excerpts, commit range."""
    p = verdict.paired
    reasons = {s.task: "; ".join(s.reasons) for s in scores if s.valid and not s.solved}
    by_task: dict[str, Mapping[str, Any]] = {}
    for record in records:
        by_task.setdefault(record["task"], record)
    out = [
        MARKER.format(fingerprint_),
        f"**{provider}**, descriptions `{variant}`: solved {pct(p.a.value)} -> "
        f"{pct(p.b.value)} on {p.n} paired tasks "
        f"(lost {p.only_a}, gained {p.only_b}, exact McNemar p = {p.p_value:.3f}).",
        "",
        f"Cause: **{cause}**. Commits since the baseline: {compare_url}",
        "",
        "| Task | Baseline | Now | Why not solved |",
        "|---|---|---|---|",
        *(f"| `{t}` | solved | not solved | {reasons.get(t, '')} |" for t in verdict.lost),
    ]
    for task in verdict.lost[:MAX_EXCERPTS]:
        if task in by_task:
            out += ["", f"<details><summary><code>{task}</code></summary>", ""]
            out += [excerpt(by_task[task]), "", "</details>"]
    return "\n".join(out) + "\n"


def infra_body(fingerprint_: str, provider: str, problem: str, run_url: str) -> str:
    return f"{MARKER.format(fingerprint_)}\n**{provider}**: {problem}.\n\nRun: {run_url}\n"


class Issues(Protocol):
    def find_open(self, label: str, fingerprint_: str) -> int | None: ...
    def create(self, title: str, body: str, labels: Sequence[str]) -> int: ...
    def comment(self, number: int, body: str) -> None: ...


class GitHubIssues:
    """The three issue calls the nightly run needs, on the GitHub REST API."""

    def __init__(self, repo: str, token: str, client: httpx2.Client | None = None):
        self.repo = repo
        self.client = client or httpx2.Client(base_url="https://api.github.com", timeout=30)
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def request(self, method: str, url: str, **kwargs: Any) -> httpx2.Response:
        response = self.client.request(method, url, headers=self.headers, **kwargs)
        response.raise_for_status()
        return response

    def find_open(self, label: str, fingerprint_: str) -> int | None:
        marker = MARKER.format(fingerprint_)
        url: str | None = f"/repos/{self.repo}/issues"
        params: dict[str, Any] | None = {"state": "open", "labels": label, "per_page": 100}
        while url:
            response = self.request("GET", url, params=params)
            for issue in response.json():
                if "pull_request" not in issue and marker in (issue.get("body") or ""):
                    return issue["number"]
            url, params = response.links.get("next", {}).get("url"), None
        return None

    def create(self, title: str, body: str, labels: Sequence[str]) -> int:
        payload = {"title": title, "body": body, "labels": list(labels)}
        return self.request("POST", f"/repos/{self.repo}/issues", json=payload).json()["number"]

    def comment(self, number: int, body: str) -> None:
        self.request("POST", f"/repos/{self.repo}/issues/{number}/comments", json={"body": body})


@dataclass(frozen=True)
class Filed:
    number: int
    created: bool


def file_issue(issues: Issues, label: str, fingerprint_: str, title: str, body: str) -> Filed:
    """Comment on the open issue with this fingerprint, or open a new one."""
    number = issues.find_open(label, fingerprint_)
    if number is not None:
        issues.comment(number, body)
        return Filed(number, created=False)
    return Filed(issues.create(title, body, [label]), created=True)
