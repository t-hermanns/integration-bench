import json
from typing import Any

import httpx2
import pytest

from bench_evals.regression import (
    INFRA_LABEL,
    MARKER,
    REGRESSION_LABEL,
    GitHubIssues,
    check,
    classify,
    confirm,
    file_issue,
    infra_problem,
    regression_body,
    regression_fingerprint,
)
from bench_evals.scorer import Score
from bench_evals.tasks import EvalTask


def outcomes(solved: int, failed: int, prefix: str = "t") -> dict[str, bool | None]:
    tasks = [f"{prefix}{i:02}" for i in range(solved + failed)]
    return {t: i < solved for i, t in enumerate(tasks)}


def worse(base: dict[str, bool | None], lost: int) -> dict[str, bool | None]:
    """The baseline with the first `lost` solved tasks failing."""
    now = dict(base)
    for task in [t for t, v in base.items() if v][:lost]:
        now[task] = False
    return now


def test_six_lost_tasks_of_forty_are_a_regression():
    base = outcomes(36, 4)
    verdict = check(base, worse(base, 6))
    # 6 discordant pairs, all one way: p = 2 / 2**6 = 0.03125; drop 15 points.
    assert verdict.paired.p_value == pytest.approx(0.03125)
    assert verdict.regressed
    assert verdict.lost == ("t00", "t01", "t02", "t03", "t04", "t05")
    assert verdict.gained == ()


def test_five_lost_tasks_are_not_significant():
    base = outcomes(36, 4)
    verdict = check(base, worse(base, 5))
    assert verdict.paired.p_value == pytest.approx(0.0625)
    assert not verdict.regressed


def test_significant_drop_below_the_threshold_is_no_regression():
    # 6 of 200 lost: McNemar p = 0.03125, but only a 3-point drop.
    base = outcomes(190, 10)
    verdict = check(base, worse(base, 6))
    assert verdict.paired.p_value < 0.05
    assert not verdict.regressed


def test_drop_of_exactly_the_threshold_counts():
    # 6 of 120 = exactly 5 points (a float subtraction would land just below).
    base = outcomes(110, 10)
    assert check(base, worse(base, 6)).regressed


def test_gains_offset_losses_and_improvements_never_regress():
    base = outcomes(30, 10)
    now = worse(base, 7)
    for task in ["t30", "t31", "t32"]:
        now[task] = True
    verdict = check(base, now)
    assert verdict.gained == ("t30", "t31", "t32")
    assert not verdict.regressed  # 7 vs 3: p = 0.34
    assert not check(worse(base, 8), base).regressed


def test_tasks_missing_on_one_side_are_left_out():
    base = outcomes(36, 4)
    now = worse(base, 6)
    now["t00"] = None  # invalid tonight
    now["new"] = False  # added since the baseline
    verdict = check(base, now)
    assert verdict.paired.n == 39
    assert "t00" not in verdict.lost
    assert "new" not in verdict.lost
    assert not verdict.regressed  # 5 lost left


def test_confirmation_round_drops_flaky_tasks():
    base = outcomes(36, 4)
    now = worse(base, 7)
    assert check(base, now).regressed
    # Two lost tasks pass in the rerun: 5 lost remain, p = 0.0625.
    assert not confirm(base, now, {"t00": True, "t01": True, "t02": False}).regressed
    # Failures or invalid reruns keep the first result; unknown tasks are ignored.
    assert confirm(base, now, {"t00": None, "t01": False, "zz": True}).regressed


def task(id_: str, *integrations: str) -> EvalTask:
    return EvalTask(id=id_, integrations=integrations, category="read", split="test", prompt="")


TASKS = {
    "a": task("a", "jira"),
    "b": task("b", "jira"),
    "c": task("c", "gdrive", "slack"),
}


def test_fingerprint_depends_on_provider_variant_and_integrations_only():
    fp = regression_fingerprint("gemini", "v2", ["a"], TASKS)
    assert fp == regression_fingerprint("gemini", "v2", ["b", "a", "unknown"], TASKS)
    assert fp != regression_fingerprint("gemini", "v2", ["a", "c"], TASKS)
    assert fp != regression_fingerprint("gemini", "v1", ["a"], TASKS)
    assert fp != regression_fingerprint("groq", "v2", ["a"], TASKS)
    assert len(fp) == 12


def score(task_: str, valid: bool = True, solved: bool = True) -> Score:
    return Score(task_, valid, solved if valid else None, None, None, ())


def test_mostly_invalid_runs_are_an_infra_problem():
    assert infra_problem([]) == "the run produced no results"
    assert infra_problem([score("a"), score("b", valid=False)]) is None
    problem = infra_problem([score("a"), score("b", valid=False), score("c", valid=False)])
    assert problem == "2 of 3 runs are invalid (model errors)"


def test_cause_is_code_when_code_or_tasks_changed():
    assert classify(["integrations/jira/descriptions/v2.yaml"], "m", "m") == "code changed"
    assert classify(["evals/tasks.yaml", "README.md"], "m", "n") == "code changed"
    assert classify(["README.md", ".github/workflows/ci.yml"], "m", "m") == "model drift"
    assert classify([], "m-001", "m-002") == "model drift (m-001 -> m-002)"


def test_body_has_marker_table_excerpts_and_commit_range():
    base = outcomes(36, 4)
    verdict = check(base, worse(base, 6))
    record: dict[str, Any] = {
        "task": "t00",
        "outcome": "answered",
        "answer": "x" * 400,
        "turns": [
            {
                "tool_calls": [{"name": "jira_search", "arguments": {}}],
                "results": [{"is_error": True}],
            }
        ],
    }
    body = regression_body(
        verdict,
        fingerprint_="abc123",
        provider="gemini",
        variant="v2",
        cause="code changed",
        compare_url="https://github.com/o/r/compare/aaa...bbb",
        scores=[Score("t00", True, False, False, 1, ("wrong tool",))],
        records=[record],
    )
    assert body.startswith(MARKER.format("abc123"))
    assert "solved 90% -> 75% on 40 paired tasks (lost 6, gained 0" in body
    assert "| `t00` | solved | not solved | wrong tool |" in body
    assert "| `t05` | solved | not solved |  |" in body
    assert "- `jira_search` (error)" in body
    assert "> " + "x" * 300 + " ..." in body
    assert body.count("<details>") == 1  # only tasks with a record get an excerpt
    assert "Cause: **code changed**. Commits since the baseline: https://github" in body


class FakeGitHub:
    """Open issues of one repo, paginated two per page, plus the calls made."""

    def __init__(self, issues: list[dict[str, Any]]):
        self.issues = issues
        self.calls: list[tuple[str, str, Any]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        assert request.headers["authorization"] == "Bearer tok"
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, request.url.path, body))
        if request.method == "GET":
            page = int(request.url.params.get("page", "1"))
            label = request.url.params.get("labels", "")
            assert request.url.params.get("state", "open") == "open"
            matching = [i for i in self.issues if label in i.get("labels", [label])]
            chunk = matching[(page - 1) * 2 : page * 2]
            headers = {}
            if page * 2 < len(matching):
                nxt = f"https://api.github.com/repos/o/r/issues?page={page + 1}&labels={label}"
                headers["link"] = f'<{nxt}>; rel="next"'
            return httpx2.Response(200, json=chunk, headers=headers)
        if request.url.path.endswith("/comments"):
            return httpx2.Response(201, json={"id": 1})
        return httpx2.Response(201, json={"number": 99})


def github(fake: FakeGitHub) -> GitHubIssues:
    client = httpx2.Client(base_url="https://api.github.com", transport=httpx2.MockTransport(fake))
    return GitHubIssues("o/r", "tok", client)


def test_second_run_comments_instead_of_opening_a_duplicate():
    fake = FakeGitHub(
        [
            {"number": 1, "body": "unrelated"},
            {"number": 2, "body": None, "pull_request": {}},
            {"number": 3, "body": MARKER.format("other")},
            {"number": 7, "body": f"{MARKER.format('fp1')}\nold table"},
        ]
    )
    filed = file_issue(github(fake), REGRESSION_LABEL, "fp1", "title", "new table")
    assert filed.number == 7
    assert not filed.created
    assert fake.calls[-1] == ("POST", "/repos/o/r/issues/7/comments", {"body": "new table"})
    assert [c[0] for c in fake.calls] == ["GET", "GET", "POST"]  # followed the Link header


def test_new_fingerprint_opens_an_issue_with_its_label():
    fake = FakeGitHub([{"number": 7, "body": MARKER.format("fp1")}])
    filed = file_issue(github(fake), INFRA_LABEL, "fp2", "Eval infra: gemini", "body")
    assert filed.number == 99
    assert filed.created
    assert fake.calls[-1] == (
        "POST",
        "/repos/o/r/issues",
        {"title": "Eval infra: gemini", "body": "body", "labels": ["eval-infra"]},
    )


def test_api_errors_are_raised():
    def deny(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(403, json={"message": "Resource not accessible"})

    client = httpx2.Client(base_url="https://api.github.com", transport=httpx2.MockTransport(deny))
    with pytest.raises(httpx2.HTTPStatusError):
        file_issue(GitHubIssues("o/r", "tok", client), REGRESSION_LABEL, "fp", "t", "b")
