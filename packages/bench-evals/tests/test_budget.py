import json
import subprocess
from pathlib import Path

import anyio
import pytest

from bench_evals import results_branch
from bench_evals.budget import BudgetedModel, BudgetExhausted, CallBudget
from bench_evals.models.base import Assistant, User
from bench_evals.models.scripted import ScriptedModel


def budget(path: Path, stop_at: int = 3, month: str = "2026-10") -> CallBudget:
    return CallBudget(path / "budget" / "cohere.json", stop_at, month=lambda: month)


def test_counts_per_calendar_month_and_stops_before_the_limit(tmp_path: Path) -> None:
    calls = budget(tmp_path)
    assert (calls.used(), calls.left()) == (0, 3)
    for _ in range(3):
        calls.charge()
    with pytest.raises(BudgetExhausted, match="3/3 in 2026-10"):
        calls.charge()
    assert calls.used() == 3

    november = budget(tmp_path, month="2026-11")
    november.charge()
    assert json.loads(calls.path.read_text()) == {"2026-10": 3, "2026-11": 1}
    assert calls.path.read_bytes().count(b"\r") == 0


def test_manual_correction_counts_against_the_month(tmp_path: Path) -> None:
    calls = budget(tmp_path, stop_at=900)
    assert calls.add(16) == 16
    assert calls.left() == 884


def test_budgeted_model_charges_every_request_and_stops_the_run(tmp_path: Path) -> None:
    calls = budget(tmp_path, stop_at=2)
    model = BudgetedModel(ScriptedModel([Assistant(text="a"), Assistant(text="b")]), calls)
    assert (model.provider, model.model) == ("scripted", "scripted")

    async def ask() -> str:
        return (await model.complete("sys", [User("hi")])).message.text

    assert anyio.run(ask) == "a"
    assert anyio.run(ask) == "b"
    with pytest.raises(BudgetExhausted):
        anyio.run(ask)
    assert calls.used() == 2


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def clone_with_remote(tmp_path: Path, name: str) -> Path:
    remote = tmp_path / "remote.git"
    if not remote.exists():
        git("init", "--bare", "-b", "main", str(remote), cwd=tmp_path)
    repo = tmp_path / name
    git("clone", str(remote), str(repo), cwd=tmp_path)
    git("config", "user.email", "bench@example.com", cwd=repo)
    git("config", "user.name", "bench", cwd=repo)
    return repo


def test_results_branch_is_created_as_orphan_and_shared_between_clones(tmp_path: Path) -> None:
    first = clone_with_remote(tmp_path, "first")
    tree = results_branch.checkout(first)
    calls = budget(tree)
    calls.charge()
    assert results_branch.publish(tree, ["budget/cohere.json"], "budget: 1")
    assert not results_branch.publish(tree, ["budget/cohere.json"], "unchanged")
    assert git("log", "--format=%s", "bench-results", cwd=first).split() == ["budget:", "1"]

    second = clone_with_remote(tmp_path, "second")
    other = budget(results_branch.checkout(second))
    assert other.used() == 1
    other.charge()
    results_branch.publish(other.path.parent.parent, ["budget/cohere.json"], "budget: 2")

    assert budget(results_branch.checkout(first)).used() == 2  # fast-forwarded
