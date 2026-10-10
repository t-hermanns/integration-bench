"""Monthly call budget for providers with a hard monthly cap (the Cohere trial key: 1,000 calls).

The counter is a JSON file `{"YYYY-MM": calls}` (UTC calendar month) kept in the `bench-results`
branch (`bench_evals.results_branch`). Every request counts, failed ones and retries included,
because Cohere does not say which ones it bills and sends no "remaining this month" header.
The run stops before the request that would pass `stop_at`; the gap to the real cap is the buffer
for uncounted calls (`bench doctor`, manual checks).
"""

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from bench_evals.models.base import Message, Model, Reply, Tool, ToolChoice

BUDGETS = {"cohere": 900}
"""Provider -> calls per calendar month after which runs stop (Cohere trial cap: 1,000)."""


class BudgetExhausted(Exception):
    """The monthly budget is used up; the run must stop (not a per-task model error)."""


def this_month() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


class CallBudget:
    def __init__(self, path: Path, stop_at: int, month: Callable[[], str] = this_month):
        self.path = path
        self.stop_at = stop_at
        self.month = month

    def counts(self) -> dict[str, int]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def used(self) -> int:
        return self.counts().get(self.month(), 0)

    def left(self) -> int:
        return max(self.stop_at - self.used(), 0)

    def add(self, calls: int) -> int:
        """Record calls (also manual corrections) and return the month's total."""
        counts = self.counts()
        month = self.month()
        counts[month] = counts.get(month, 0) + calls
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(dict(sorted(counts.items())), indent=2) + "\n"
        self.path.write_bytes(text.encode())  # written after every call, so a crash loses none
        return counts[month]

    def charge(self) -> None:
        """Count one call before it is made; raises when the budget has no call left."""
        used = self.used()
        if used >= self.stop_at:
            raise BudgetExhausted(
                f"monthly call budget used up: {used}/{self.stop_at} in {self.month()}"
            )
        self.add(1)


class BudgetedModel:
    """Wraps a model so that every request is charged to the budget first."""

    def __init__(self, model: Model, budget: CallBudget):
        self.inner = model
        self.budget = budget
        self.provider = model.provider
        self.model = model.model

    async def complete(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[Tool] = (),
        tool_choice: ToolChoice = "auto",
    ) -> Reply:
        self.budget.charge()
        return await self.inner.complete(system, messages, tools, tool_choice)
