"""Statistics for the description experiment: per-task outcomes, Wilson, exact McNemar.

The unit of analysis is the task, not the run (see the headline metric in
`bench_evals.scorer`):

1. `task_outcomes` folds the k runs of each task into one outcome: True if the metric
   held in more than half of the task's valid runs (2 of 3, 2 of 2, 1 of 1), False
   otherwise, None if the task has no valid run.
2. `rate` gives the share of tasks with outcome True and its 95 % Wilson interval.
3. `compare` pairs two variants (v1, v2) on the tasks that have an outcome in both and
   runs the exact McNemar test on the discordant pairs.

No bootstrap: with ~24 test tasks the exact test and Wilson intervals are enough.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from scipy.stats import binomtest

from bench_evals.scorer import HEADLINE, Score

CONFIDENCE = 0.95


@dataclass(frozen=True)
class Rate:
    """`successes` of `n` tasks, with a Wilson interval; `n == 0` has no rate."""

    successes: int
    n: int
    low: float | None
    high: float | None

    @property
    def value(self) -> float | None:
        return self.successes / self.n if self.n else None


@dataclass(frozen=True)
class Paired:
    """Variant a against variant b on the tasks with an outcome in both.

    `only_a`/`only_b` are the discordant tasks (true for one variant only); the exact
    McNemar p-value (two-sided) depends on them alone. `unpaired` counts tasks that
    have an outcome in just one variant: they are left out, not counted as failures.
    """

    n: int
    a: Rate
    b: Rate
    only_a: int
    only_b: int
    p_value: float
    unpaired: int

    @property
    def difference(self) -> float | None:
        """Rate of b minus rate of a, on the paired tasks."""
        return (self.only_b - self.only_a) / self.n if self.n else None


def task_outcomes(scores: Iterable[Score], metric: str = HEADLINE) -> dict[str, bool | None]:
    """One outcome per task: the metric held in more than half of its valid runs."""
    runs: defaultdict[str, list[bool]] = defaultdict(list)
    for s in scores:
        values = runs[s.task]  # a task with only invalid runs still gets an entry (None)
        if not s.valid:
            continue
        value = getattr(s, metric)
        if not isinstance(value, bool):
            raise TypeError(f"{metric} of {s.task} is {value!r}, not a bool")
        values.append(value)
    return {task: (2 * sum(v) > len(v) if v else None) for task, v in runs.items()}


def wilson(successes: int, n: int, confidence: float = CONFIDENCE) -> Rate:
    if not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n, got {successes}/{n}")
    if n == 0:
        return Rate(0, 0, None, None)
    ci = binomtest(successes, n).proportion_ci(confidence_level=confidence, method="wilson")
    return Rate(successes, n, float(ci.low), float(ci.high))


def rate(outcomes: Mapping[str, bool | None], confidence: float = CONFIDENCE) -> Rate:
    """Share of tasks with outcome True; tasks without an outcome are left out."""
    known = [v for v in outcomes.values() if v is not None]
    return wilson(sum(known), len(known), confidence)


def compare(
    a: Mapping[str, bool | None],
    b: Mapping[str, bool | None],
    confidence: float = CONFIDENCE,
) -> Paired:
    """Exact McNemar test of a against b, paired by task."""
    known_a = {t for t, v in a.items() if v is not None}
    known_b = {t for t, v in b.items() if v is not None}
    paired = known_a & known_b
    both = sum(1 for t in paired if a[t] and b[t])
    only_a = sum(1 for t in paired if a[t] and not b[t])
    only_b = sum(1 for t in paired if b[t] and not a[t])
    # Exact McNemar: two-sided binomial test of the discordant pairs against 1/2.
    # No discordant pair at all is no evidence of a difference (p = 1).
    discordant = only_a + only_b
    p_value = float(binomtest(only_b, discordant, 0.5).pvalue) if discordant else 1.0
    return Paired(
        n=len(paired),
        a=wilson(both + only_a, len(paired), confidence),
        b=wilson(both + only_b, len(paired), confidence),
        only_a=only_a,
        only_b=only_b,
        p_value=p_value,
        unpaired=len(known_a ^ known_b),
    )
