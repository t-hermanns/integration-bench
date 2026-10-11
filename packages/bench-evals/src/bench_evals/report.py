"""REPORT.md from the experiment logs: one section per model, v1 against v2.

Each model is compared only with itself (v1 vs. v2 on the same tasks); there is no
table across models. The template (`evals/REPORT.template.md`) holds the prose and
`$placeholders` for the generated parts.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import mean
from string import Template
from typing import Any

from bench_evals.scorer import Score, score_run
from bench_evals.stats import Paired, Rate, compare, rate, task_outcomes
from bench_evals.tasks import EvalTask

ALPHA = 0.05


@dataclass(frozen=True)
class ModelReport:
    title: str
    model: str
    k: int  # runs per task
    tasks: int  # tasks in the experiment split
    solved: Paired  # headline: majority of the k runs
    right_tool: Paired
    calls: tuple[float | None, float | None]  # mean tool calls per valid run, v1 and v2
    only_v1: tuple[str, ...]
    only_v2: tuple[str, ...]
    reliability: Paired | None  # pass^k, post hoc; None for k = 1
    baseline: Rate | None  # v2 on all tasks, k = 1
    baseline_tasks: int
    categories: Mapping[str, str] = field(default_factory=dict[str, str])  # task -> category


def runs_per_task(scores: Iterable[Score]) -> int:
    counts: defaultdict[str, int] = defaultdict(int)
    for s in scores:
        if s.valid:
            counts[s.task] += 1
    return max(counts.values(), default=0)


def pass_k(scores: Iterable[Score], k: int) -> dict[str, bool | None]:
    """True if a task was solved in every one of at least k valid runs (pass^k)."""
    runs: defaultdict[str, list[bool]] = defaultdict(list)
    for s in scores:
        values = runs[s.task]
        if s.valid:
            values.append(bool(s.solved))
    return {t: (all(v) if len(v) >= k else None) for t, v in runs.items()}


def mean_calls(scores: Iterable[Score]) -> float | None:
    calls = [s.calls for s in scores if s.valid and s.calls is not None]
    return mean(calls) if calls else None


def model_report(
    title: str,
    tasks: Sequence[EvalTask],
    v1: Iterable[dict[str, Any]],
    v2: Iterable[dict[str, Any]],
    baseline: Iterable[dict[str, Any]] | None = None,
) -> ModelReport:
    v1, v2 = list(v1), list(v2)
    s1, s2 = score_run(tasks, v1), score_run(tasks, v2)
    a, b = task_outcomes(s1), task_outcomes(s2)
    k = min(runs_per_task(s1), runs_per_task(s2))
    paired = sorted(t for t in set(a) & set(b) if a[t] is not None and b[t] is not None)
    base = score_run(tasks, baseline) if baseline is not None else None
    return ModelReport(
        title=title,
        model=str(v2[0]["model"]) if v2 else "?",
        k=k,
        tasks=len({s.task for s in s2}),
        solved=compare(a, b),
        right_tool=compare(task_outcomes(s1, "right_tool"), task_outcomes(s2, "right_tool")),
        calls=(mean_calls(s1), mean_calls(s2)),
        only_v1=tuple(t for t in paired if a[t] and not b[t]),
        only_v2=tuple(t for t in paired if b[t] and not a[t]),
        reliability=compare(pass_k(s1, k), pass_k(s2, k)) if k > 1 else None,
        baseline=rate(task_outcomes(base)) if base is not None else None,
        baseline_tasks=len({s.task for s in base}) if base is not None else 0,
        categories={t.id: t.category for t in tasks},
    )


def show(r: Rate) -> str:
    if r.value is None or r.low is None or r.high is None:
        return "n/a"
    return f"{r.successes}/{r.n} ({r.value:.0%}, 95% CI {r.low:.0%}-{r.high:.0%})"


def show_p(p: float) -> str:
    return f"p = {p:.3f}"


def show_calls(c: float | None) -> str:
    return "n/a" if c is None else f"{c:.1f}"


def pct(r: Rate) -> str:
    return "n/a" if r.value is None else f"{r.value:.0%}"


def direction(p: Paired) -> int:
    return (p.only_b > p.only_a) - (p.only_b < p.only_a)


def significance(p: float) -> str:
    return ("significant" if p < ALPHA else "not significant") + f" at the {ALPHA:.0%} level"


def headline(m: ModelReport) -> str:
    """The v1 -> v2 change of the headline metric in one sentence (no final period)."""
    s = m.solved
    verb = {1: "raises", 0: "leaves", -1: "lowers"}[direction(s)]
    points = "" if s.difference is None else f", {s.difference * 100:+.0f} points"
    return (
        f"v2 {verb} solved test tasks from {s.a.successes} to {s.b.successes} of {s.n} "
        f"({pct(s.a)} → {pct(s.b)}{points}): {s.only_b} gained, {s.only_a} lost, "
        f"exact McNemar {show_p(s.p_value)}, {significance(s.p_value)}"
    )


def by_category(tasks: Sequence[str], categories: Mapping[str, str]) -> str:
    counts = Counter(categories.get(t, "?") for t in tasks)
    return ", ".join(f"{c} {n}" for c, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])))


def tasks_line(label: str, tasks: Sequence[str], categories: Mapping[str, str]) -> str:
    if not tasks:
        return f"- {label}: none"
    return f"- {label}: " + ", ".join(f"`{t}` ({categories.get(t, '?')})" for t in tasks)


def runs(k: int) -> str:
    return "1 run" if k == 1 else f"{k} runs"


def section(m: ModelReport) -> str:
    s = m.solved
    rule = "majority of " + runs(m.k) if m.k > 1 else runs(m.k)
    lines = [
        f"## {m.title}",
        "",
        f"**{headline(m)}.**",
        "",
        f"Model `{m.model}`, {m.tasks} test tasks, {runs(m.k)} per task and variant.",
        "",
        "| | v1 | v2 |",
        "|---|---|---|",
        f"| **solved** ({rule}) | {show(s.a)} | {show(s.b)} |",
        f"| right_tool | {show(m.right_tool.a)} | {show(m.right_tool.b)} |",
        f"| tool calls per run (mean) | {show_calls(m.calls[0])} | {show_calls(m.calls[1])} |",
        "",
        f"Paired on {s.n} tasks: v2 solves {s.only_b} that v1 does not, v1 solves "
        f"{s.only_a} that v2 does not; exact McNemar {show_p(s.p_value)}"
        + (f"; {s.unpaired} task(s) unpaired" if s.unpaired else "")
        + ".",
        "",
        tasks_line("Solved only with v2", m.only_v2, m.categories),
        tasks_line("Solved only with v1", m.only_v1, m.categories),
    ]
    if m.baseline is not None:
        lines += [
            "",
            f"Baseline (v2, all {m.baseline_tasks} tasks incl. dev, 1 run): "
            f"{show(m.baseline)} solved"
            + (
                f"; {m.baseline_tasks - m.baseline.n} task(s) without a valid run."
                if m.baseline.n < m.baseline_tasks
                else "."
            ),
        ]
    return "\n".join(lines)


def summary(models: Sequence[ModelReport]) -> str:
    """Bullets: the headline model in full, the others only by direction."""
    head, rest = models[0], models[1:]
    lines = [f"- **{head.title}:** {headline(head)}."]
    if head.only_v2:
        lines.append(f"  Gained tasks by category: {by_category(head.only_v2, head.categories)}.")
    r = head.reliability
    if r is not None:
        lines.append(
            f"- **{head.title}, post hoc:** tasks solved in all {head.k} runs rise from "
            f"{r.a.successes} to {r.b.successes} of {r.n}, so v2 also makes it more "
            "consistent (section below)."
            if direction(r) > 0
            else f"- **{head.title}, post hoc:** tasks solved in all {head.k} runs: "
            f"{r.a.successes} → {r.b.successes} of {r.n} (section below)."
        )
    if rest:
        same = all(direction(m.solved) == direction(head.solved) for m in rest)
        significant = [m.title for m in rest if m.solved.p_value < ALPHA]
        names = " and ".join(m.title for m in rest)
        if not same:
            lines.append(f"- **{names}:** not all in the same direction (sections below).")
        else:
            lines.append(
                f"- **{names}:** same direction, "
                + (
                    "not significant on their own (ceiling effect, see Limitations)."
                    if not significant
                    else "significant for " + ", ".join(significant) + "."
                )
            )
    return "\n".join(lines)


def reliability(models: Sequence[ModelReport]) -> str:
    lines: list[str] = []
    for m in models:
        r = m.reliability
        if r is None:
            lines.append(f"- {m.title}: only 1 run per task, no pass^k.")
            continue
        lines.append(
            f"- {m.title} (pass^{m.k}): v1 {show(r.a)}, v2 {show(r.b)}; "
            f"only v2 {r.only_b}, only v1 {r.only_a}, exact McNemar {show_p(r.p_value)}."
        )
    return "\n".join(lines)


def render(template: str, models: Sequence[ModelReport], generated: str) -> str:
    if not models:
        raise ValueError("no models to report")
    text = Template(template).substitute(
        generated=generated,
        summary=summary(models),
        models="\n\n".join(section(m) for m in models),
        reliability=reliability(models),
    )
    return text.rstrip("\n") + "\n"
