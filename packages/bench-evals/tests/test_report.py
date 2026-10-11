import pytest

from bench_evals.report import ModelReport, pass_k, render, runs_per_task, section, summary
from bench_evals.scorer import Score
from bench_evals.stats import compare, wilson


def s(task: str, solved: bool | None, valid: bool = True) -> Score:
    return Score(task, valid, solved, solved, 1)


def outcomes(solved: int, n: int) -> dict[str, bool | None]:
    return {f"t{i}": i < solved for i in range(n)}


def model(title: str, a: int, b: int, k: int = 3, n: int = 25) -> ModelReport:
    paired = compare(outcomes(a, n), outcomes(b, n))
    return ModelReport(
        title=title,
        model=title.lower(),
        k=k,
        tasks=n,
        solved=paired,
        right_tool=paired,
        calls=(1.25, None),
        only_v1=tuple(f"t{i}" for i in range(b, a)),
        only_v2=tuple(f"t{i}" for i in range(a, b)),
        reliability=paired if k > 1 else None,
        baseline=wilson(50, 62) if k > 1 else None,
        baseline_tasks=63,
        categories={"t13": "read", "t14": "pagination", "t15": "read"},
    )


def test_pass_k_needs_k_valid_runs_all_solved():
    scores = [s("a", True)] * 3 + [s("b", True), s("b", False), s("b", True)]
    scores += [s("c", True), s("c", True), s("c", None, valid=False)]
    assert pass_k(scores, 3) == {"a": True, "b": False, "c": None}
    assert runs_per_task(scores) == 3


def test_summary_names_headline_model_first():
    text = summary([model("Alpha", 13, 19), model("Beta", 20, 23), model("Gamma", 22, 24)])
    assert text.startswith(
        "- **Alpha:** v2 raises solved test tasks from 13 to 19 of 25 (52% → 76%, +24 points):"
        " 6 gained, 0 lost, exact McNemar p = 0.031, significant at the 5% level."
    )
    assert "Gained tasks by category: ? 3, read 2, pagination 1." in text
    assert "all 3 runs rise from 13 to 19 of 25" in text
    assert "- **Beta and Gamma:** same direction, not significant on their own" in text


def test_summary_flags_mixed_directions():
    text = summary([model("Alpha", 13, 19), model("Beta", 23, 20)])
    assert "not all in the same direction" in text


def test_section_leads_with_headline_and_lists_changed_tasks():
    text = section(model("Alpha", 13, 15))
    assert "## Alpha\n\n**v2 raises solved test tasks from 13 to 15 of 25" in text
    assert "| **solved** (majority of 3 runs) | 13/25 (52%" in text
    assert "| tool calls per run (mean) | 1.2 | n/a |" in text
    assert "- Solved only with v2: `t13` (read), `t14` (pagination)" in text
    assert "- Solved only with v1: none" in text
    assert "50/62" in text
    assert "1 task(s) without a valid run" in text


def test_section_of_a_loss():
    assert "**v2 lowers solved test tasks from 15 to 13 of 25" in section(model("A", 15, 13))


def test_render_fills_template():
    text = render(
        "$generated\n$summary\n$models\n$reliability\n$$x\n",
        [model("Alpha", 13, 19), model("Gamma", 22, 24, k=1)],
        "today",
    )
    assert text.startswith("today\n- **Alpha:** v2 raises")
    assert "## Gamma" in text
    assert "Gamma: only 1 run per task, no pass^k." in text
    assert "Alpha (pass^3): v1 13/25" in text
    assert text.endswith("$x\n")
    with pytest.raises(ValueError, match="no models"):
        render("$summary", [], "today")
