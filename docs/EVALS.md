# Evals

Live evals run each task through a model, the task's MCP servers and the real sandbox
APIs (`bench_evals.runner`), then score the logged record offline (`bench_evals.scorer`).

## Tasks

`integrations/<name>/evals/tasks.yaml`, plus `evals/tasks.yaml` for tasks that span
integrations. Format and validation: `bench_evals.tasks`; argument matchers:
`bench_evals.matchers`.

```yaml
integrations: [jira]
tasks:
  - id: jira-create-bug
    category: multi_step_write   # read, arguments, multi_step_write, pagination,
                                 # permission, abstention, confusable
    split: dev                   # dev (for iterating descriptions) or test
    prompt: "Create a bug 'Checkout crash {run_id}' in BENCH."
    expected_calls:
      - tool: jira_create_issue
        args:
          project_key: BENCH
          summary: {$contains: "{run_id}"}
          issue_type: bug        # strings ignore case
    forbidden_tools: [jira_transition_issue]
    answer_contains: [[BENCH-, created]]   # all items; each item: any alternative
```

## Metrics

Per task and run:

- **solved**: final answer given, every expected call matched by its own call (arguments
  match; result succeeded, or failed for `error: true`), no forbidden tool called, answer
  contains every `answer_contains` item.
- **right_tool**: every expected tool called at least once, no forbidden tool called.
- **calls**: number of tool calls.

Runs where the provider failed (`model_error`) are invalid and left out.

## Headline metric (fixed before the first measurement)

The description experiment (v1 vs. v2) is judged on **solved**, per task, on the test
split. With k > 1 runs, a task counts as solved if it was solved in more than half of
its valid runs. v1 and v2 are compared with the exact McNemar test on the paired tasks,
with 95 % Wilson intervals for each rate. `right_tool` and `calls` are reported but not
tested.

Implementation: `bench_evals.stats` (`task_outcomes` → `rate` / `compare`). Ties (1 of 2
runs) count as not solved; tasks with no valid run, or with an outcome in only one
variant, are left out of the comparison and counted as `unpaired`.

## Description variants

Each integration has `descriptions/v1.yaml` (short, OpenAPI-summary style) and
`descriptions/v2.yaml`, written for every tool after Anthropic's guide "Writing
effective tools for agents": what the tool is for and when a sibling tool fits better,
argument formats with examples, limits, the response shape, pagination and errors. v2
was tuned only on failures in the dev split (with Gemini); a tool whose v2 text did
clearly worse there than v1 would have kept its v1 text. The test split stays untouched
until the one measurement.

## Running

```bash
bench eval gemini --split dev --descriptions v2
```

runs the tasks live against the sandbox (writes leave `{run_id}` objects behind), appends
one JSON line per run to `.bench/results/` and prints the score. `--k 3` repeats every
task, `--task ID` picks single tasks. `bench score a.jsonl b.jsonl` scores saved logs and
compares b against a (exact McNemar). `bench report` writes [REPORT.md](../REPORT.md)
from `.bench/results/exp-<provider>-<v1|v2>-test.jsonl` (plus
`baseline-<provider>-v2-all.jsonl`) and `evals/REPORT.template.md`.

## Call budget

The Cohere trial key allows 1,000 calls per month and reports no remaining count, so `bench eval cohere`
counts every request itself (retries and failed requests included) per UTC calendar month and stops
before call 900. The counter is `budget/cohere.json` in the orphan branch `bench-results`, checked out
as a worktree in `.bench/bench-results` and pushed after each run. `bench budget` shows it;
`bench budget --add N` records calls made elsewhere (the remaining 100 cover `bench doctor` and manual
checks). Whether the cap is per endpoint and when it resets is not documented; the counter assumes
one shared cap that resets with the calendar month.

## Regression rule (nightly, fixed before the first nightly run)

`bench_evals.regression` compares tonight's per-task outcomes with the baseline, paired by task. It is
a regression if the solved rate dropped by at least 5 percentage points **and** the exact McNemar test
gives p < 0.05. The lost tasks are then run once more; a lost task that passes in this confirmation
round counts as flaky and the rule is applied again. No trend analysis: one baseline, one run.

A confirmed regression files a GitHub issue labelled `eval-regression`. Its fingerprint (provider,
description variant, affected integrations) sits in a hidden marker in the body; if an open issue with
the same fingerprint exists, the run comments there instead of opening a duplicate. The body has the
lost tasks, transcript excerpts, the commit range since the baseline and the cause: "code changed" if
the range touches `integrations/`, `packages/` or `evals/`, otherwise "model drift". A run where more
than half of the runs are invalid is not judged; it files an `eval-infra` issue instead.

`bench regress gemini` does all of this (`.github/workflows/nightly.yml`, Gemini over all tasks with
k=1). A failing `bench doctor` or a crashed run also files an `eval-infra` issue. The baseline is
`baselines/<provider>-<variant>.json` in the `bench-results` branch (outcomes, commit, model ID); the
first run sets it and it only moves with `--set-baseline`, so slow drift adds up instead of hiding.
Every judged run is appended to `history/<provider>-<variant>.jsonl` there; the full logs are workflow
artifacts. The schedule only runs while the repo variable `NIGHTLY_ENABLED` is `true`.
`.github/workflows/eval.yml` runs `bench eval` by hand (provider, integration, variant, split, k).
