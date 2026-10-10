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
