# Report: tool-description tuning

$generated Template: `evals/REPORT.template.md`.

This report is about the integrations, not the models: do better tool descriptions help
a model use the MCP servers in this repository (Jira, GitHub, Google Drive, Slack; 21
tools) correctly? Each model is compared only with itself, v1 descriptions against v2.
The models differ in runs per task and in how their runs were scheduled, so their
numbers are not comparable with each other and this is not a ranking.

## Summary

$summary

## What v2 fixed (Command A)

From the logs of the six tasks Command A solves only with v2; v1 failed the same way in
all three runs:

- **Where an ID comes from (Slack, 3 tasks):** with v1 it passed the channel name
  `#bench-sandbox` as `channel_id`. The server rejected it and pointed to
  `slack_list_channels`, but the model gave up. v2 says the ID comes from the channel
  listing, and the model lists channels first.
- **What a read tool returns (Jira, 1 task):** with v1 it reported the status but said
  it had no tool for the platform. v2 says `jira_get_issue` returns the full description
  with environment details, where the platform is.
- **Paging (Drive, 1 task):** with v1 the request for the second page failed, so the
  answer missed a file; with v2 it fetched the second page. The logs do not show why the
  v1 request was rejected.
- **The right tool for the question (Drive, 1 task):** asked for a file's MIME type,
  with v1 it read it off the search result instead of calling `gdrive_get_file_metadata`,
  the tool for file details that the task checks. v2 says which of the two tools fits,
  and the model calls the metadata tool.

## Method

- **Variants:** `descriptions/v1.yaml` (short, OpenAPI-summary style) and
  `descriptions/v2.yaml` (purpose, when a sibling tool fits better, argument formats,
  limits, response shape, pagination, errors), tuned only on dev-split failures. Details:
  [docs/EVALS.md](docs/EVALS.md#description-variants).
- **Tasks:** 25 held-out test tasks of 63 (read, arguments, multi-step write,
  pagination, permission, abstention, confusable tools), 6 per integration plus one
  cross-integration task.
- **Runs:** live, against one sandbox account per service; the sandbox is reset between
  runs. For Command A and Gemini, v1 and v2 runs alternate; the Qwen v2 run is its
  earlier baseline run on the test split.
- **Headline metric:** `solved`, fixed before the first measurement: final answer,
  every expected call matched, no forbidden tool, required answer content. With k > 1
  runs a task counts as solved if more than half of its valid runs solved it.
- **Statistics:** 95% Wilson intervals per rate; v1 and v2 paired by task, exact McNemar
  test on the discordant tasks. `right_tool` and tool calls are reported, not tested.

$models

## Secondary, post hoc: reliability (pass^3)

Added after the experiment, not planned before it, and computed offline from the same
logs: a task counts only if every run solved it. Read it as a description of the data,
not as a confirmed result.

$reliability

## Limitations

- **Test prompts were visible** while v2 was written. v2 was tuned only on dev-split
  failures and committed before the first test run, but it was not written blind.
- **Few tasks, ceiling effect:** with 25 tasks, at least 6 tasks must change (all in
  the same direction) for p < 0.05. Models that already solve most tasks cannot show a
  significant gain.
- **Lenient argument checks for Slack and Drive:** IDs and timestamps the services
  assign themselves are only checked for presence (`$$present`), text arguments by
  substring; extra tool calls are allowed; scoring reads the call log, not the
  resulting state of the service.
- **One sandbox per service:** all runs share it, so leftovers of one run could affect
  the next; the reset between runs is the only guard.
- **pass^3 is post hoc** (see above).
