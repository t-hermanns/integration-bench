# Integration template

[Copier](https://copier.readthedocs.io) template for one integration package under
`integrations/<name>`, extracted from the hand-built GitHub integration. Only the
`template/` directory is copied; the questions are in `questions.yml`.

```bash
uv run bench new jira -d auth=oauth_refresh -d pagination=next_page_token  # asks the rest
uv run bench update jira  # later: merge template changes into integrations/jira
```

`bench new` renders the package, adds it to the root `pyproject.toml`, runs `uv sync` and
writes the synthetic cassettes (`bench synth`). `--defaults` skips the questions that
`-d KEY=VALUE` does not answer.

The copier entry point is `copier.yml` in the repo root, which includes `questions.yml`:
copier records the template version (`_commit` in `.copier-answers.yml`) only for a git
repository root, and `bench update` (`copier update`) merges the changes since that commit,
keeping local edits. Both commands render from the committed `HEAD`, so commit template
changes first (`bench new --allow-dirty` renders the working tree, but the result cannot
be updated later).

## Questions

| Question | Choices | Decides |
|---|---|---|
| `name` | 2-15 lowercase letters/digits | package `bench_<name>`, tool prefix `<name>_` |
| `provider` | display name | the name the model reads in messages |
| `base_url` | `https://...`, may carry a path prefix | `ApiClient` base URL; fake routes include the prefix |
| `auth` | `oauth_refresh`, `client_credentials`, `bearer`, `basic` | `auth.py`, token endpoint in the fake, auth tests |
| `token_url` | only for the two OAuth kinds | `TokenEndpoint` URL |
| `pagination` | `link_header`, `next_page_token`, `page_token`, `cursor`, `next_cursor` | `bench_core.http` paginator, list shape in the fake |
| `retry` | `slack`, `jira`, `gdrive`, `github` | `bench_core.http` retry policy |
| `error_format` | `message`, `jira`, `google`, `slack`, `error` | `raise_for_status` and the fake's error bodies |

## What it generates

An example resource ("items") with three tools (`list`, `get`, `create`) wired through
every layer the GitHub integration has: typed client, models, scopes, tools with
`response_format`, server, v1 descriptions, a fake backend in the chosen API style,
cassette scenarios, and tests against cassettes, the server and the auth/retry paths.
After `bench synth <name>` the package is green; then replace "items" with the real API.

## Checking template changes

The `generator` CI job renders five combinations that together use every answer, then
runs ruff (check and format), pyright and pytest on each. Locally: commit, render into
`integrations/` with `bench new`, check, then remove the package and reset `pyproject.toml`
and `uv.lock`. The template must render code that is already formatted, so keep lines
whose length depends on `name` or `provider` well under the limit (CI tests a
15-character name and a 30-character provider).
