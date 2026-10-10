# Integration template

[Copier](https://copier.readthedocs.io) template for one integration package under
`integrations/<name>`, extracted from the hand-built GitHub integration. Only the
`template/` directory is copied; this README and `copier.yml` stay here.

## Questions

| Question | Choices | Decides |
|---|---|---|
| `name` | 2-15 lowercase letters/digits | package `bench_<name>`, tool prefix `<name>_` |
| `provider` | display name | the name the model reads in messages |
| `base_url` | `https://...`, may carry a path prefix | `ApiClient` base URL; fake routes include the prefix |
| `auth` | `oauth_refresh`, `client_credentials`, `bearer`, `basic` | `auth.py`, token endpoint in the fake, auth tests |
| `token_url` | only for the two OAuth kinds | `TokenEndpoint` URL |
| `pagination` | `link_header`, `next_page_token`, `page_token`, `cursor` | `bench_core.http` paginator, list shape in the fake |
| `retry` | `slack`, `jira`, `gdrive`, `github` | `bench_core.http` retry policy |
| `error_format` | `message`, `jira`, `google`, `slack` | `raise_for_status` and the fake's error bodies |

## What it generates

An example resource ("items") with three tools (`list`, `get`, `create`) wired through
every layer the GitHub integration has: typed client, models, scopes, tools with
`response_format`, server, v1 descriptions, a fake backend in the chosen API style,
cassette scenarios, and tests against cassettes, the server and the auth/retry paths.
After `bench synth <name>` the package is green; then replace "items" with the real API.

## Checking template changes

Render each answer combination into `integrations/`, then run ruff (check and format),
pyright, `bench synth` and pytest on the result. The template must render code that is
already formatted, so keep lines whose length depends on `name` or `provider` well
under the limit (test with a 15-character name and a 30-character provider).
