# Adding an integration

A new integration took **10 minutes** from "no account" to "live cassettes recorded and
smoke test green": Todoist (`integrations/todoist`), October 2026, with Claude Code
editing the generated code. The two commits show what the template gives you and what
was left to do by hand:

- `82fcfe5`: raw output of `bench new`, green out of the box (25 tests)
- `292ac27`: the example resource replaced with the real API

| Minute | Step |
|---|---|
| 0-2 | Free account, API token into `.env` |
| 2-4 | Probe the API with `curl`, pick the template answers, `bench new` |
| 4-8 | Replace the example "items" with the real resource |
| 8-10 | `bench sandbox seed`, `bench record`, `bench doctor`, `bench smoke` |

## 1. Account and token

Use a free plan and a dedicated test account; nothing in it may matter. Put the secret
into `.env` in the repo root (gitignored), named `<NAME>_TOKEN` for a static token. The
other auth kinds use `<NAME>_CLIENT_ID`/`_CLIENT_SECRET` or `<NAME>_EMAIL`/`_API_TOKEN`
(see the generated `sandbox.py`). Prefer the narrowest credential the provider offers: a
Todoist personal token can do everything in the account, which is acceptable only because
the account holds nothing but seed data.

## 2. Look at the real API first

Before generating, make four read-only calls with `curl` and note the answers. They map
one-to-one to the template questions:

| Look at | Template question |
|---|---|
| a list endpoint: where are the items, where is the next-page token? | `pagination` |
| an unknown id and a bad argument: status code and error body | `error_format` |
| a call with a wrong token: 401? | `auth` |
| response headers: `Retry-After`, `x-ratelimit-*`? | `retry` |

Real APIs differ from their docs in small ways that matter for the fake. Todoist, for
example, answers a malformed id with 400 but an unknown well-formed id with 404, sends
`Retry-After` even on 400, and accepts a 2000-character title.

## 3. Generate

```bash
uv run bench new todoist --defaults -d provider=Todoist \
  -d base_url=https://api.todoist.com/api/v1 -d auth=bearer \
  -d pagination=next_cursor -d retry=slack -d error_format=error
uv run pytest integrations/todoist
git add integrations/todoist pyproject.toml uv.lock && git commit -m "todoist: raw output of bench new"
```

Commit the raw output on its own: `bench update todoist` later merges template changes
against exactly this commit. The template's questions are listed in
[templates/integration/README.md](../templates/integration/README.md). If no answer fits,
take the closest one and adapt `client.py` and `fake/backend.py`; then add the missing
answer to the template, as `next_cursor` and `error` were added after the Todoist run.

## 4. Replace the example resource

The package ships an example resource ("items") with `list`, `get` and `create` tools in
every layer. Rename it (`sed 's/Item/Task/g; s/item/task/g'` over `src` and `tests`) and
then fit each layer to the real API:

| File | What to change |
|---|---|
| `models.py` | field names and aliases (`Field(validation_alias="content")`), derived fields |
| `client.py` | paths, query parameters, request bodies, the key that holds the list |
| `fake/backend.py` | the same routes in the provider's raw JSON shape, including error bodies |
| `fake/seed.py` | seed data; `bench sandbox seed` creates the same data live |
| `scenarios.py` | ids that exist live (look them up instead of hard-coding), a `teardown` for anything a scenario creates |
| `sandbox.py` | `seed` (idempotent) and the read-only `SMOKE` calls |
| `doctor.py` | `PROBE`, a cheap authenticated GET |
| `descriptions/v1.yaml` | what the model reads about each tool |

Delete the synthetic cassettes before renaming scenarios, then run `uv run bench synth
<name>` and the tests until they are green against the fake.

## 5. Go live

```bash
uv run bench sandbox seed todoist   # once; reruns change nothing
uv run bench record todoist         # replaces synthetic cassettes with recorded ones
uv run pytest integrations/todoist
uv run bench doctor todoist
uv run bench smoke todoist
```

`bench record` redacts auth headers and cookies and refuses cassettes that match known
secret patterns; read the diff anyway before committing (account ids, e-mail addresses,
signed URLs). Scenarios that need a shaped fake (for example a token without the write
scope) stay synthetic.

## What is not included

Eval tasks (`evals/tasks.yaml`), v2 descriptions and CI secrets are separate steps; see
[EVALS.md](EVALS.md) and [SETUP.md](SETUP.md#ci-secrets).
