# integration-bench

MCP integrations for Jira Cloud, GitHub, Google Drive and Slack: scaffold, build, test, measure with evals, improve and monitor.

## Results

Rewriting the tool descriptions (`descriptions/v1.yaml` → `v2.yaml`) raised the share of
held-out test tasks that Cohere Command A solves from **13 to 19 of 25** (exact McNemar
p = 0.031, 3 runs per task, live sandbox accounts). Two other models moved in the same
direction; details, method and limitations are in [REPORT.md](REPORT.md), the metric in
[docs/EVALS.md](docs/EVALS.md).

Adding a fifth integration (Todoist) took about ten minutes:
[docs/ADDING_AN_INTEGRATION.md](docs/ADDING_AN_INTEGRATION.md).

## Development

```bash
python -m uv sync
python -m uv run ruff check
python -m uv run pyright
python -m uv run pytest
```

New integrations come from the copier template in `templates/integration`
(see its README):

```bash
python -m uv run bench new <name>
```

## Test accounts

Live evals and recordings need real test accounts; [docs/SETUP.md](docs/SETUP.md) walks
through creating them (free plans only). Settings go into a gitignored `.env`; check them with

```bash
python -m uv run bench doctor
```

## Security

- **Least-privilege scopes:** each tool declares the scopes it needs, and a server drops
  tools its credentials do not cover, so the model never sees a tool that can only fail.
  The accounts ask for little: Google Drive only `drive.file` (files the app created),
  the GitHub App only Issues, Pull requests and Contents on one repository, Slack three read/post
  scopes.
- **Read-only mode:** `BENCH_READ_ONLY=1` removes every writing tool; `bench smoke` and the
  `live-smoke.yml` check always run read-only.
- **Tokens in the keyring:** `bench auth login <name>` stores OAuth tokens in the OS
  credential store, not in files. CI uses non-rotating credentials from repository secrets
  through a read-only store.
- **No secrets in the repo:** recorded cassettes are scrubbed of tokens and e-mail
  addresses, and gitleaks scans the full history on every CI run.
