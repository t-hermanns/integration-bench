# integration-bench

MCP integrations for Jira Cloud, GitHub, Google Drive and Slack: scaffold, build, test, measure with evals, improve and monitor.

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
