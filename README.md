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
