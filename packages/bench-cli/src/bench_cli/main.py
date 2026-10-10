"""`bench` command line. Integrations are found by convention: `bench_<name>.scenarios`."""

import importlib
import tempfile
from pathlib import Path
from typing import Annotated

import anyio
import typer

from bench_core.testing import CassetteSuite, stale_cassettes, synthesize

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Build, test, evaluate and monitor MCP integrations."""


def load_suite(name: str) -> CassetteSuite:
    try:
        module = importlib.import_module(f"bench_{name}.scenarios")
    except ModuleNotFoundError as exc:
        if exc.name not in (f"bench_{name}", f"bench_{name}.scenarios"):
            raise
        raise typer.BadParameter(f"no integration {name!r} with cassette scenarios") from None
    suite = getattr(module, "SUITE", None)
    if not isinstance(suite, CassetteSuite):
        raise typer.BadParameter(f"bench_{name}.scenarios has no SUITE")
    return suite


@app.command()
def synth(
    name: Annotated[str, typer.Argument(help="Integration, e.g. github")],
    only: Annotated[
        list[str] | None, typer.Option("--only", help="Scenario to (re)generate; repeatable")
    ] = None,
    force: Annotated[bool, typer.Option(help="Also overwrite recorded cassettes")] = False,
    check: Annotated[
        bool, typer.Option(help="Write nothing; exit 1 if synthetic cassettes are stale")
    ] = False,
) -> None:
    """Generate synthetic cassettes by running the scenarios against the in-memory fake."""
    suite = load_suite(name)
    if check:
        with tempfile.TemporaryDirectory() as scratch:
            stale = anyio.run(stale_cassettes, suite, Path(scratch))
        if stale:
            typer.echo(f"stale cassettes: {', '.join(stale)}; run `bench synth {name}`")
            raise typer.Exit(1)
        typer.echo(f"{name}: all synthetic cassettes are up to date")
        return
    try:
        result = anyio.run(lambda: synthesize(suite, only=only, force=force))
    except KeyError as exc:
        raise typer.BadParameter(str(exc.args[0])) from None
    typer.echo(f"{name}: wrote {len(result.written)} synthetic cassette(s) to {suite.cassette_dir}")
    if result.skipped_recorded:
        typer.echo(f"kept recorded: {', '.join(result.skipped_recorded)} (use --force)")
    if result.removed:
        typer.echo(f"removed obsolete: {', '.join(result.removed)}")
