"""`bench` command line. Integrations are found by convention: `bench_<name>.scenarios`."""

import importlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import anyio
import typer

from bench_core.testing import CassetteSuite, stale_cassettes, synthesize

from bench_cli import scaffold

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


def parse_data(pairs: list[str] | None) -> dict[str, str]:
    data: dict[str, str] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise typer.BadParameter(f"expected KEY=VALUE, got {pair!r}", param_hint="--data")
        data[key] = value
    return data


@app.command()
def new(
    name: Annotated[str, typer.Argument(help="Integration name, e.g. jira")],
    data: Annotated[
        list[str] | None,
        typer.Option("--data", "-d", help="Template answer KEY=VALUE; repeatable"),
    ] = None,
    defaults: Annotated[
        bool, typer.Option(help="Use the default for every question not given via --data")
    ] = False,
    sync: Annotated[
        bool, typer.Option(help="Run `uv sync` and `bench synth <name>` afterwards")
    ] = True,
    allow_dirty: Annotated[
        bool, typer.Option(help="Generate from uncommitted template changes (not updatable)")
    ] = False,
) -> None:
    """Generate integrations/<name> from the template and register it in the workspace."""
    answers = parse_data(data)
    try:
        root = scaffold.find_root(Path.cwd())
        dst = scaffold.generate(root, name, answers, defaults=defaults, allow_dirty=allow_dirty)
        typer.echo(f"generated {dst.relative_to(root).as_posix()}, added bench-{name} to the root")
        if sync:
            subprocess.run([*scaffold.uv_command(), "sync"], cwd=root, check=True)
            # A fresh interpreter, so the newly installed bench_<name> is importable.
            subprocess.run([sys.executable, "-m", "bench_cli", "synth", name], check=True)
    except scaffold.ScaffoldError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None
    except subprocess.CalledProcessError as exc:
        typer.echo(f"error: {' '.join(map(str, exc.cmd))} failed", err=True)
        raise typer.Exit(1) from None
    typer.echo(
        f'next: `uv run pytest integrations/{name}`, then replace the example "items" '
        "resource with the real API (client, models, fake, scenarios, tools)"
    )


@app.command()
def update(
    name: Annotated[str, typer.Argument(help="Generated integration, e.g. jira")],
    vcs_ref: Annotated[
        str | None, typer.Option(help="Template git ref to update to (default: HEAD)")
    ] = None,
    defaults: Annotated[bool, typer.Option(help="Use defaults for new questions")] = False,
) -> None:
    """Merge later template changes into integrations/<name> (conflicts are marked inline)."""
    try:
        root = scaffold.find_root(Path.cwd())
        scaffold.update(root, name, vcs_ref=vcs_ref, defaults=defaults)
    except scaffold.ScaffoldError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"updated integrations/{name}; review the diff, then `bench synth {name} --check`")
