"""`bench` command line. Integrations are found by convention: `bench_<name>.scenarios`."""

import importlib
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import webbrowser
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Annotated

import anyio
import typer
from mcp.server.mcpserver.exceptions import ToolError

from bench_core.auth import LoginError, LoginSpec, OAuthError, default_store, login
from bench_core.doctor import load_env, missing
from bench_core.sandbox import SandboxError
from bench_core.sandbox import smoke as smoke_live
from bench_core.testing import (
    CassetteSuite,
    RecordResult,
    SecretsFound,
    record_live,
    stale_cassettes,
    synthesize,
)

from bench_cli import doctor as doctor_checks
from bench_cli import scaffold
from bench_evals.models.providers import PROVIDERS
from bench_evals.runner import live_servers, run_tasks
from bench_evals.scorer import score_run
from bench_evals.stats import Rate, compare, rate, task_outcomes
from bench_evals.tasks import EvalTask, load_tasks, task_files

app = typer.Typer(no_args_is_help=True, add_completion=False)
auth_app = typer.Typer(no_args_is_help=True, help="Sign in to the integrations' APIs.")
app.add_typer(auth_app, name="auth")
sandbox_app = typer.Typer(no_args_is_help=True, help="Prepare the live test accounts.")
app.add_typer(sandbox_app, name="sandbox")


@app.callback()
def main() -> None:
    """Build, test, evaluate and monitor MCP integrations."""


def load_integration(name: str, submodule: str, what: str) -> ModuleType:
    """`bench_<name>.<submodule>`; `what` completes "no integration 'x'" if absent."""
    try:
        return importlib.import_module(f"bench_{name}.{submodule}")
    except ModuleNotFoundError as exc:
        if exc.name not in (f"bench_{name}", f"bench_{name}.{submodule}"):
            raise
        raise typer.BadParameter(f"no integration {name!r}{what}") from None


def load_suite(name: str) -> CassetteSuite:
    module = load_integration(name, "scenarios", " with cassette scenarios")
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


EnvFile = Annotated[
    Path | None, typer.Option(help="Settings file (default: .env in the repo root)")
]


def env_settings(env_file: Path | None) -> dict[str, str]:
    """Process environment over `.env` (default: the repo root's)."""
    if env_file is None:
        try:
            env_file = scaffold.find_root(Path.cwd()) / ".env"
        except scaffold.ScaffoldError:
            env_file = Path(".env")
    return load_env(env_file)


def load_sandbox(name: str) -> ModuleType:
    return load_integration(name, "sandbox", " with a sandbox module")


@sandbox_app.command("seed")
def sandbox_seed(
    name: Annotated[str, typer.Argument(help="Integration, e.g. jira")],
    env_file: EnvFile = None,
) -> None:
    """Create the fake's seed data in the live sandbox (once; reruns change nothing)."""
    module = load_sandbox(name)
    env = env_settings(env_file)
    try:
        anyio.run(lambda: module.seed(env, echo=typer.echo))
    except (SandboxError, ToolError, OAuthError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None


@app.command()
def record(
    name: Annotated[str, typer.Argument(help="Integration, e.g. github")],
    only: Annotated[
        list[str] | None, typer.Option("--only", help="Scenario to record; repeatable")
    ] = None,
    env_file: EnvFile = None,
) -> None:
    """Re-record cassettes by running the scenarios against the live sandbox."""
    suite = load_suite(name)
    sandbox = load_sandbox(name)
    env = env_settings(env_file)

    async def run() -> RecordResult:
        async with sandbox.live_api(env) as api:
            return await record_live(suite, api, only=only)

    try:
        result = anyio.run(run)
    except KeyError as exc:
        raise typer.BadParameter(str(exc.args[0])) from None
    except (SandboxError, SecretsFound, ToolError, OAuthError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"{name}: recorded {len(result.recorded)} cassette(s) in {suite.cassette_dir}")
    if result.synthetic_only:
        typer.echo(f"kept synthetic (need a shaped fake): {', '.join(result.synthetic_only)}")
    if shutil.which("gitleaks"):
        subprocess.run(["gitleaks", "dir", str(suite.cassette_dir)], check=False)
    else:
        typer.echo("gitleaks not found; only the built-in secret patterns were checked")
    typer.echo(f"next: `uv run pytest integrations/{name}`, review the diff, commit")


async def smoke_one(sandbox: ModuleType, server: ModuleType, env: dict[str, str]) -> str | None:
    async with sandbox.live_api(env) as api:
        live = server.create_server(api=api, read_only=True)
        return await smoke_live(live, sandbox.SMOKE, lambda line: typer.echo(f"  {line}"))


@app.command()
def smoke(
    names: Annotated[list[str], typer.Argument(help="Integrations, e.g. github slack")],
    env_file: EnvFile = None,
) -> None:
    """Call read-only tools through each MCP server against the live sandbox."""
    logging.getLogger("httpx2").setLevel(logging.WARNING)  # the server logs every request
    env = env_settings(env_file)
    failed: list[str] = []
    for name in names:
        sandbox = load_sandbox(name)
        server = load_integration(name, "server", "")
        typer.echo(f"{name}:")
        try:
            failure = anyio.run(smoke_one, sandbox, server, env)
        except KeyError as exc:
            raise typer.BadParameter(str(exc.args[0])) from None
        except (SandboxError, ToolError, OAuthError) as exc:
            failure = str(exc)
        if failure:
            typer.echo(f"  FAIL {failure}", err=True)
            failed.append(name)
    if failed:
        typer.echo(f"smoke failed: {', '.join(failed)}", err=True)
        raise typer.Exit(1)
    typer.echo("smoke ok")


def repo_root() -> Path:
    try:
        return scaffold.find_root(Path.cwd())
    except scaffold.ScaffoldError as exc:
        raise typer.BadParameter(str(exc)) from None


def all_tasks() -> list[EvalTask]:
    return load_tasks(task_files(repo_root()))


def pick_tasks(tasks: list[EvalTask], split: str, ids: list[str] | None) -> list[EvalTask]:
    if ids:
        unknown = sorted(set(ids) - {t.id for t in tasks})
        if unknown:
            raise typer.BadParameter(f"unknown task(s): {', '.join(unknown)}")
        return [t for t in tasks if t.id in ids]
    return [t for t in tasks if split == "all" or t.split == split]


async def eval_run(
    provider: str, tasks: list[EvalTask], env: dict[str, str], variant: str, k: int, out: Path
) -> None:
    model = PROVIDERS[provider].build(env)
    names = sorted({i for t in tasks for i in t.integrations})
    async with live_servers(names, env, variant) as servers:
        for _ in range(k):
            run_id = uuid.uuid4().hex[:12]
            await run_tasks(model, [t.for_run(run_id) for t in tasks], servers, out, run_id=run_id)


def read_records(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def show_rate(r: Rate) -> str:
    if r.value is None or r.low is None or r.high is None:
        return "n/a"
    return f"{r.successes}/{r.n} = {r.value:.0%} (95% CI {r.low:.0%}-{r.high:.0%})"


def outcomes_of(path: Path, tasks: list[EvalTask], verbose: bool) -> dict[str, bool | None]:
    scores = score_run(tasks, read_records(path))
    if verbose:
        for s in scores:
            mark = "ok  " if s.solved else ("FAIL" if s.valid else "INV ")
            typer.echo(f"  {mark} {s.task}  {'; '.join(s.reasons)}")
    return task_outcomes(scores)


@app.command("eval")
def eval_(
    provider: Annotated[str, typer.Argument(help="Model provider: " + ", ".join(PROVIDERS))],
    split: Annotated[str, typer.Option(help="dev, test or all")] = "dev",
    descriptions: Annotated[str, typer.Option(help="Tool descriptions variant")] = "v1",
    k: Annotated[int, typer.Option(min=1, help="Runs per task")] = 1,
    task: Annotated[list[str] | None, typer.Option(help="Only these task ids")] = None,
    out: Annotated[Path | None, typer.Option(help="JSONL log (default: .bench/results/)")] = None,
    env_file: EnvFile = None,
) -> None:
    """Run eval tasks live against the sandbox and score them (writes leave {run_id} objects)."""
    if provider not in PROVIDERS:
        raise typer.BadParameter(f"unknown provider {provider!r}")
    if split not in ("dev", "test", "all"):
        raise typer.BadParameter(f"unknown split {split!r}")
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    tasks = pick_tasks(all_tasks(), split, task)
    if out is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = (
            repo_root() / ".bench" / "results" / f"{provider}-{descriptions}-{split}-{stamp}.jsonl"
        )
    typer.echo(f"{len(tasks)} task(s) x {k} with {provider}, descriptions {descriptions} -> {out}")
    try:
        anyio.run(eval_run, provider, tasks, env_settings(env_file), descriptions, k, out)
    except KeyError as exc:
        raise typer.BadParameter(f"{exc.args[0]} is not set") from None
    typer.echo(f"solved: {show_rate(rate(outcomes_of(out, tasks, verbose=True)))}")


@app.command()
def score(
    logs: Annotated[list[Path], typer.Argument(help="One JSONL log, or two to compare (a b)")],
    verbose: Annotated[bool, typer.Option(help="One line per run with reasons")] = False,
) -> None:
    """Score eval logs: solved rate per log; with two logs, exact McNemar of b against a."""
    if len(logs) > 2:
        raise typer.BadParameter("give one or two logs")
    tasks = all_tasks()
    outcomes = []
    for log in logs:
        typer.echo(f"{log.name}:")
        outcomes.append(outcomes_of(log, tasks, verbose))
        typer.echo(f"  solved: {show_rate(rate(outcomes[-1]))}")
    if len(outcomes) == 2:
        a, b = outcomes
        for t in sorted(set(a) & set(b)):
            if a[t] is not None and b[t] is not None and a[t] != b[t]:
                typer.echo(f"  {'only b' if b[t] else 'only a'}: {t}")
        c = compare(a, b)
        typer.echo(
            f"paired n={c.n}: a {show_rate(c.a)}, b {show_rate(c.b)}, "
            f"only a {c.only_a}, only b {c.only_b}, McNemar p={c.p_value:.3f}"
            + (f", {c.unpaired} unpaired" if c.unpaired else "")
        )


def load_login(name: str) -> LoginSpec:
    spec = getattr(load_integration(name, "auth", ""), "LOGIN", None)
    if not isinstance(spec, LoginSpec):
        raise typer.BadParameter(f"bench_{name} has no browser sign-in (auth.LOGIN)")
    return spec


@auth_app.command("login")
def auth_login(
    name: Annotated[str, typer.Argument(help="Integration, e.g. jira")],
    env_file: EnvFile = None,
    browser: Annotated[
        bool, typer.Option(help="Open the browser (else only print the sign-in URL)")
    ] = True,
) -> None:
    """Sign in through the browser and keep the tokens in the token store."""
    spec = load_login(name)
    env = env_settings(env_file)
    needed = [spec.client_id_var, *([spec.client_secret_var] if spec.client_secret_var else [])]
    if absent := missing(env, *needed):
        typer.echo(f"error: set {', '.join(absent)} (see docs/SETUP.md)", err=True)
        raise typer.Exit(1)

    def show(url: str) -> None:
        typer.echo(f"Sign in here if no browser opens:\n{url}\nWaiting for the redirect ...")
        if browser:
            webbrowser.open(url)

    try:
        tokens = anyio.run(
            lambda: login(
                spec,
                client_id=env[spec.client_id_var],
                client_secret=env[spec.client_secret_var] if spec.client_secret_var else None,
                open_browser=show,
            )
        )
        default_store(env).save(spec.account, tokens)
    except (LoginError, OAuthError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None
    details = []
    if tokens.expires_at is not None:
        details.append(f"access token valid for {(tokens.expires_at - time.time()) / 60:.0f} min")
    details.append("refresh token stored" if tokens.refresh_token else "no refresh token")
    if tokens.scopes:
        details.append(f"scopes {', '.join(sorted(tokens.scopes))}")
    typer.echo(f"{name}: signed in; {'; '.join(details)}")
    typer.echo(f"next: `bench doctor {name}`")


@app.command()
def doctor(
    targets: Annotated[
        list[str] | None,
        typer.Argument(help="Integrations or model providers to check (default: all)"),
    ] = None,
    env_file: EnvFile = None,
) -> None:
    """Check that the configured accounts and model keys work (read-only calls)."""
    try:
        env = env_settings(env_file)
        results = anyio.run(doctor_checks.run, env, targets)
    except (KeyError, ValueError) as exc:
        raise typer.BadParameter(str(exc.args[0])) from None
    width = max((len(r.target) for r in results), default=0)
    for r in results:
        typer.echo(f"{r.status.upper():<5} {r.target:<{width}}  {r.detail}")
    failed = sum(r.status == "fail" for r in results)
    skipped = sum(r.status == "skip" for r in results)
    typer.echo(f"{len(results) - failed - skipped} ok, {failed} failed, {skipped} skipped")
    if failed:
        raise typer.Exit(1)
