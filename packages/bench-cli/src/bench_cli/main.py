"""`bench` command line. Integrations are found by convention: `bench_<name>.scenarios`."""

import importlib
import json
import logging
import os
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
from typing import Annotated, NoReturn

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
from bench_evals import results_branch
from bench_evals.budget import BUDGETS, BudgetedModel, BudgetExhausted, CallBudget
from bench_evals.models.providers import PROVIDERS
from bench_evals.regression import (
    INFRA_LABEL,
    REGRESSION_LABEL,
    GitHubIssues,
    changed_files,
    check,
    classify,
    confirm,
    file_issue,
    fingerprint,
    infra_body,
    infra_problem,
    regression_body,
    regression_fingerprint,
)
from bench_evals.runner import live_servers, run_tasks
from bench_evals.scorer import Score, score_run
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


def pick_tasks(
    tasks: list[EvalTask], split: str, ids: list[str] | None, integration: str | None = None
) -> list[EvalTask]:
    if integration:
        tasks = [t for t in tasks if integration in t.integrations]
        if not tasks:
            raise typer.BadParameter(f"no tasks for integration {integration!r}")
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
    budget = None
    if provider in BUDGETS:
        tree = results_branch.checkout(repo_root())
        budget = CallBudget(tree / budget_file(provider), BUDGETS[provider])
        typer.echo(f"{provider} budget: {budget.used()}/{budget.stop_at} used this month")
        model = BudgetedModel(model, budget)
    names = sorted({i for t in tasks for i in t.integrations})
    try:
        async with live_servers(names, env, variant) as servers:
            for _ in range(k):
                run_id = uuid.uuid4().hex[:12]
                runs = [t.for_run(run_id) for t in tasks]
                await run_tasks(model, runs, servers, out, run_id=run_id)
    finally:
        if budget is not None:
            save_budget(provider, budget)


def budget_file(provider: str) -> str:
    return f"budget/{provider}.json"


def save_budget(provider: str, budget: CallBudget) -> None:
    tree = budget.path.parent.parent
    message = f"budget: {provider} {budget.used()}/{budget.stop_at} in {budget.month()}"
    results_branch.publish(tree, [budget_file(provider)], message)
    typer.echo(f"{message} (pushed to {results_branch.BRANCH})")


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
    integration: Annotated[
        str | None, typer.Option(help="Only tasks that use this integration")
    ] = None,
    out: Annotated[Path | None, typer.Option(help="JSONL log (default: .bench/results/)")] = None,
    env_file: EnvFile = None,
) -> None:
    """Run eval tasks live against the sandbox and score them (writes leave {run_id} objects)."""
    if provider not in PROVIDERS:
        raise typer.BadParameter(f"unknown provider {provider!r}")
    if split not in ("dev", "test", "all"):
        raise typer.BadParameter(f"unknown split {split!r}")
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    tasks = pick_tasks(all_tasks(), split, task, integration)
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
    except BudgetExhausted as exc:
        typer.echo(f"stopped: {exc}", err=True)
        if not out.exists():
            raise typer.Exit(1) from None
    typer.echo(f"solved: {show_rate(rate(outcomes_of(out, tasks, verbose=True)))}")


@app.command()
def budget(
    provider: Annotated[str, typer.Argument(help="Budgeted provider")] = "cohere",
    add: Annotated[int, typer.Option(help="Record calls made outside `bench eval`")] = 0,
) -> None:
    """Show (or correct) the monthly call budget kept in the bench-results branch."""
    if provider not in BUDGETS:
        raise typer.BadParameter(f"no budget for {provider!r} (only {', '.join(BUDGETS)})")
    tree = results_branch.checkout(repo_root())
    calls = CallBudget(tree / budget_file(provider), BUDGETS[provider])
    if add:
        calls.add(add)
        save_budget(provider, calls)
    typer.echo(
        f"{provider}: {calls.used()}/{calls.stop_at} used in {calls.month()}, {calls.left()} left"
    )


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


def results_file(kind: str, provider: str, variant: str, ext: str) -> str:
    return f"{kind}/{provider}-{variant}.{ext}"


def github_env() -> tuple[str, str] | None:
    """(server, owner/repo) inside GitHub Actions, else None."""
    server, repo = os.environ.get("GITHUB_SERVER_URL"), os.environ.get("GITHUB_REPOSITORY")
    return (server, repo) if server and repo else None


def run_url() -> str:
    gh, run = github_env(), os.environ.get("GITHUB_RUN_ID")
    return f"{gh[0]}/{gh[1]}/actions/runs/{run}" if gh and run else "(local run)"


def report(label: str, fingerprint_: str, title: str, body: str) -> None:
    """File the issue with GITHUB_TOKEN in GitHub Actions; elsewhere only print it."""
    gh, token = github_env(), os.environ.get("GITHUB_TOKEN")
    if gh is None or not token:
        typer.echo(f"not in GitHub Actions, would file [{label}] {title}:\n{body}")
        return
    filed = file_issue(GitHubIssues(gh[1], token), label, fingerprint_, title, body)
    typer.echo(f"{'opened' if filed.created else 'commented on'} issue #{filed.number}")


def infra_issue(provider: str, variant: str, problem: str, env: dict[str, str]) -> NoReturn:
    for value in env.values():
        if len(value) >= 8:  # never echo a secret into a public issue
            problem = problem.replace(value, "***")
    fp = fingerprint("infra", provider, variant)
    title = f"Eval infrastructure problem ({provider}, {variant})"
    report(INFRA_LABEL, fp, title, infra_body(fp, provider, problem[:2000], run_url()))
    raise typer.Exit(1)


async def failed_checks(env: dict[str, str], targets: list[str]) -> list[str]:
    checks = await doctor_checks.run(env, targets)
    return [f"{c.target}: {c.detail}" for c in checks if c.status == "fail"]


def run_or_report(
    provider: str, tasks: list[EvalTask], env: dict[str, str], variant: str, out: Path
) -> list[Score]:
    """Run the tasks once and score them; any crash becomes an eval-infra issue."""
    try:
        anyio.run(eval_run, provider, tasks, env, variant, 1, out)
    except Exception as exc:  # the nightly reports a crash, it does not raise it
        infra_issue(provider, variant, f"the run failed: {type(exc).__name__}: {exc}", env)
    return score_run(tasks, read_records(out))


@app.command()
def regress(
    provider: Annotated[str, typer.Argument(help="Model provider")] = "gemini",
    descriptions: Annotated[str, typer.Option(help="Tool descriptions variant")] = "v2",
    split: Annotated[str, typer.Option(help="dev, test or all")] = "all",
    log: Annotated[
        Path | None, typer.Option(help="Judge this JSONL log instead of running the tasks")
    ] = None,
    set_baseline: Annotated[bool, typer.Option(help="Make this run the new baseline")] = False,
    env_file: EnvFile = None,
) -> None:
    """Nightly check: run the tasks once, compare with the baseline, file an issue on a drop.

    Baseline and history live in the bench-results branch. A regression (rule in
    docs/EVALS.md) gets a confirmation round of the lost tasks before the issue is filed.
    """
    if provider not in PROVIDERS:
        raise typer.BadParameter(f"unknown provider {provider!r}")
    if split not in ("dev", "test", "all"):
        raise typer.BadParameter(f"unknown split {split!r}")
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    root = repo_root()
    tasks = pick_tasks(all_tasks(), split, None)
    by_id = {t.id: t for t in tasks}
    env = env_settings(env_file)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if log is None:
        names = sorted({i for t in tasks for i in t.integrations})
        failed = anyio.run(failed_checks, env, [*names, provider])
        if failed:
            infra_issue(provider, descriptions, "bench doctor failed: " + "; ".join(failed), env)
        log = root / ".bench" / "results" / f"regress-{provider}-{descriptions}-{stamp}.jsonl"
        scores = run_or_report(provider, tasks, env, descriptions, log)
    else:
        scores = score_run(tasks, read_records(log))
    problem = infra_problem(scores)
    if problem:
        infra_issue(provider, descriptions, problem, env)
    current = task_outcomes(scores)
    typer.echo(f"solved: {show_rate(rate(current))}")

    head = results_branch.git("rev-parse", "HEAD", cwd=root).strip()
    model = PROVIDERS[provider].model_id(env)
    entry = {"date": stamp, "commit": head, "model": model, "split": split, "outcomes": current}
    tree = results_branch.checkout(root)
    history = results_file("history", provider, descriptions, "jsonl")
    baseline_path = results_file("baselines", provider, descriptions, "json")
    (tree / history).parent.mkdir(parents=True, exist_ok=True)
    with (tree / history).open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    if set_baseline or not (tree / baseline_path).exists():
        (tree / baseline_path).parent.mkdir(parents=True, exist_ok=True)
        (tree / baseline_path).write_text(
            json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        results_branch.publish(
            tree, [history, baseline_path], f"baseline: {provider} {descriptions} at {head[:7]}"
        )
        typer.echo(f"new baseline {baseline_path} at {head[:7]}")
        return
    results_branch.publish(tree, [history], f"nightly: {provider} {descriptions} at {head[:7]}")

    baseline = json.loads((tree / baseline_path).read_text(encoding="utf-8"))
    verdict = check(baseline["outcomes"], current)
    typer.echo(
        f"baseline {baseline['commit'][:7]} ({baseline['date']}): lost {len(verdict.lost)}, "
        f"gained {len(verdict.gained)}, McNemar p={verdict.paired.p_value:.3f}"
    )
    if verdict.regressed:
        typer.echo(f"confirmation round: {', '.join(verdict.lost)}")
        again = log.with_name(f"{log.stem}-confirm.jsonl")
        lost = [by_id[t] for t in verdict.lost]
        rerun = task_outcomes(run_or_report(provider, lost, env, descriptions, again))
        verdict = confirm(baseline["outcomes"], current, rerun)
    if not verdict.regressed:
        typer.echo("no regression")
        return

    try:
        cause = classify(changed_files(root, baseline["commit"], head), baseline["model"], model)
    except results_branch.ResultsBranchError:
        cause = "unknown (baseline commit not in this clone)"
    gh = github_env()
    span = f"{baseline['commit'][:7]}...{head[:7]}"
    fp = regression_fingerprint(provider, descriptions, verdict.lost, by_id)
    integrations = sorted({i for t in verdict.lost for i in by_id[t].integrations})
    body = regression_body(
        verdict,
        fingerprint_=fp,
        provider=provider,
        variant=descriptions,
        cause=cause,
        compare_url=f"{gh[0]}/{gh[1]}/compare/{span}" if gh else span,
        scores=scores,
        records=read_records(log),
    )
    title = f"Eval regression: {provider} {descriptions} ({', '.join(integrations)})"
    report(REGRESSION_LABEL, fp, title, body)
    raise typer.Exit(1)


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
