"""`bench doctor`: integration accounts (`bench_<name>.doctor`) plus the eval models
(`bench_evals.doctor`)."""

import importlib
import pkgutil
from collections.abc import Awaitable, Callable, Mapping
from types import ModuleType

import httpx2

from bench_core.doctor import Check, probe

from bench_evals.doctor import check_model
from bench_evals.models import PROVIDERS, Provider

NOT_INTEGRATIONS = {"bench_core", "bench_cli", "bench_evals"}


def integration_modules() -> dict[str, ModuleType]:
    """Installed `bench_<name>` packages that ship a `doctor` module, by name."""
    found: dict[str, ModuleType] = {}
    for info in pkgutil.iter_modules():
        if not info.name.startswith("bench_") or info.name in NOT_INTEGRATIONS:
            continue
        try:
            found[info.name.removeprefix("bench_")] = importlib.import_module(f"{info.name}.doctor")
        except ModuleNotFoundError as exc:
            if exc.name != f"{info.name}.doctor":
                raise
    return dict(sorted(found.items()))


CheckRunner = Callable[[Mapping[str, str]], Awaitable[list[Check]]]


def runners(transport: httpx2.AsyncBaseTransport | None = None) -> dict[str, CheckRunner]:
    """Target name -> its checks: integrations first, then the model providers."""
    out: dict[str, CheckRunner] = {}
    for name, module in integration_modules().items():
        out[name] = lambda env, module=module: module.checks(env, transport=transport)
    for provider in PROVIDERS.values():

        async def model(env: Mapping[str, str], provider: Provider = provider) -> list[Check]:
            target = f"{provider.name} model"
            return [await probe(target, check_model(provider, env, transport))]

        out[provider.name] = model
    return out


async def run(
    env: Mapping[str, str],
    only: list[str] | None = None,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> list[Check]:
    available = runners(transport)
    unknown = sorted(set(only or ()) - set(available))
    if unknown:
        raise KeyError(f"unknown target(s) {', '.join(unknown)}; known: {', '.join(available)}")
    results: list[Check] = []
    for name, runner in available.items():
        if not only or name in only:
            results += await runner(env)
    return results
