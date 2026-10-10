"""`bench doctor`: integration accounts (`bench_<name>.doctor`) plus the eval model keys.

Model checks fetch the pinned model's metadata: one free, read-only call per provider
that proves the key works and the model still exists (provider line-ups change often).
They move next to the model adapters once those exist.
"""

import importlib
import pkgutil
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import ModuleType

import httpx2

from bench_core.doctor import Check, http_failure, missing, not_configured, ok, probe

NOT_INTEGRATIONS = {"bench_core", "bench_cli"}


@dataclass(frozen=True)
class ModelProvider:
    name: str
    key_var: str
    model_var: str
    default_model: str
    url: str
    """Model metadata endpoint; `{model}` is filled in."""
    auth_header: str
    auth_prefix: str = "Bearer "


MODEL_PROVIDERS = [
    ModelProvider(
        "cohere",
        "COHERE_API_KEY",
        "COHERE_MODEL",
        "command-a-03-2025",
        "https://api.cohere.com/v1/models/{model}",
        "Authorization",
    ),
    ModelProvider(
        "gemini",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "gemini-3.5-flash-lite",
        "https://generativelanguage.googleapis.com/v1beta/models/{model}",
        "x-goog-api-key",
        auth_prefix="",
    ),
    ModelProvider(
        "groq",
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "qwen/qwen3.8-27b",
        "https://api.groq.com/openai/v1/models/{model}",
        "Authorization",
    ),
]


async def check_model(
    provider: ModelProvider,
    env: Mapping[str, str],
    transport: httpx2.AsyncBaseTransport | None = None,
) -> Check:
    target = f"{provider.name} model"
    absent = missing(env, provider.key_var)
    if absent:
        return not_configured(target, absent)
    model = env.get(provider.model_var) or provider.default_model
    headers = {provider.auth_header: provider.auth_prefix + env[provider.key_var]}
    async with httpx2.AsyncClient(transport=transport, timeout=30) as client:
        response = await client.get(provider.url.format(model=model), headers=headers)
    if response.status_code != 200:
        return http_failure(target, response, model)
    return ok(target, f"key valid; {model} available")


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
    for provider in MODEL_PROVIDERS:

        async def model(env: Mapping[str, str], provider: ModelProvider = provider) -> list[Check]:
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
