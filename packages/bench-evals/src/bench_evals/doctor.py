"""Doctor checks for the eval models.

Each fetches the pinned model's metadata: one free, read-only call per provider that
proves the key works and the model still exists.
"""

from collections.abc import Mapping

import httpx2

from bench_core.doctor import Check, http_failure, missing, not_configured, ok

from bench_evals.models.providers import Provider


async def check_model(
    provider: Provider,
    env: Mapping[str, str],
    transport: httpx2.AsyncBaseTransport | None = None,
) -> Check:
    target = f"{provider.name} model"
    absent = missing(env, provider.key_var)
    if absent:
        return not_configured(target, absent)
    model = provider.model_id(env)
    headers = {provider.auth_header: provider.auth_prefix + env[provider.key_var]}
    async with httpx2.AsyncClient(transport=transport, timeout=30) as client:
        response = await client.get(provider.metadata_url.format(model=model), headers=headers)
    if response.status_code != 200:
        return http_failure(target, response, model)
    return ok(target, f"key valid; {model} available")
