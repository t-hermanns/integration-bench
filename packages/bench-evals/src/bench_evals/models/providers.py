"""The eval models: pinned ids, settings and adapters, in one place.

Provider line-ups change often (Groq swaps models, Gemini's free quota differs per
model), so each model id is pinned here, can be overridden per `.env`, and
`bench doctor` checks it still exists before a live run.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import httpx2

from bench_evals.models.base import Model
from bench_evals.models.http import CohereModel, OpenAICompatModel

Factory = Callable[[str, str, httpx2.AsyncBaseTransport | None], Model]
"""(api key, model id, transport) -> adapter."""


@dataclass(frozen=True)
class Provider:
    name: str
    key_var: str
    model_var: str
    default_model: str
    metadata_url: str
    """Model metadata endpoint for the doctor; `{model}` is filled in."""
    auth_header: str
    factory: Factory
    auth_prefix: str = "Bearer "

    def model_id(self, env: Mapping[str, str]) -> str:
        return env.get(self.model_var) or self.default_model

    def build(
        self, env: Mapping[str, str], transport: httpx2.AsyncBaseTransport | None = None
    ) -> Model:
        """The adapter for this provider; raises KeyError when the API key is not set."""
        return self.factory(env[self.key_var], self.model_id(env), transport)


GEMINI_OPENAI = "https://generativelanguage.googleapis.com/v1beta/openai"
GROQ_OPENAI = "https://api.groq.com/openai/v1"

PROVIDERS = {
    p.name: p
    for p in [
        Provider(
            "cohere",
            "COHERE_API_KEY",
            "COHERE_MODEL",
            "command-a-03-2025",
            "https://api.cohere.com/v1/models/{model}",
            "Authorization",
            lambda key, model, transport: CohereModel(key, model, transport=transport),
        ),
        Provider(
            "gemini",
            "GEMINI_API_KEY",
            "GEMINI_MODEL",
            "gemini-3.5-flash-lite",
            "https://generativelanguage.googleapis.com/v1beta/models/{model}",
            "x-goog-api-key",
            # Gemini answers about one request in eight with a spurious 403 PERMISSION_DENIED,
            # and its OpenAI layer rejects `seed`.
            lambda key, model, transport: OpenAICompatModel(
                "gemini",
                GEMINI_OPENAI,
                key,
                model,
                seed=None,
                retry_once_on={403},
                transport=transport,
            ),
            auth_prefix="",
        ),
        Provider(
            "groq",
            "GROQ_API_KEY",
            "GROQ_MODEL",
            "qwen/qwen3.8-27b",
            "https://api.groq.com/openai/v1/models/{model}",
            "Authorization",
            lambda key, model, transport: OpenAICompatModel(
                "groq", GROQ_OPENAI, key, model, transport=transport
            ),
        ),
    ]
}
