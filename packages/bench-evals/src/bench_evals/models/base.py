"""The provider-neutral chat interface every model adapter implements.

Only what the evals need: text and tool calls, one complete reply per request (no
streaming). A conversation is a list of `Message`s; the adapter translates it into the
provider's wire format and the reply back into a `Reply`.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx2

ToolChoice = Literal["auto", "required", "none"]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    """JSON Schema of the arguments, already in a form the provider accepts (step 34)."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw: dict[str, Any] | None = None
    """The provider's own tool call object, sent back verbatim in later turns
    (Gemini attaches thought signatures it requires again)."""


@dataclass(frozen=True)
class User:
    text: str


@dataclass(frozen=True)
class Assistant:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    plan: str = ""
    """Cohere's `tool_plan`: the model's reasoning before its tool calls."""


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    content: str
    is_error: bool = False


Message = User | Assistant | ToolResult


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class Reply:
    message: Assistant
    model: str
    """The model id the provider reports, for the run log."""
    usage: Usage = Usage()
    finish_reason: str = ""
    limits: dict[str, str] = field(default_factory=dict)
    """Rate-limit and quota headers of the response (feeds the call budget, step 41)."""


class Model(Protocol):
    provider: str
    model: str

    async def complete(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[Tool] = (),
        tool_choice: ToolChoice = "auto",
    ) -> Reply: ...


class ModelError(Exception):
    """The provider rejected a request; `retry_after` is set for rate limits."""

    def __init__(self, provider: str, status: int, detail: str, retry_after: float | None = None):
        super().__init__(f"{provider}: HTTP {status}: {detail}")
        self.provider = provider
        self.status = status
        self.detail = detail
        self.retry_after = retry_after


LIMIT_HEADER_PREFIXES = ("x-ratelimit-", "x-trial-", "x-endpoint-", "retry-after")


def limit_headers(response: httpx2.Response) -> dict[str, str]:
    return {
        k: v for k, v in response.headers.items() if k.lower().startswith(LIMIT_HEADER_PREFIXES)
    }


def error_from(provider: str, response: httpx2.Response) -> ModelError:
    try:
        body = response.json()
    except ValueError:
        detail = response.text[:300]
    else:
        if isinstance(body, list) and body:  # Gemini wraps its error in a list
            body = body[0]
        error = body.get("error", body) if isinstance(body, dict) else body
        detail = error.get("message", str(error)) if isinstance(error, dict) else str(error)
    try:
        retry_after = float(response.headers["retry-after"])
    except (KeyError, ValueError):
        retry_after = None
    return ModelError(provider, response.status_code, str(detail)[:300], retry_after)
