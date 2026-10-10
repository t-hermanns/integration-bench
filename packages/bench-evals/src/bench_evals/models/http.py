"""Plain-HTTP adapters: Cohere's v2 chat API and OpenAI-compatible chat completions.

No SDKs: one POST per turn through httpx2, so tests run against `MockTransport`.
Sampling is fixed (temperature 0, a seed) to keep runs as repeatable as the providers allow.
"""

import json
from collections.abc import Collection, Sequence
from typing import Any

import httpx2

from bench_evals.models.base import (
    Assistant,
    Message,
    Reply,
    Tool,
    ToolCall,
    ToolChoice,
    ToolResult,
    Usage,
    User,
    error_from,
    limit_headers,
)

TIMEOUT = 120


def parse_arguments(text: str | None) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string; anything else is kept for the scorer to see."""
    try:
        value = json.loads(text or "{}")
    except ValueError:
        return {"_unparsed": text}
    return value if isinstance(value, dict) else {"_unparsed": text}


def function_tools(tools: Sequence[Tool]) -> list[dict[str, Any]]:
    """Both APIs take OpenAI-style function tools."""
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


def function_call(call: ToolCall) -> dict[str, Any]:
    return call.raw or {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
    }


def tool_calls_from(raw_calls: list[dict[str, Any]] | None) -> tuple[ToolCall, ...]:
    calls = []
    for i, raw in enumerate(raw_calls or ()):
        function = raw.get("function") or {}
        call_id = raw.get("id") or f"call_{i}"
        raw = {**raw, "id": call_id}
        calls.append(
            ToolCall(
                call_id, function.get("name", ""), parse_arguments(function.get("arguments")), raw
            )
        )
    return tuple(calls)


class HttpModel:
    """Shared request plumbing; subclasses build the body and parse the reply."""

    provider: str
    url: str

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        seed: int | None = 0,
        retry_once_on: Collection[int] = (),
        transport: httpx2.AsyncBaseTransport | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.seed = seed
        self.retry_once_on = frozenset(retry_once_on)
        self.transport = transport

    async def post(self, body: dict[str, Any]) -> httpx2.Response:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx2.AsyncClient(transport=self.transport, timeout=TIMEOUT) as client:
            response = await client.post(self.url, json=body, headers=headers)
            if response.status_code in self.retry_once_on:
                response = await client.post(self.url, json=body, headers=headers)
        if response.status_code != 200:
            raise error_from(self.provider, response)
        return response

    def sampling(self) -> dict[str, Any]:
        return {"temperature": 0} | ({"seed": self.seed} if self.seed is not None else {})


class CohereModel(HttpModel):
    """Cohere v2 chat (`/v2/chat`): tool plans, `REQUIRED`/`NONE` tool choice, no `auto`."""

    provider = "cohere"
    url = "https://api.cohere.com/v2/chat"

    async def complete(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[Tool] = (),
        tool_choice: ToolChoice = "auto",
    ) -> Reply:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *map(self.wire, messages)],
            **self.sampling(),
        }
        if tools:
            body["tools"] = function_tools(tools)
        if tool_choice != "auto":
            body["tool_choice"] = tool_choice.upper()
        response = await self.post(body)
        data = response.json()
        message = data.get("message") or {}
        text = "".join(
            c.get("text", "") for c in message.get("content") or () if c.get("type") == "text"
        )
        tokens = (data.get("usage") or {}).get("tokens") or {}
        return Reply(
            Assistant(
                text, tool_calls_from(message.get("tool_calls")), message.get("tool_plan") or ""
            ),
            self.model,
            Usage(int(tokens.get("input_tokens", 0)), int(tokens.get("output_tokens", 0))),
            data.get("finish_reason", ""),
            limit_headers(response),
        )

    @staticmethod
    def wire(message: Message) -> dict[str, Any]:
        match message:
            case User(text):
                return {"role": "user", "content": text}
            case Assistant(text, (), _):
                return {"role": "assistant", "content": text}
            case Assistant(_, calls, plan):
                out: dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": [function_call(c) for c in calls],
                }
                return out | ({"tool_plan": plan} if plan else {})
            case ToolResult(call_id, _, content, _):
                return {"role": "tool", "tool_call_id": call_id, "content": string_ids(content)}


def string_ids(content: str) -> str:
    """Cohere reads a JSON object in a tool result as a document and rejects it (HTTP 400)
    when its top-level `id` is not a string, e.g. a GitHub comment id."""
    try:
        data = json.loads(content)
    except ValueError:
        return content
    items = data if isinstance(data, list) else [data]
    changed = False
    for item in items:
        if isinstance(item, dict) and "id" in item and not isinstance(item["id"], str):
            item["id"] = json.dumps(item["id"])
            changed = True
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")) if changed else content


class OpenAICompatModel(HttpModel):
    """OpenAI-style `/chat/completions`, as served by Gemini and Groq."""

    def __init__(self, provider: str, base_url: str, api_key: str, model: str, **kwargs: Any):
        super().__init__(api_key, model, **kwargs)
        self.provider = provider
        self.url = base_url.rstrip("/") + "/chat/completions"

    async def complete(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[Tool] = (),
        tool_choice: ToolChoice = "auto",
    ) -> Reply:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *map(self.wire, messages)],
            **self.sampling(),
        }
        if tools:
            body["tools"] = function_tools(tools)
            body["tool_choice"] = tool_choice
        response = await self.post(body)
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return Reply(
            Assistant(message.get("content") or "", tool_calls_from(message.get("tool_calls"))),
            data.get("model") or self.model,
            Usage(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))),
            choice.get("finish_reason") or "",
            limit_headers(response),
        )

    @staticmethod
    def wire(message: Message) -> dict[str, Any]:
        match message:
            case User(text):
                return {"role": "user", "content": text}
            case Assistant(text, (), _):
                return {"role": "assistant", "content": text}
            case Assistant(text, calls, _):
                return {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [function_call(c) for c in calls],
                }
            case ToolResult(call_id, _, content, _):
                return {"role": "tool", "tool_call_id": call_id, "content": content}
