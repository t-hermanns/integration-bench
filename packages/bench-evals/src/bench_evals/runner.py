"""Eval runner: model <-> in-process MCP client <-> server (fake or live sandbox).

Tasks run one after another. Each task gets only the tools of its own integrations
(Groq's 8K tokens/min hardly fit one request with all 21 tools), and each provider
gets a client-side `Limiter`, so a run stays inside the free tiers instead of
hammering into 429s. Every finished task becomes one JSON line.
"""

import importlib
import json
import uuid
from collections import deque
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp.client import Client
from mcp.server import MCPServer
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as McpTool

from bench_core.clock import Clock, SystemClock

from bench_evals.models import (
    Assistant,
    Message,
    Model,
    ModelError,
    Reply,
    ToolCall,
    ToolResult,
    Usage,
    User,
)
from bench_evals.schema import provider_tools

SYSTEM = (
    "You are an assistant with access to {integrations} through tools. "
    "Use the tools to complete the user's request, then answer briefly. "
    "Do not ask follow-up questions; if something is missing, say so in your answer."
)
MAX_TURNS = 8
"""Model requests per task; a model still calling tools after that is cut off."""
MAX_RATE_RETRIES = 3
LOGGED_RESULT_CHARS = 2000


@dataclass(frozen=True)
class Task:
    """What the runner needs of a task; format and expectations: `bench_evals.tasks.EvalTask`."""

    id: str
    integrations: tuple[str, ...]
    prompt: str


@dataclass(frozen=True)
class Limit:
    rpm: int
    tpm: int | None = None


LIMITS = {
    # Trial chat endpoint: 20 calls/min (header x-trial-endpoint-call-limit).
    "cohere": Limit(rpm=20),
    # AI Studio free tier for 3.5 Flash Lite: 15 RPM, 250K TPM (never the bottleneck).
    "gemini": Limit(rpm=15),
    # Free tier: 8K tokens/min is the bottleneck; RPM kept well below the header's.
    "groq": Limit(rpm=30, tpm=8000),
}


class Limiter:
    """Sliding 60 s window over requests and their tokens.

    `wait(tokens)` blocks until a request of about `tokens` fits and reserves it;
    `settle(tokens)` replaces that estimate with the reply's real usage. A single
    request above the token limit waits for an empty window instead of forever.
    """

    WINDOW = 60.0

    def __init__(
        self,
        limit: Limit,
        clock: Clock | None = None,
    ):
        self.limit = limit
        self.clock = clock or SystemClock()
        self.window: deque[list[float]] = deque()  # [start, tokens]

    def delay(self, tokens: int) -> float:
        now = self.clock.time()
        while self.window and self.window[0][0] <= now - self.WINDOW:
            self.window.popleft()
        delay = 0.0
        if len(self.window) >= self.limit.rpm:
            delay = self.window[-self.limit.rpm][0] + self.WINDOW - now
        if self.limit.tpm is not None:
            excess = sum(t for _, t in self.window) + tokens - self.limit.tpm
            for start, used in self.window:
                if excess <= 0:
                    break
                excess -= used
                delay = max(delay, start + self.WINDOW - now)
        return delay

    async def wait(self, tokens: int) -> float:
        waited = 0.0
        while (delay := self.delay(tokens)) > 0:
            await self.clock.sleep(delay)
            waited += delay
        self.window.append([self.clock.time(), tokens])
        return waited

    def settle(self, tokens: int) -> None:
        if self.window:
            self.window[-1][1] = tokens


def limiter_for(provider: str, clock: Clock | None = None) -> Limiter | None:
    limit = LIMITS.get(provider)
    return Limiter(limit, clock) if limit else None


class Toolbox:
    """Open MCP sessions of all servers in a run; routes tool calls by name."""

    def __init__(self, tools: Mapping[str, list[McpTool]], clients: Mapping[str, Client]):
        self.tools = tools
        self.owner = {t.name: name for name, ts in tools.items() for t in ts}
        self.clients = clients

    def for_task(self, task: Task) -> list[McpTool]:
        unknown = [i for i in task.integrations if i not in self.tools]
        if unknown:
            raise KeyError(f"task {task.id}: no server for {', '.join(unknown)}")
        return [t for i in task.integrations for t in self.tools[i]]

    async def call(self, call: ToolCall, allowed: set[str]) -> ToolResult:
        if call.name not in allowed:
            return ToolResult(call.id, call.name, f"Unknown tool: {call.name}", is_error=True)
        client = self.clients[self.owner[call.name]]
        try:
            result = await client.call_tool(call.name, call.arguments)
        except MCPError as exc:
            return ToolResult(call.id, call.name, str(exc), is_error=True)
        return ToolResult(call.id, call.name, result_text(result), is_error=result.is_error)


def result_text(result: CallToolResult) -> str:
    """What the model sees: compact JSON of the structured result, else the text parts.

    Our tools return structured content; the text block FastMCP adds holds the same
    data, indented, so the compact form saves tokens without hiding anything.
    """
    if result.structured_content is not None and not result.is_error:
        return json.dumps(result.structured_content, ensure_ascii=False, separators=(",", ":"))
    return "\n".join(c.text for c in result.content if isinstance(c, TextContent))


@asynccontextmanager
async def open_toolbox(servers: Mapping[str, MCPServer[Any]]) -> AsyncIterator[Toolbox]:
    async with AsyncExitStack() as stack:
        clients, tools = {}, {}
        for name, server in servers.items():
            clients[name] = await stack.enter_async_context(Client(server))
            tools[name] = (await clients[name].list_tools()).tools
        yield Toolbox(tools, clients)


@asynccontextmanager
async def live_servers(
    names: Sequence[str], env: Mapping[str, str]
) -> AsyncIterator[dict[str, MCPServer[Any]]]:
    """Read-write servers against the live sandbox, like `bench smoke` but for writes too."""
    async with AsyncExitStack() as stack:
        servers = {}
        for name in names:
            sandbox = importlib.import_module(f"bench_{name}.sandbox")
            server = importlib.import_module(f"bench_{name}.server")
            api = await stack.enter_async_context(sandbox.live_api(env))
            servers[name] = server.create_server(api=api, read_only=False)
        yield servers


async def complete(
    model: Model,
    limiter: Limiter | None,
    estimate: int,
    *args: Any,
    clock: Clock,
) -> tuple[Reply, float]:
    """One model request: waits for the limiter, retries 429s after `retry_after`."""
    waited = 0.0
    for attempt in range(MAX_RATE_RETRIES + 1):
        if limiter:
            waited += await limiter.wait(estimate)
        try:
            reply = await model.complete(*args)
        except ModelError as exc:
            if exc.status != 429 or attempt == MAX_RATE_RETRIES:
                raise
            pause = exc.retry_after if exc.retry_after is not None else Limiter.WINDOW
            await clock.sleep(pause)
            waited += pause
            continue
        if limiter:
            limiter.settle(reply.usage.input_tokens + reply.usage.output_tokens)
        return reply, waited
    raise AssertionError("unreachable")


async def run_task(
    model: Model,
    task: Task,
    toolbox: Toolbox,
    limiter: Limiter | None = None,
    max_turns: int = MAX_TURNS,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Runs one task to the model's final answer; the returned record is JSON-ready.

    `outcome` is "answered", "max_turns" (still calling tools) or "model_error"
    (the provider failed; not the model's fault, so the scorer can tell them apart).
    """
    clock = clock or SystemClock()
    mcp_tools = toolbox.for_task(task)
    tools = provider_tools(mcp_tools, model.provider)
    allowed = {t.name for t in tools}
    system = SYSTEM.format(integrations=", ".join(task.integrations))
    messages: list[Message] = [User(task.prompt)]
    record: dict[str, Any] = {
        "task": task.id,
        "integrations": list(task.integrations),
        "provider": model.provider,
        "model": model.model,
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "outcome": "max_turns",
        "answer": "",
        "error": None,
        "turns": [],
    }
    usage = Usage()
    waited = 0.0
    estimate = 0
    began = clock.time()
    for _ in range(max_turns):
        try:
            reply, pause = await complete(
                model, limiter, estimate, system, messages, tools, "auto", clock=clock
            )
        except ModelError as exc:
            record["outcome"], record["error"] = "model_error", str(exc)
            break
        waited += pause
        message = reply.message
        record["model"] = reply.model or model.model
        usage = Usage(
            usage.input_tokens + reply.usage.input_tokens,
            usage.output_tokens + reply.usage.output_tokens,
        )
        # The next request carries this one's context plus the new turn.
        estimate = reply.usage.input_tokens + reply.usage.output_tokens
        turn = log_turn(reply)
        record["turns"].append(turn)
        messages.append(message)
        if not message.tool_calls:
            record["outcome"], record["answer"] = "answered", message.text
            break
        for call in message.tool_calls:
            result = await toolbox.call(call, allowed)
            messages.append(result)
            turn["results"].append(
                {
                    "name": result.name,
                    "is_error": result.is_error,
                    "content": result.content[:LOGGED_RESULT_CHARS],
                }
            )
    record["tool_calls"] = sum(len(t["tool_calls"]) for t in record["turns"])
    record["usage"] = {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}
    record["waited_s"] = round(waited, 1)
    record["seconds"] = round(clock.time() - began, 1)
    return record


def log_turn(reply: Reply) -> dict[str, Any]:
    message: Assistant = reply.message
    return {
        "text": message.text,
        "plan": message.plan,
        "tool_calls": [
            {"id": c.id, "name": c.name, "arguments": c.arguments} for c in message.tool_calls
        ],
        "finish_reason": reply.finish_reason,
        "usage": {
            "input_tokens": reply.usage.input_tokens,
            "output_tokens": reply.usage.output_tokens,
        },
        "results": [],
    }


async def run_tasks(
    model: Model,
    tasks: Sequence[Task],
    servers: Mapping[str, MCPServer[Any]],
    out: Path,
    *,
    run_id: str | None = None,
    limiter: Limiter | None = None,
    max_turns: int = MAX_TURNS,
    clock: Clock | None = None,
) -> list[dict[str, Any]]:
    """Runs `tasks` one after another and appends one JSON line per task to `out`.

    Lines are flushed as tasks finish, so an aborted run keeps what it has.
    """
    run_id = run_id or uuid.uuid4().hex[:12]
    if limiter is None:
        limiter = limiter_for(model.provider, clock)
    records = []
    out.parent.mkdir(parents=True, exist_ok=True)
    async with open_toolbox(servers) as toolbox:
        for task in tasks:
            record = await run_task(model, task, toolbox, limiter, max_turns, clock)
            record = {"run_id": run_id, **record}
            records.append(record)
            with out.open("ab") as fh:
                fh.write(json.dumps(record, ensure_ascii=False).encode() + b"\n")
    return records
