"""`bench sandbox`, `bench smoke`: shared pieces of the `bench_<name>.sandbox` modules.

`bench_<name>.sandbox.seed(env, echo=...)` creates the fake's seed data in the live test
account, so scenarios and their assertions hold for synthetic and recorded cassettes
alike. It is idempotent: an already seeded sandbox is left alone, anything else that is
not empty is refused (issue numbers would not match the fake).

`bench_<name>.sandbox.SMOKE` lists read-only tool calls for `bench smoke <name>`, which
runs them through the real MCP server (descriptions, permission filter, tools, client)
against `live_api(env)`. A call's `args` may be a function of the previous call's
structured result, so a smoke test can read what a search found.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mcp.client import Client
from mcp.server import MCPServer
from mcp.types import TextContent

from bench_core.doctor import missing

Echo = Callable[[str], None]
Args = Mapping[str, Any] | Callable[[dict[str, Any]], Mapping[str, Any]]


class SandboxError(Exception):
    """The sandbox is not in a state the command can work from; the message says why."""


def require(env: Mapping[str, str], *names: str) -> None:
    if absent := missing(env, *names):
        raise SandboxError(f"set {', '.join(absent)} (see `bench doctor`)")


@dataclass(frozen=True)
class SmokeCall:
    tool: str
    args: Args


async def smoke(server: MCPServer[Any], calls: Sequence[SmokeCall], echo: Echo) -> str | None:
    """Run `calls` through an in-memory MCP client; returns the failure, if any.

    A failed call ends the run, since later calls may build on its result.
    """
    async with Client(server) as client:
        listed = {tool.name for tool in (await client.list_tools()).tools}
        echo(f"{len(listed)} tools listed")
        previous: dict[str, Any] = {}
        for call in calls:
            if call.tool not in listed:
                return f"{call.tool}: not listed (read-only filter or descriptions?)"
            try:
                args = dict(call.args(previous) if callable(call.args) else call.args)
            except (KeyError, IndexError, StopIteration, TypeError) as exc:
                return f"{call.tool}: previous result lacks {exc!r}"
            result = await client.call_tool(call.tool, args)
            if result.is_error or result.structured_content is None:
                text = " ".join(c.text for c in result.content if isinstance(c, TextContent))
                return f"{call.tool}: {text or 'no structured result'}"
            previous = result.structured_content
            echo(f"ok  {call.tool}")
    return None
