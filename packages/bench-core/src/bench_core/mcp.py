"""Assemble an integration's MCP server: YAML descriptions, permission filter, lifespan.

Tool functions carry the behaviour and the argument types; everything the model reads
(tool description, title, parameter descriptions, server instructions) comes from a
swappable `descriptions/<variant>.yaml`, selected by `BENCH_DESCRIPTIONS` (default `v1`).
That keeps the description experiment a pure data change. A file looks like:

    instructions: Optional server-level instructions.
    tools:
      github_get_issue:
        title: Get issue            # optional
        description: Get one issue by number.
        params:                     # optional, per argument
          number: Issue number, e.g. 42.

`build_server` checks the file against the code (every tool described, no unknown tools
or parameters), drops tools that `select_tools` rejects, and wires the lifespan that owns
the HTTP client. Tools read that state with `lifespan_state(ctx)`.
"""

import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from typing import Any, Literal

import httpx2
import yaml
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.tools import Tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bench_core.http import ApiClient
from bench_core.permissions import (
    Granted,
    Selection,
    ToolAccess,
    no_expansion,
    read_only_from_env,
    select_tools,
)

DEFAULT_VARIANT = "v1"
_VARIANT = re.compile(r"[a-z0-9][a-z0-9_-]*")

ResponseFormat = Literal["concise", "detailed"]
"""Shared `response_format` argument: concise for the model's next step, detailed for IDs."""


# --- Descriptions -------------------------------------------------------------------------


class ToolText(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str
    title: str | None = None
    params: dict[str, str] = {}


class Descriptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    variant: str = DEFAULT_VARIANT
    instructions: str | None = None
    tools: dict[str, ToolText]


def variant_from_env(environ: Mapping[str, str] | None = None) -> str:
    value = (os.environ if environ is None else environ).get("BENCH_DESCRIPTIONS", "").strip()
    variant = value or DEFAULT_VARIANT
    if not _VARIANT.fullmatch(variant):
        raise ValueError(f"BENCH_DESCRIPTIONS={variant!r} is not a valid variant name")
    return variant


def load_descriptions(directory: Traversable, variant: str | None = None) -> Descriptions:
    """Load `<directory>/<variant>.yaml`; `variant=None` reads `BENCH_DESCRIPTIONS`.

    `directory` is usually `importlib.resources.files("bench_<name>") / "descriptions"`.
    """
    variant = variant or variant_from_env()
    if not _VARIANT.fullmatch(variant):
        raise ValueError(f"{variant!r} is not a valid descriptions variant name")
    source = directory / f"{variant}.yaml"
    if not source.is_file():
        raise FileNotFoundError(f"No descriptions variant {variant!r} ({source})")
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{source}: expected a mapping at the top level")
    try:
        return Descriptions.model_validate({**data, "variant": variant})
    except ValidationError as exc:
        raise ValueError(f"{source}: {exc}") from exc


# --- Tools --------------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """One tool as an integration declares it: the function plus its permission facts.

    The MCP tool name is `access.name`, not the function name.
    """

    fn: Callable[..., Awaitable[Any]]
    access: ToolAccess

    @property
    def name(self) -> str:
        return self.access.name


def lifespan_state[StateT](ctx: Context[StateT]) -> StateT:
    """The object the server's lifespan yielded (e.g. a dataclass holding the `ApiClient`)."""
    return ctx.request_context.lifespan_context


@contextmanager
def tool_call(api: ApiClient, *, provider: str, wait_budget: float) -> Iterator[None]:
    """Scope of one tool call: one wait budget shared by all its requests (pagination,
    follow-up reads), and transport errors turned into a `ToolError` the model can act on.
    """
    with api.wait_budget(wait_budget):
        try:
            yield
        except httpx2.TransportError as exc:
            # Reads were already retried; writes are not, because the request may have
            # reached the provider before the connection broke.
            raise ToolError(
                f"Could not reach {provider} ({type(exc).__name__}). Try again later. If this "
                "call creates something, first check whether it was created."
            ) from exc


# --- Tool output helpers ------------------------------------------------------------------


def _is_none(value: Any) -> bool:
    return value is None


def omit() -> Any:
    """Default for format-dependent output fields: None, and left out of the output when None.

    Use it as `author: str | None = omit()` on the tool's output model, not inside a `type`
    alias (Pydantic ignores `Field(default=...)` there).
    """
    return Field(default=None, exclude_if=_is_none)


def clip(text: str | None, limit: int) -> tuple[str | None, bool | None]:
    """Text cut to `limit` characters, plus `True` if it was cut (else None, i.e. omitted)."""
    if text is None or len(text) <= limit:
        return text, None
    return text[:limit], True


def _short(value: Any, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def format_argument_errors(tool: str, exc: ValidationError) -> str:
    """Pydantic's raw text (model name, error codes, docs URL) condensed for the model."""
    problems = []
    for error in exc.errors(include_url=False):
        where = ".".join(str(part) for part in error["loc"]) or "arguments"
        problem = f"{where}: {error['msg']}"
        if error["type"] != "missing":
            problem += f" (got {_short(error['input'])})"
        problems.append(problem)
    return f"Invalid arguments for {tool}: {'; '.join(problems)}. Fix them and call again."


class _BenchTool(Tool):
    async def run(self, arguments: dict[str, Any], context: Any, convert_result: bool = False):
        # Validate up front so argument mistakes get a short message; `Tool.run` would
        # otherwise wrap pydantic's full text. Validation is cheap and side-effect free.
        try:
            self.fn_metadata.validate_arguments(arguments)
        except ValidationError as exc:
            raise ToolError(format_argument_errors(self.name, exc)) from exc
        return await super().run(arguments, context, convert_result)


def _make_tool(spec: ToolSpec, text: ToolText) -> Tool:
    tool = _BenchTool.from_function(
        spec.fn,
        name=spec.name,
        title=text.title,
        description=text.description.strip(),
        annotations=spec.access.annotations(title=text.title),
    )
    properties: dict[str, Any] = tool.parameters.get("properties", {})
    unknown = sorted(set(text.params) - set(properties))
    if unknown:
        raise ValueError(f"Descriptions for {spec.name} name unknown parameter(s): {unknown}")
    for param, description in text.params.items():
        properties[param]["description"] = description.strip()
    return tool


# --- Server -------------------------------------------------------------------------------


class BenchServer[StateT](MCPServer[StateT]):
    """An `MCPServer` that also remembers which tools were dropped and why."""

    selection: Selection


def build_server[StateT](
    name: str,
    tools: Sequence[ToolSpec],
    descriptions: Descriptions,
    lifespan: Callable[[], AbstractAsyncContextManager[StateT]],
    *,
    granted: Granted = None,
    read_only: bool | None = None,
    expand: Callable[[frozenset[str]], frozenset[str]] = no_expansion,
) -> BenchServer[StateT]:
    """Build the server for integration `name`; every tool name must start with `name_`.

    `granted` are the scopes the credentials carry (`None` = unknown, keep every tool).
    `read_only=None` reads `BENCH_READ_ONLY`. `lifespan` is entered once per client
    session and should yield the state tools need, typically holding an `ApiClient`.
    """
    specs = {spec.name: spec for spec in tools}
    if len(specs) != len(tools):
        raise ValueError(f"Duplicate tool names in {name}")
    if bad := sorted(n for n in specs if not n.startswith(f"{name}_")):
        raise ValueError(f"Tool names must start with {name + '_'!r}: {bad}")
    if missing := sorted(set(specs) - set(descriptions.tools)):
        raise ValueError(f"Descriptions {descriptions.variant!r} lack tool(s): {missing}")
    if unknown := sorted(set(descriptions.tools) - set(specs)):
        raise ValueError(f"Descriptions {descriptions.variant!r} name unknown tool(s): {unknown}")

    if read_only is None:
        read_only = read_only_from_env()
    selection = select_tools(
        (spec.access for spec in tools), granted, read_only=read_only, expand=expand
    )
    # Build every tool, not just the enabled ones, so a broken description cannot hide
    # behind a scope or read-only filter until another configuration runs.
    built = {n: _make_tool(spec, descriptions.tools[n]) for n, spec in specs.items()}
    mcp_tools = [built[access.name] for access in selection.enabled]

    @asynccontextmanager
    async def server_lifespan(_: MCPServer[StateT]) -> AsyncIterator[StateT]:
        async with lifespan() as state:
            yield state

    server = BenchServer[StateT](
        name,
        instructions=descriptions.instructions,
        tools=mcp_tools,
        lifespan=server_lifespan,
    )
    server.selection = selection
    return server
