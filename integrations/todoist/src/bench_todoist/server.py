"""Todoist MCP server.

`create_server` wires the tools, the descriptions variant and the permission filter; the
lifespan owns one `ApiClient` per client session. Tests pass a fake `transport`; live
runs pass a credential. `main()` serves over stdio for manual use; it reads
`TODOIST_TOKEN`.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext
from importlib.resources import files

import httpx2

from bench_core.auth import Credential, StaticToken
from bench_core.clock import Clock
from bench_core.http import ApiClient
from bench_core.mcp import BenchServer, build_server, load_descriptions
from bench_core.permissions import Granted

from bench_todoist.client import BASE_URL, PROVIDER, TodoistClient, make_api
from bench_todoist.scopes import EXPAND
from bench_todoist.tools import TOOLS, TodoistState

NAME = "todoist"


def create_server(
    *,
    credential: Credential | None = None,
    api: ApiClient | None = None,
    transport: httpx2.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
    base_url: str = BASE_URL,
    granted: Granted = None,
    read_only: bool | None = None,
    variant: str | None = None,
) -> BenchServer[TodoistState]:
    """`granted` are the credential's scopes (None = unknown, keep all tools).

    `api` is an open client the caller owns (live smoke test, evals); it replaces
    `credential`, `transport` and `base_url`. `read_only=None` reads `BENCH_READ_ONLY`,
    `variant=None` reads `BENCH_DESCRIPTIONS`.
    """
    descriptions = load_descriptions(files("bench_todoist") / "descriptions", variant)

    @asynccontextmanager
    async def lifespan() -> AsyncIterator[TodoistState]:
        # A caller-owned client stays open after the session.
        session = (
            nullcontext(api)
            if api is not None
            else make_api(
                credential=credential, base_url=base_url, transport=transport, clock=clock
            )
        )
        async with session as session_api:
            yield TodoistState(TodoistClient(session_api))

    return build_server(
        NAME,
        TOOLS,
        descriptions,
        lifespan,
        granted=granted,
        read_only=read_only,
        expand=EXPAND,
    )


def main() -> None:
    token_var = "TODOIST_TOKEN"
    token = os.environ.get(token_var)
    if not token:
        raise SystemExit(f"Set {token_var} to run the {PROVIDER} MCP server over stdio.")
    create_server(credential=StaticToken(NAME, token)).run("stdio")
