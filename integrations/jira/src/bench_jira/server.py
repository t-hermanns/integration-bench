"""Jira MCP server.

`create_server` wires the tools, the descriptions variant and the permission filter; the
lifespan owns one `ApiClient` per client session. Tests pass a fake `transport`; live
runs pass a credential. `main()` serves over stdio for manual use; it reads
`JIRA_TOKEN`.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files

import httpx2

from bench_core.auth import Credential, StaticToken
from bench_core.clock import Clock
from bench_core.mcp import BenchServer, build_server, load_descriptions
from bench_core.permissions import Granted

from bench_jira.client import BASE_URL, PROVIDER, JiraClient, make_api
from bench_jira.scopes import EXPAND
from bench_jira.tools import TOOLS, JiraState

NAME = "jira"


def create_server(
    *,
    credential: Credential | None = None,
    transport: httpx2.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
    base_url: str = BASE_URL,
    granted: Granted = None,
    read_only: bool | None = None,
    variant: str | None = None,
) -> BenchServer[JiraState]:
    """`granted` are the credential's scopes (None = unknown, keep all tools).

    `read_only=None` reads `BENCH_READ_ONLY`, `variant=None` reads `BENCH_DESCRIPTIONS`.
    """
    descriptions = load_descriptions(files("bench_jira") / "descriptions", variant)

    @asynccontextmanager
    async def lifespan() -> AsyncIterator[JiraState]:
        api = make_api(credential=credential, base_url=base_url, transport=transport, clock=clock)
        async with api:
            yield JiraState(JiraClient(api))

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
    token_var = "JIRA_TOKEN"
    token = os.environ.get(token_var)
    if not token:
        raise SystemExit(f"Set {token_var} to run the {PROVIDER} MCP server over stdio.")
    create_server(credential=StaticToken(NAME, token)).run("stdio")
