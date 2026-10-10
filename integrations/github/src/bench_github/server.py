"""GitHub MCP server.

`create_server` wires the tools, the descriptions variant and the permission filter; the
lifespan owns one `ApiClient` per client session. Tests pass a fake `transport`; live
runs pass a credential. `main()` serves over stdio with `GITHUB_TOKEN` for manual use;
`bench` wires App credentials and granted permissions in later steps.
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

from bench_github.client import BASE_URL, GitHubClient, make_api
from bench_github.scopes import EXPAND
from bench_github.tools import TOOLS, GitHubState

NAME = "github"


def create_server(
    *,
    credential: Credential | None = None,
    transport: httpx2.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
    base_url: str = BASE_URL,
    granted: Granted = None,
    read_only: bool | None = None,
    variant: str | None = None,
) -> BenchServer[GitHubState]:
    """`granted` are App permissions like `issues:write` (None = unknown, keep all tools).

    `read_only=None` reads `BENCH_READ_ONLY`, `variant=None` reads `BENCH_DESCRIPTIONS`.
    """
    descriptions = load_descriptions(files("bench_github") / "descriptions", variant)

    @asynccontextmanager
    async def lifespan() -> AsyncIterator[GitHubState]:
        api = make_api(credential=credential, base_url=base_url, transport=transport, clock=clock)
        async with api:
            yield GitHubState(GitHubClient(api))

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
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("Set GITHUB_TOKEN to run the GitHub MCP server over stdio.")
    create_server(credential=StaticToken(NAME, token)).run("stdio")
