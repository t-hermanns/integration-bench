"""Jira MCP server.

`create_server` wires the tools, the descriptions variant and the permission filter; the
lifespan owns one `ApiClient` per client session. Tests pass a fake `transport`; live
runs pass a credential plus the site (API token) or the gateway URL (OAuth). `main()`
serves over stdio for manual use with an API token: it reads `JIRA_SITE` (e.g.
bench-sandbox), `JIRA_EMAIL` and `JIRA_API_TOKEN`.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext
from importlib.resources import files

import httpx2

from bench_core.auth import Credential
from bench_core.clock import Clock
from bench_core.http import ApiClient
from bench_core.mcp import BenchServer, build_server, load_descriptions
from bench_core.permissions import Granted

from bench_jira.auth import api_token
from bench_jira.client import BASE_URL, PROVIDER, SITE_URL, JiraClient, make_api, site_api_url
from bench_jira.scopes import EXPAND
from bench_jira.tools import TOOLS, JiraState

NAME = "jira"


def create_server(
    *,
    credential: Credential | None = None,
    api: ApiClient | None = None,
    transport: httpx2.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
    base_url: str = BASE_URL,
    site_url: str = SITE_URL,
    granted: Granted = None,
    read_only: bool | None = None,
    variant: str | None = None,
) -> BenchServer[JiraState]:
    """`granted` are the credential's scopes (None = unknown, keep all tools).

    `site_url` is where browse links point (the API may go through the gateway).
    `api` is an open client the caller owns (live smoke test, evals); it replaces
    `credential`, `transport` and `base_url`. `read_only=None` reads `BENCH_READ_ONLY`,
    `variant=None` reads `BENCH_DESCRIPTIONS`.
    """
    descriptions = load_descriptions(files("bench_jira") / "descriptions", variant)

    @asynccontextmanager
    async def lifespan() -> AsyncIterator[JiraState]:
        # A caller-owned client stays open after the session.
        session = (
            nullcontext(api)
            if api is not None
            else make_api(
                credential=credential, base_url=base_url, transport=transport, clock=clock
            )
        )
        async with session as session_api:
            yield JiraState(JiraClient(session_api, site=site_url))

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
    names = ("JIRA_SITE", "JIRA_EMAIL", "JIRA_API_TOKEN")
    site, email, token = (os.environ.get(name) for name in names)
    if not site or not email or not token:
        raise SystemExit(f"Set {', '.join(names)} to run the {PROVIDER} MCP server over stdio.")
    create_server(
        credential=api_token(NAME, email=email, token=token),
        base_url=site_api_url(site),
        site_url=f"https://{site}.atlassian.net",
    ).run("stdio")
