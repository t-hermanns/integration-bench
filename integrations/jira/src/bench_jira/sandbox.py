"""`bench sandbox seed jira`: the fake's seed issues in the live project (API token).

Issue keys follow creation order, so the issues are created in list order into an empty
project; Jira never reuses numbers, so a project that once had issues needs replacing.
The live site has a single user: the fake's own account (`BOT`) maps to the API token's
user, who is the only assignee and writes every comment.
"""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx2

from bench_core.http import ApiClient
from bench_core.sandbox import Echo, SandboxError, require

from bench_jira.adf import text_to_adf
from bench_jira.auth import api_token
from bench_jira.client import (
    BASE_URL,
    JiraClient,
    make_api,
    project_key,
    raise_for_status,
    site_api_url,
)
from bench_jira.fake.backend import BOT
from bench_jira.fake.seed import ISSUES, SeedIssue
from bench_jira.models import IssueSummary

DEFAULT_PROJECT = "BENCH"


async def seed(
    env: Mapping[str, str],
    *,
    echo: Echo = print,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> None:
    require(env, "JIRA_SITE", "JIRA_EMAIL", "JIRA_API_TOKEN")
    key = project_key(env.get("JIRA_PROJECT") or DEFAULT_PROJECT)
    credential = api_token("jira-sandbox", email=env["JIRA_EMAIL"], token=env["JIRA_API_TOKEN"])
    async with make_api(
        credential=credential, base_url=site_api_url(env["JIRA_SITE"]), transport=transport
    ) as api:
        client = JiraClient(api)
        existing = await _issues(client, key)
        if existing:
            _check_seeded(existing, key)
            echo(f"jira: {key} already holds the {len(ISSUES)} seed issues; nothing to do")
            return
        me = await api.get("/myself")
        raise_for_status(me, action="read the API token's user", what="/myself")
        account_id = me.json()["accountId"]
        for n, spec in enumerate(ISSUES, 1):
            created = await api.post("/issue", json={"fields": _fields(key, spec, account_id)})
            raise_for_status(created, action="create a seed issue", what=f"Project {key}")
            issue = created.json()["key"]
            if issue != f"{key}-{n}":
                raise SandboxError(
                    f"created {issue} instead of {key}-{n}: the project had issues before "
                    "(Jira never reuses numbers); seed into a fresh project"
                )
            for text in spec.comments:
                await client.add_comment(issue, text)
            if spec.status != "To Do":
                await _move(client, issue, spec.status)
            echo(f"{issue} [{spec.status}] {spec.summary}")
    echo(f"jira: created {len(ISSUES)} issues in {key}")


def _fields(project: str, spec: SeedIssue, account_id: str) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "project": {"key": project},
        "summary": spec.summary,
        "issuetype": {"name": spec.issue_type},
        "priority": {"name": spec.priority},
    }
    if spec.labels:
        fields["labels"] = list(spec.labels)
    if spec.description:
        fields["description"] = text_to_adf(spec.description)
    if spec.assignee == BOT["accountId"]:
        fields["assignee"] = {"accountId": account_id}
    return fields


async def _move(client: JiraClient, issue: str, status: str) -> None:
    for transition in await client.transitions(issue):
        if transition.to_status == status:
            await client.transition_issue(issue, transition.id)
            return
    raise SandboxError(f"{issue}: no transition to status {status!r} in the workflow")


async def _issues(client: JiraClient, key: str) -> list[IssueSummary]:
    found: list[IssueSummary] = []
    cursor: str | None = None
    while True:
        page = await client.search_issues(
            f"project = {key} ORDER BY key ASC", cursor=cursor, page_size=100
        )
        found += page.items
        cursor = page.next_cursor
        if cursor is None:
            return found


def _check_seeded(existing: list[IssueSummary], key: str) -> None:
    expected = [(f"{key}-{n}", spec.summary) for n, spec in enumerate(ISSUES, 1)]
    actual = sorted(
        ((i.key, i.summary) for i in existing), key=lambda pair: int(pair[0].split("-")[1])
    )
    if actual[: len(expected)] != expected:
        raise SandboxError(
            f"{key} has {len(existing)} issue(s) that do not match the seed; "
            "seed into a fresh, empty project"
        )


@asynccontextmanager
async def live_api(env: Mapping[str, str]) -> AsyncIterator[ApiClient]:
    """`bench record jira`: the API token against the sandbox site the fake mimics."""
    require(env, "JIRA_SITE", "JIRA_EMAIL", "JIRA_API_TOKEN")
    base_url = site_api_url(env["JIRA_SITE"])
    if base_url != BASE_URL or project_key(env.get("JIRA_PROJECT") or DEFAULT_PROJECT) != "BENCH":
        raise SandboxError(f"cassettes replay against {BASE_URL}, project BENCH; record there")
    credential = api_token("jira-record", email=env["JIRA_EMAIL"], token=env["JIRA_API_TOKEN"])
    async with make_api(credential=credential, base_url=base_url) as api:
        yield api
