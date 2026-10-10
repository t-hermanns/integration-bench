"""`bench doctor`: do the configured accounts work, before a live run spends its budget?

Each integration provides `bench_<name>.doctor.checks(env)`, which reads its settings
from `env` (process environment over the repo's `.env`) and makes one or two cheap,
read-only calls. A check is `ok`, `fail` (configured but broken: stop before a live run)
or `skip` (not configured, or only verifiable later, e.g. at `bench auth login`).
Details never contain secrets, only ids, names and granted scopes.
"""

import os
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx2
from mcp.server.mcpserver.exceptions import ToolError

from bench_core.auth import OAuthError

Status = Literal["ok", "fail", "skip"]


@dataclass(frozen=True)
class Check:
    target: str
    """What was checked, e.g. "github app" or "jira api token"."""
    status: Status
    detail: str


def ok(target: str, detail: str) -> Check:
    return Check(target, "ok", detail)


def fail(target: str, detail: str) -> Check:
    return Check(target, "fail", detail)


def skip(target: str, detail: str) -> Check:
    return Check(target, "skip", detail)


def missing(env: Mapping[str, str], *names: str) -> list[str]:
    return [name for name in names if not env.get(name)]


def not_configured(target: str, names: list[str]) -> Check:
    return skip(target, f"not configured: set {', '.join(names)}")


async def probe(target: str, check: Awaitable[Check]) -> Check:
    """Run one check; auth, rate-limit and network errors become a `fail` with the reason.

    `ToolError` covers `AuthError`, `RateLimited` and invalid names from the validators.
    """
    try:
        return await check
    except (ToolError, OAuthError) as exc:
        return fail(target, str(exc))
    except httpx2.HTTPError as exc:
        return fail(target, f"{type(exc).__name__}: {exc}")
    except ValueError as exc:  # malformed setting, e.g. a broken private key or site name
        return fail(target, f"invalid setting: {exc}")


def http_failure(target: str, response: httpx2.Response, what: str) -> Check:
    """A non-2xx answer as a `fail`, with the start of the provider's error text."""
    body = " ".join(response.text.split())[:200]
    return fail(target, f"{what}: HTTP {response.status_code} {body}".rstrip())


# --- Settings -----------------------------------------------------------------------------


def parse_env_file(text: str) -> dict[str, str]:
    """`KEY=value` lines; `#` comments, blank lines, `export ` and matching quotes allowed.

    Unquoted values lose a trailing ` # comment`; quoted values are taken literally.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key.isidentifier():
            raise ValueError(f"not a KEY=value line: {raw!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def load_env(path: Path | None, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The `.env` file (if any) overlaid by the process environment, as CI sets secrets."""
    values = parse_env_file(path.read_text(encoding="utf-8")) if path and path.is_file() else {}
    values.update({k: v for k, v in (os.environ if environ is None else environ).items() if v})
    return values
