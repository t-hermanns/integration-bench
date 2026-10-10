"""Which tools a server may expose, given the granted scopes and `BENCH_READ_ONLY`.

Every tool declares the scopes it needs (`Requirement`) and whether it writes. At startup
the server determines the granted scopes (OAuth token response, `X-OAuth-Scopes` header,
GitHub App installation permissions, ...) and `select_tools` drops what is not covered,
so the model never sees a tool that can only fail. `None` as granted scopes means "not
determinable" (e.g. Jira API tokens, GitHub App user tokens): all tools stay registered
and a missing permission surfaces at call time as `PermissionDenied`.

Provider-specific parsing of error responses into `PermissionDenied` /
`NotFoundOrNoAccess` lives in each integration's client.
"""

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from bench_core.auth.store import parse_scopes

Granted = frozenset[str] | None


# --- Scope expansion ----------------------------------------------------------------------

LEVELS = ("read", "write", "admin")


def no_expansion(granted: frozenset[str]) -> frozenset[str]:
    return granted


def expand_levels(granted: Iterable[str]) -> frozenset[str]:
    """`name:write` implies `name:read`, `name:admin` implies both.

    Only for GitHub App permissions: Slack's `channels:write` does *not* grant
    `channels:read`, so this must never be a global default.
    """
    out = set(granted)
    for scope in list(out):
        name, sep, level = scope.rpartition(":")
        if sep and level in LEVELS:
            out.update(f"{name}:{lower}" for lower in LEVELS[: LEVELS.index(level)])
    return frozenset(out)


def expand_implied(granted: Iterable[str], implies: Mapping[str, Iterable[str]]) -> frozenset[str]:
    """Transitive closure over a provider's "broader scope covers narrower one" table."""
    out = set(granted)
    todo = list(out)
    while todo:
        for implied in implies.get(todo.pop(), ()):
            if implied not in out:
                out.add(implied)
                todo.append(implied)
    return frozenset(out)


def scopes_from_header(value: str | None) -> Granted:
    """Parse `X-OAuth-Scopes`-style headers (GitHub classic, Slack). Missing header -> None."""
    return None if value is None else parse_scopes(value)


# --- Requirements -------------------------------------------------------------------------


@dataclass(frozen=True)
class Requirement:
    """Satisfied if *any* alternative is fully granted. No alternatives = no scopes needed."""

    alternatives: tuple[frozenset[str], ...] = ()

    def missing(self, granted: Granted) -> frozenset[str] | None:
        """None if satisfied (or granted is unknown), else the smallest missing set."""
        if granted is None or not self.alternatives:
            return None
        gaps = [alt - granted for alt in self.alternatives]
        if any(not gap for gap in gaps):
            return None
        return min(gaps, key=lambda gap: (len(gap), sorted(gap)))

    def describe(self) -> str:
        if not self.alternatives:
            return "no scopes"
        return " or ".join(" + ".join(sorted(alt)) for alt in self.alternatives)


def requires(*scopes: str) -> Requirement:
    """All of `scopes` are needed."""
    return Requirement((frozenset(scopes),) if scopes else ())


def any_of(*requirements: Requirement) -> Requirement:
    return Requirement(tuple(alt for r in requirements for alt in r.alternatives))


# --- Tool selection -----------------------------------------------------------------------


@dataclass(frozen=True)
class ToolAccess:
    """Permission-relevant facts about one tool."""

    name: str
    requirement: Requirement = field(default_factory=Requirement)
    write: bool = False
    destructive: bool = False
    idempotent: bool = False

    def annotations(self, title: str | None = None) -> ToolAnnotations:
        return ToolAnnotations(
            title=title,
            read_only_hint=not self.write,
            destructive_hint=self.destructive if self.write else None,
            idempotent_hint=self.idempotent if self.write else None,
            open_world_hint=True,
        )


@dataclass(frozen=True)
class Selection:
    enabled: tuple[ToolAccess, ...]
    disabled: Mapping[str, str]
    """Tool name -> human-readable reason (for `bench doctor` and startup logs)."""


def read_only_from_env(environ: Mapping[str, str] | None = None) -> bool:
    value = (os.environ if environ is None else environ).get("BENCH_READ_ONLY", "")
    return value.strip().lower() in ("1", "true", "yes", "on")


def select_tools(
    tools: Iterable[ToolAccess],
    granted: Granted,
    *,
    read_only: bool = False,
    expand: Callable[[frozenset[str]], frozenset[str]] = no_expansion,
) -> Selection:
    effective = None if granted is None else expand(granted)
    enabled: list[ToolAccess] = []
    disabled: dict[str, str] = {}
    for tool in tools:
        if read_only and tool.write:
            disabled[tool.name] = "write tool hidden by BENCH_READ_ONLY"
            continue
        gap = tool.requirement.missing(effective)
        if gap is not None:
            disabled[tool.name] = f"missing scope(s): {', '.join(sorted(gap))}"
            continue
        enabled.append(tool)
    return Selection(tuple(enabled), disabled)


# --- Errors the model sees ----------------------------------------------------------------


class PermissionDenied(ToolError):
    def __init__(
        self,
        provider: str,
        *,
        action: str,
        missing: Iterable[str] | None = None,
        hint: str | None = None,
    ) -> None:
        self.provider = provider
        self.action = action
        self.missing = frozenset(missing or ())
        need = f" Missing permission: {', '.join(sorted(self.missing))}." if self.missing else ""
        hint = hint or (
            "Do not retry; tell the user which permission is needed or choose another action."
        )
        super().__init__(f"Permission denied by {provider} for {action}.{need} {hint}")


class NotFoundOrNoAccess(ToolError):
    """Some APIs (GitHub, Drive with `drive.file`) answer 404 for resources they hide."""

    def __init__(self, provider: str, what: str) -> None:
        self.provider = provider
        self.what = what
        super().__init__(
            f"{what} was not found on {provider}, or the credentials have no access to it. "
            "Check the identifier; if it is correct, the user may need to grant access."
        )
