"""Pydantic models for the GitHub REST API, in the shape the tools hand to the model.

Each model validates GitHub's raw JSON (nested `user.login`, `labels[].name`, ...) via
validation aliases and serializes to a flat, compact structure. Unknown fields are
ignored, so new GitHub fields never break parsing.
"""

import re
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import AliasPath, BaseModel, BeforeValidator, ConfigDict, Field, model_validator

_REPO_URL = re.compile(r"/repos/(?P<full_name>[^/]+/[^/]+)$")


def _login(value: Any) -> Any:
    """`{"login": "octocat", ...}` -> "octocat"; deleted users come back as null."""
    if isinstance(value, dict):
        return value.get("login")
    return value


def _names(value: Any) -> Any:
    """`[{"name": "bug", ...}]` -> ["bug"]; GitHub sometimes sends plain strings."""
    if isinstance(value, list):
        return [item.get("name") if isinstance(item, dict) else item for item in value]
    return value


def _logins(value: Any) -> Any:
    return [_login(item) for item in value] if isinstance(value, list) else value


Login = Annotated[str | None, BeforeValidator(_login)]
Logins = Annotated[list[str], BeforeValidator(_logins)]
Names = Annotated[list[str], BeforeValidator(_names)]


class GitHubModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, frozen=True)


class IssueSummary(GitHubModel):
    """One hit of the issue search (issues and pull requests share this endpoint)."""

    repository: str = Field(validation_alias="repository_url")
    number: int
    title: str
    state: Literal["open", "closed"]
    is_pull_request: bool = Field(default=False, validation_alias="pull_request")
    author: Login = Field(default=None, validation_alias="user")
    labels: Names = []
    comments: int = 0
    created_at: datetime
    updated_at: datetime
    html_url: str

    @model_validator(mode="before")
    @classmethod
    def _flatten(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        # `repository_url` is "https://api.github.com/repos/{owner}/{repo}".
        url = data.get("repository_url")
        if isinstance(url, str) and (m := _REPO_URL.search(url)):
            data["repository_url"] = m["full_name"]
        # Pull requests carry a `pull_request` object; plain issues do not.
        if "pull_request" in data and not isinstance(data["pull_request"], bool):
            data["pull_request"] = data["pull_request"] is not None
        return data


class Issue(IssueSummary):
    body: str | None = None
    state_reason: str | None = None
    assignees: Logins = []
    milestone: str | None = Field(default=None, validation_alias=AliasPath("milestone", "title"))
    closed_at: datetime | None = None


class Comment(GitHubModel):
    id: int
    author: Login = Field(default=None, validation_alias="user")
    body: str
    created_at: datetime
    updated_at: datetime
    html_url: str


class PullRequestSummary(GitHubModel):
    number: int
    title: str
    state: Literal["open", "closed"]
    draft: bool = False
    author: Login = Field(default=None, validation_alias="user")
    head: str = Field(validation_alias=AliasPath("head", "ref"))
    base: str = Field(validation_alias=AliasPath("base", "ref"))
    labels: Names = []
    created_at: datetime
    updated_at: datetime
    merged_at: datetime | None = None
    html_url: str


EntryType = Literal["file", "dir", "symlink", "submodule"]


class DirectoryEntry(GitHubModel):
    name: str
    path: str
    type: EntryType
    size: int = 0


class FileContent(GitHubModel):
    """A file from the contents API.

    `text` is None when the file is binary or too large for the contents API (> 1 MB
    comes back without content); `note` then says why.
    """

    path: str
    sha: str
    size: int
    text: str | None = None
    note: str | None = None
    html_url: str | None = None


class DirectoryListing(GitHubModel):
    path: str
    entries: list[DirectoryEntry]


class Page[T](BaseModel):
    """One page of results plus the opaque cursor for the next page."""

    items: list[T]
    next_cursor: str | None = None
    total_count: int | None = None
    incomplete_results: bool = False
