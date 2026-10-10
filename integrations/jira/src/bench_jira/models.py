"""Pydantic models for the Jira Cloud REST API (v3), shaped for the tools.

Models validate Jira's raw JSON: issue fields are lifted out of `fields`, nested objects
(`status.name`, `assignee.displayName`, ...) are read via aliases, and ADF rich text
becomes plain text. Unknown fields are ignored, so new Jira fields never break parsing.
"""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import AliasPath, BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from bench_jira.adf import adf_to_text

StatusCategory = Literal["new", "indeterminate", "done"]
"""Jira's fixed buckets behind the workflow-specific status names (To Do / In Progress / Done)."""

Text = Annotated[str | None, BeforeValidator(adf_to_text)]


def _display_name(value: Any) -> Any:
    """`{"accountId": ..., "displayName": "Alex Doe"}` -> "Alex Doe"; unassigned is null."""
    if isinstance(value, dict):
        return value.get("displayName") or value.get("accountId")
    return value


Person = Annotated[str | None, BeforeValidator(_display_name)]


class JiraModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, frozen=True)


class IssueModel(JiraModel):
    """Issue JSON keeps everything but id/key under `fields`; lift it to the top level."""

    @model_validator(mode="before")
    @classmethod
    def _lift_fields(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("fields"), dict):
            return {**data["fields"], **{k: v for k, v in data.items() if k != "fields"}}
        return data


class Comment(JiraModel):
    id: str
    author: Person = None
    body: Text = None
    created: datetime
    updated: datetime


class IssueSummary(IssueModel):
    """One search hit: the fields `SEARCH_FIELDS` asks for."""

    id: str
    key: str
    summary: str
    status: str = Field(validation_alias=AliasPath("status", "name"))
    status_category: StatusCategory = Field(
        validation_alias=AliasPath("status", "statusCategory", "key")
    )
    issue_type: str = Field(validation_alias=AliasPath("issuetype", "name"))
    priority: str | None = Field(default=None, validation_alias=AliasPath("priority", "name"))
    assignee: Person = None
    labels: list[str] = []
    created: datetime
    updated: datetime


class Issue(IssueSummary):
    project: str = Field(validation_alias=AliasPath("project", "key"))
    reporter: Person = None
    description: Text = None
    comments: list[Comment] = Field(default=[], validation_alias=AliasPath("comment", "comments"))
    comment_total: int = Field(default=0, validation_alias=AliasPath("comment", "total"))


class CreatedIssue(JiraModel):
    id: str
    key: str


class Transition(JiraModel):
    id: str
    name: str
    to_status: str = Field(validation_alias=AliasPath("to", "name"))


class Page[T](BaseModel):
    """One page of results plus the opaque cursor for the next page."""

    items: list[T]
    next_cursor: str | None = None
