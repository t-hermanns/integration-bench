"""Pydantic models for the Todoist API, shaped for the tools.

Models validate the provider's raw JSON (`/api/v1/tasks`) and ignore unknown fields, so
new provider fields never break parsing. Todoist names a task's title `content` and has
no URL field, so `title` is an alias and `url` is derived from the id.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

WEB_URL = "https://app.todoist.com/app/task"


class TodoistModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, frozen=True)


class Task(TodoistModel):
    id: str
    title: str = Field(validation_alias="content")
    status: Literal["open", "closed"]
    description: str | None = None
    created_at: datetime = Field(validation_alias="added_at")
    updated_at: datetime
    url: str

    @model_validator(mode="before")
    @classmethod
    def _derive(cls, data: Any) -> Any:
        if isinstance(data, dict) and "checked" in data:
            return {
                **data,
                "status": "closed" if data["checked"] else "open",
                "description": data.get("description") or None,
                "url": f"{WEB_URL}/{data.get('id')}",
            }
        return data


class Page[T](BaseModel):
    """One page of results plus the opaque cursor for the next page."""

    items: list[T]
    next_cursor: str | None = None
