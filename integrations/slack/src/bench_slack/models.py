"""Pydantic models for the Slack Web API, shaped for the tools.

Models validate Slack's raw JSON (nested fields via `AliasPath`) and ignore unknown
fields, so new provider fields never break parsing. Message timestamps (`ts`) are
Slack's message ids: strings like "1791655876.394709", unique per channel.
"""

from datetime import UTC, datetime

from pydantic import AliasPath, BaseModel, ConfigDict, Field


class SlackModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, frozen=True)


class Channel(SlackModel):
    id: str
    name: str
    topic: str = Field("", validation_alias=AliasPath("topic", "value"))
    purpose: str = Field("", validation_alias=AliasPath("purpose", "value"))
    is_member: bool = False
    is_archived: bool = False
    num_members: int | None = None


class Message(SlackModel):
    ts: str
    text: str = ""
    user: str | None = None
    bot_id: str | None = None
    subtype: str | None = None
    thread_ts: str | None = None
    reply_count: int = 0
    latest_reply: str | None = None

    @property
    def posted_at(self) -> datetime:
        return datetime.fromtimestamp(float(self.ts), UTC)

    @property
    def is_thread_parent(self) -> bool:
        return self.thread_ts == self.ts


class Page[T](BaseModel):
    """One page of results plus the opaque cursor for the next page."""

    items: list[T]
    next_cursor: str | None = None
