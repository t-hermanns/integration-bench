"""Pydantic models for the Google Drive API (v3), shaped for the tools.

Models validate Drive's raw JSON and ignore unknown fields, so new provider fields never
break parsing. Drive sends `size` as a string and leaves it out for folders.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

FOLDER = "application/vnd.google-apps.folder"
DOCUMENT = "application/vnd.google-apps.document"
GOOGLE_PREFIX = "application/vnd.google-apps."

EXPORT_TYPES = {
    DOCUMENT: "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}
"""Docs Editors files and the text format they are read in (`files.export`)."""

KIND = {
    FOLDER: "folder",
    DOCUMENT: "document",
    "application/vnd.google-apps.spreadsheet": "spreadsheet",
    "application/vnd.google-apps.presentation": "presentation",
}


class GdriveModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, frozen=True)


class File(GdriveModel):
    id: str
    name: str
    mime_type: str = Field(alias="mimeType")
    parents: list[str] = []
    modified_time: datetime | None = Field(default=None, alias="modifiedTime")
    created_time: datetime | None = Field(default=None, alias="createdTime")
    size: int | None = None
    description: str | None = None
    web_view_link: str | None = Field(default=None, alias="webViewLink")

    @property
    def kind(self) -> str:
        """What the model reads instead of a MIME type: folder, document, ... or the type."""
        return KIND.get(self.mime_type, self.mime_type)

    @property
    def is_folder(self) -> bool:
        return self.mime_type == FOLDER


class Content(GdriveModel):
    file: File
    text: str


class Page[T](BaseModel):
    """One page of results plus the opaque cursor for the next page."""

    items: list[T]
    next_cursor: str | None = None
