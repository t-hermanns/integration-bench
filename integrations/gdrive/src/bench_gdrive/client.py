"""Typed client for the slice of the Google Drive API (v3) the tools need.

Every method validates its identifiers before building a URL (no path injection from
model-supplied arguments) and quotes search text before it goes into a Drive query.
Error responses become `ToolError`s the model can act on: 401 -> sign in again, 403
with a scope reason -> `PermissionDenied`, 404 -> `NotFoundOrNoAccess` (with
`drive.file` the app cannot see files it did not create), validation errors -> Drive's
message and the offending parameter. Rate limits (429, 403 with a rate-limit `reason`)
and 5xx are retried by the `ApiClient` and surface as `RateLimited` once the wait budget
is used up.

Live behaviour this relies on (probed 2026-10-10): `alt=media` on a Docs Editors file is
403 `fileNotDownloadable` and `export` of any other file is 403 `fileNotExportable`, so
`read_file` picks the endpoint from the MIME type; exports start with a BOM, use CRLF and
carry no charset.
"""

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

import httpx2
from mcp.server.mcpserver.exceptions import ToolError

from bench_core.auth import Credential, CredentialAuth
from bench_core.clock import Clock
from bench_core.http import (
    ApiClient,
    DriveRetryPolicy,
    fetch_page,
    page_token,
)
from bench_core.permissions import NotFoundOrNoAccess, PermissionDenied

from bench_gdrive.models import DOCUMENT, EXPORT_TYPES, FOLDER, GOOGLE_PREFIX, Content, File, Page
from bench_gdrive.scopes import DRIVE_FILE

PROVIDER = "Google Drive"
BASE_URL = "https://www.googleapis.com"
FILES = "/drive/v3/files"
UPLOAD = "/upload/drive/v3/files"
USER_AGENT = "integration-bench"
MAX_PAGE_SIZE = 100
PAGINATION = page_token()

FILE_FIELDS = "id,name,mimeType,parents,modifiedTime,createdTime,size,description,webViewLink"
LIST_FIELDS = f"nextPageToken,files({FILE_FIELDS})"

MAX_NAME = 200
MAX_UPLOAD = 1_000_000
"""Characters of text `create_document` accepts."""
MAX_DOWNLOAD = 5_000_000
"""Bytes `read_file` downloads at most; larger files are refused before the download."""

TEXT_TYPES = frozenset(
    {"application/json", "application/xml", "application/x-yaml", "application/javascript"}
)

SCOPE_REASONS = frozenset({"insufficientPermissions", "ACCESS_TOKEN_SCOPE_INSUFFICIENT"})


def make_api(
    *,
    credential: Credential | None = None,
    base_url: str = BASE_URL,
    transport: httpx2.AsyncBaseTransport | None = None,
    clock: Clock | None = None,
    default_wait_budget: float = 30.0,
) -> ApiClient:
    return ApiClient(
        base_url,
        policy=DriveRetryPolicy(name=PROVIDER),
        clock=clock,
        transport=transport,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        auth=CredentialAuth(credential) if credential else None,
        default_wait_budget=default_wait_budget,
    )


# --- Argument validation ------------------------------------------------------------------

_FILE_ID = re.compile(r"^[A-Za-z0-9_-]{10,100}$")


def file_id(value: str, *, allow_root: bool = False) -> str:
    if allow_root and value == "root":
        return value
    if not _FILE_ID.match(value):
        raise ToolError(
            f"Invalid file id {value!r}: use the id from a search or folder listing, "
            "e.g. '1pX6iyYHhy002eXLvA7olcAV3VaWehbRT'."
        )
    return value


def quote(text: str) -> str:
    """A Drive query string literal: backslash and single quote escaped."""
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _positive(name: str, value: int) -> int:
    if value < 1:
        raise ToolError(f"{name} must be at least 1, got {value}.")
    return value


def multipart(metadata: Mapping[str, Any], text: str) -> tuple[bytes, str]:
    """`multipart/related` body for an upload: JSON metadata, then the text content.

    The boundary derives from the content, so equal uploads produce equal bodies.
    """
    meta = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    boundary = "bench-" + hashlib.sha256((meta + text).encode()).hexdigest()[:24]
    body = (
        f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{meta}\r\n"
        f"--{boundary}\r\nContent-Type: text/plain; charset=UTF-8\r\n\r\n{text}\r\n"
        f"--{boundary}--\r\n"
    )
    return body.encode(), f"multipart/related; boundary={boundary}"


def decode_text(content: bytes) -> str:
    """Drive text without a charset: UTF-8 (BOM dropped), LF line ends."""
    return content.decode("utf-8-sig", errors="replace").replace("\r\n", "\n")


# --- Error translation --------------------------------------------------------------------


def _error(response: httpx2.Response) -> Mapping[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {}
    error = data.get("error") if isinstance(data, Mapping) else None
    return error if isinstance(error, Mapping) else {}


def _first_detail(response: httpx2.Response) -> Mapping[str, Any]:
    errors = _error(response).get("errors")
    if isinstance(errors, list) and errors and isinstance(errors[0], Mapping):
        return errors[0]
    return {}


def reasons(response: httpx2.Response) -> set[str]:
    """`error.errors[].reason` plus `error.details[].reason` (ErrorInfo)."""
    error = _error(response)
    found: set[str] = set()
    for key in ("errors", "details"):
        items = error.get(key)
        if isinstance(items, list):
            found.update(
                str(item["reason"])
                for item in items
                if isinstance(item, Mapping) and item.get("reason")
            )
    return found


def error_text(response: httpx2.Response) -> str:
    """`{"error": {"message": "Invalid Value", "errors": [{"location": "q"}]}}`."""
    message = _error(response).get("message")
    text = message if isinstance(message, str) and message else response.reason_phrase
    location = _first_detail(response).get("location")
    return f"{text} (parameter {location})" if location else text


def raise_for_status(response: httpx2.Response, *, action: str, what: str) -> None:
    """Raise a model-facing `ToolError` for any non-2xx response."""
    if response.is_success:
        return
    status = response.status_code
    if status == 401:
        raise ToolError(
            f"{PROVIDER} rejected the credentials while trying to {action} (401). "
            "Do not retry; the user has to sign in again."
        )
    if status == 403:
        found = reasons(response)
        if found & SCOPE_REASONS:
            raise PermissionDenied(PROVIDER, action=action, missing=[DRIVE_FILE])
        if "appNotAuthorizedToFile" in found:
            raise NotFoundOrNoAccess(PROVIDER, what)
        raise ToolError(f"{PROVIDER} refused to {action}: {error_text(response)}.")
    if status == 404:
        raise NotFoundOrNoAccess(PROVIDER, what)
    if status == 400:
        raise ToolError(
            f"{PROVIDER} rejected the request to {action}: {error_text(response)}. "
            "Fix the arguments and call again."
        )
    if status >= 500:
        raise ToolError(
            f"{PROVIDER} had a server error ({status}) while trying to {action}. Try again later."
        )
    raise ToolError(
        f"{PROVIDER} returned {status} while trying to {action}: {error_text(response)}."
    )


# --- Client -------------------------------------------------------------------------------


class GdriveClient:
    def __init__(self, api: ApiClient) -> None:
        self.api = api

    async def _list(
        self, q: str, *, cursor: str | None, page_size: int, order_by: str, action: str, what: str
    ) -> Page[File]:
        params: dict[str, Any] = {
            "q": q,
            "fields": LIST_FIELDS,
            "orderBy": order_by,
            "pageSize": min(_positive("page_size", page_size), MAX_PAGE_SIZE),
        }
        page = await fetch_page(self.api, FILES, PAGINATION, cursor=cursor, params=params)
        raise_for_status(page.response, action=action, what=what)
        data = page.response.json().get("files") or []
        return Page[File](
            items=[File.model_validate(f) for f in data], next_cursor=page.next_cursor
        )

    async def search_files(
        self,
        text: str,
        *,
        folder_id: str | None = None,
        cursor: str | None = None,
        page_size: int = 20,
    ) -> Page[File]:
        """Files whose name or content contains `text`, newest first."""
        if not text.strip():
            raise ToolError("The search text must not be empty; use list_folder to browse.")
        literal = quote(text.strip())
        q = f"(name contains {literal} or fullText contains {literal}) and trashed = false"
        if folder_id is not None:
            q += f" and {quote(file_id(folder_id, allow_root=True))} in parents"
        return await self._list(
            q,
            cursor=cursor,
            page_size=page_size,
            order_by="modifiedTime desc",
            action="search files",
            what="The search",
        )

    async def list_folder(
        self, folder_id: str = "root", *, cursor: str | None = None, page_size: int = 20
    ) -> Page[File]:
        """Direct children of a folder, folders first, then by name."""
        fid = file_id(folder_id, allow_root=True)
        return await self._list(
            f"{quote(fid)} in parents and trashed = false",
            cursor=cursor,
            page_size=page_size,
            order_by="folder,name",
            action="list the folder",
            what=f"Folder {folder_id}",
        )

    async def get_file(self, fid: str) -> File:
        response = await self.api.get(f"{FILES}/{file_id(fid)}", params={"fields": FILE_FIELDS})
        raise_for_status(response, action="read the file metadata", what=f"File {fid}")
        return File.model_validate(response.json())

    async def read_file(self, fid: str) -> Content:
        """Text of a file: Docs Editors files exported, text files downloaded."""
        file = await self.get_file(fid)
        if file.is_folder:
            raise ToolError(
                f"{file.name!r} is a folder; use list_folder with folder_id {file.id!r}."
            )
        export = EXPORT_TYPES.get(file.mime_type)
        if export is not None:
            response = await self.api.get(f"{FILES}/{file.id}/export", params={"mimeType": export})
        elif file.mime_type.startswith("text/") or file.mime_type in TEXT_TYPES:
            if file.size is not None and file.size > MAX_DOWNLOAD:
                raise ToolError(
                    f"{file.name!r} has {file.size} bytes; only files up to {MAX_DOWNLOAD} "
                    "bytes are read."
                )
            response = await self.api.get(f"{FILES}/{file.id}", params={"alt": "media"})
        else:
            kind = "Google file" if file.mime_type.startswith(GOOGLE_PREFIX) else "file"
            raise ToolError(
                f"{file.name!r} is a {kind} of type {file.mime_type} without readable text. "
                "Only Google Docs, Sheets, Slides and text files can be read."
            )
        raise_for_status(response, action="read the file content", what=f"File {fid}")
        return Content(file=file, text=decode_text(response.content))

    async def create_document(self, name: str, text: str, *, folder_id: str | None = None) -> File:
        """A Google Doc with `text` as its body (plain text, converted by Drive)."""
        name = name.strip()
        if not name:
            raise ToolError("The document name must not be empty.")
        if len(name) > MAX_NAME:
            raise ToolError(f"The document name has {len(name)} characters; at most {MAX_NAME}.")
        if len(text) > MAX_UPLOAD:
            raise ToolError(f"The text has {len(text)} characters; at most {MAX_UPLOAD}.")
        metadata: dict[str, Any] = {"name": name, "mimeType": DOCUMENT}
        if folder_id is not None:
            metadata["parents"] = [file_id(folder_id, allow_root=True)]
        body, content_type = multipart(metadata, text)
        response = await self.api.post(
            UPLOAD,
            params={"uploadType": "multipart", "fields": FILE_FIELDS},
            content=body,
            headers={"Content-Type": content_type},
        )
        what = f"Folder {folder_id}" if folder_id else "My Drive"
        raise_for_status(response, action="create the document", what=what)
        return File.model_validate(response.json())

    async def create_folder(self, name: str, *, parent_id: str | None = None) -> File:
        """Not a tool: the sandbox seed and scenario setup create folders."""
        metadata: dict[str, Any] = {"name": name, "mimeType": FOLDER}
        if parent_id is not None:
            metadata["parents"] = [file_id(parent_id, allow_root=True)]
        response = await self.api.post(FILES, params={"fields": FILE_FIELDS}, json=metadata)
        raise_for_status(response, action="create the folder", what="My Drive")
        return File.model_validate(response.json())

    async def delete_file(self, fid: str) -> None:
        """Not a tool: scenario teardown (folders take their files with them)."""
        response = await self.api.delete(f"{FILES}/{file_id(fid)}")
        raise_for_status(response, action="delete the file", what=f"File {fid}")
