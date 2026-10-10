"""In-memory Google Drive API (v3) for development and unit tests.

Serves the endpoints the client uses in Drive's raw JSON shape and error format, as
probed against the live API (2026-10-10): `files.list` with the two query forms the
client builds, `files.get` (metadata and `alt=media`), `files.export`, multipart upload,
folder creation and delete. Docs exports start with a BOM and use CRLF like the real
ones. The fake also serves the OAuth token endpoint; Google does not rotate refresh
tokens, so a refresh token works any number of times and responses omit a new one.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import httpx2

from bench_core.auth import StaticToken
from bench_core.clock import Clock
from bench_core.testing import FakeBackend, FakeClock, FakeRequest

from bench_gdrive.auth import TOKEN_URL
from bench_gdrive.client import FILES, UPLOAD
from bench_gdrive.models import DOCUMENT, EXPORT_TYPES, FOLDER
from bench_gdrive.scopes import DRIVE_FILE

TOKEN_PATH = urlsplit(TOKEN_URL).path
ROOT_ID = "0AFakeRootFolderUk9PVA"
DOC_SIZE = "1024"
"""What Drive reports as `size` for a Google Doc, whatever its length."""
PUBLIC = (
    "id",
    "name",
    "mimeType",
    "parents",
    "createdTime",
    "modifiedTime",
    "size",
    "description",
    "webViewLink",
)

_LITERAL = r"'(?:[^'\\]|\\.)*'"
SEARCH_QUERY = re.compile(
    rf"^\(name contains (?P<a>{_LITERAL}) or fullText contains (?P<b>{_LITERAL})\)"
    rf" and trashed = false(?: and (?P<parent>{_LITERAL}) in parents)?$"
)
FOLDER_QUERY = re.compile(rf"^(?P<parent>{_LITERAL}) in parents and trashed = false$")


def timestamp(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def credential(token: str) -> StaticToken:
    """The static credential that presents `token` (see `GdriveFake.issue_token`)."""
    return StaticToken("gdrive", token)


def unquote(literal: str) -> str:
    return re.sub(r"\\(.)", r"\1", literal[1:-1])


def multipart_parts(request: httpx2.Request) -> list[str]:
    """Bodies of a `multipart/related` request, each without its part headers."""
    match = re.search(r"boundary=([^;]+)", request.headers.get("Content-Type", ""))
    if match is None:
        return []
    chunks = request.content.decode().split(f"--{match[1]}")
    if len(chunks) < 3 or not chunks[-1].startswith("--"):
        return []
    parts = []
    for chunk in chunks[1:-1]:
        _, sep, body = chunk.partition("\r\n\r\n")
        if not sep:
            return []
        parts.append(body.removesuffix("\r\n"))
    return parts


def _oauth_error(status: int, error: str, description: str) -> httpx2.Response:
    return httpx2.Response(status, json={"error": error, "error_description": description})


def link(file_id: str, mime_type: str) -> str:
    if mime_type == FOLDER:
        return f"https://drive.google.com/drive/folders/{file_id}"
    if mime_type == DOCUMENT:
        return f"https://docs.google.com/document/d/{file_id}/edit?usp=drivesdk"
    return f"https://drive.google.com/file/d/{file_id}/view?usp=drivesdk"


@dataclass
class GdriveFake(FakeBackend):
    clock: Clock = field(default_factory=FakeClock)
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    """id -> raw file JSON plus the private keys `content` and `trashed`."""
    oauth_clients: dict[str, str] = field(default_factory=dict)
    """client_id -> client_secret of OAuth apps that may refresh user tokens."""
    refresh_tokens: dict[str, frozenset[str]] = field(default_factory=dict)
    """Valid refresh tokens -> scopes of the access tokens they mint (never consumed)."""
    _next_id: int = field(default=1000, init=False, repr=False)
    _minted: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        scopes = [DRIVE_FILE]
        self.route("POST", TOKEN_PATH, public=True)(self._token)
        self.route("GET", FILES, scopes=scopes)(self._list)
        self.route("POST", FILES, scopes=scopes)(self._create_folder)
        self.route("POST", UPLOAD, scopes=scopes)(self._upload)
        self.route("GET", FILES + "/{file_id}", scopes=scopes)(self._get)
        self.route("DELETE", FILES + "/{file_id}", scopes=scopes)(self._delete)
        self.route("GET", FILES + "/{file_id}/export", scopes=scopes)(self._export)

    # --- setup ----------------------------------------------------------------------------

    def _mint(self, prefix: str) -> str:
        self._minted += 1
        return f"fake-{prefix}-{self._minted}"

    def user_tokens(self, scopes: Sequence[str] = ()) -> tuple[str, str]:
        """A fresh access + refresh token pair, as after `bench auth login gdrive`."""
        access = self.issue_token(self._mint("access"), scopes)
        refresh = self._mint("refresh")
        self.refresh_tokens[refresh] = frozenset(scopes)
        return access, refresh

    def now(self) -> str:
        return timestamp(self.clock.time())

    def add_file(
        self,
        name: str,
        *,
        mime_type: str = DOCUMENT,
        parent: str = ROOT_ID,
        content: str = "",
        description: str | None = None,
        modified: str | None = None,
    ) -> dict[str, Any]:
        self._next_id += 1
        file_id = f"1Fake{self._next_id:06d}DriveFileIdXyZ"
        when = modified or self.now()
        file: dict[str, Any] = {
            "id": file_id,
            "name": name,
            "mimeType": mime_type,
            "parents": [parent],
            "createdTime": when,
            "modifiedTime": when,
            "webViewLink": link(file_id, mime_type),
            "content": content,
            "trashed": False,
        }
        if mime_type in EXPORT_TYPES:
            file["size"] = DOC_SIZE
        elif mime_type != FOLDER:
            file["size"] = str(len(content.encode()))
        if description is not None:
            file["description"] = description
        self.files[file_id] = file
        return file

    def add_folder(self, name: str, *, parent: str = ROOT_ID, modified: str | None = None) -> str:
        return self.add_file(name, mime_type=FOLDER, parent=parent, modified=modified)["id"]

    # --- provider error format ------------------------------------------------------------

    def _error(self, status: int, message: str, reason: str, **extra: Any) -> httpx2.Response:
        detail = {"message": message, "domain": "global", "reason": reason, **extra}
        error = {"code": status, "message": message, "errors": [detail]}
        return httpx2.Response(status, json={"error": error})

    def _invalid(self, location: str) -> httpx2.Response:
        return self._error(
            400, "Invalid Value", "invalid", location=location, locationType="parameter"
        )

    def _not_found(self, file_id: str) -> httpx2.Response:
        return self._error(
            404,
            f"File not found: {file_id}.",
            "notFound",
            location="fileId",
            locationType="parameter",
        )

    def unauthorized_response(self, request: httpx2.Request) -> httpx2.Response:
        response = self._error(
            401,
            "Request had invalid authentication credentials.",
            "authError",
            location="Authorization",
            locationType="header",
        )
        response.headers["WWW-Authenticate"] = 'Bearer realm="https://accounts.google.com/"'
        return response

    def forbidden_response(
        self, request: httpx2.Request, missing: Sequence[str]
    ) -> httpx2.Response:
        error = {
            "code": 403,
            "message": "Request had insufficient authentication scopes.",
            "errors": [
                {
                    "message": "Insufficient Permission",
                    "domain": "global",
                    "reason": "insufficientPermissions",
                }
            ],
            "status": "PERMISSION_DENIED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": "ACCESS_TOKEN_SCOPE_INSUFFICIENT",
                    "domain": "googleapis.com",
                }
            ],
        }
        return httpx2.Response(403, json={"error": error})

    def not_found_response(self, request: httpx2.Request) -> httpx2.Response:
        return self._error(404, "Not Found", "notFound")

    def server_error_response(self, request: httpx2.Request, status: int) -> httpx2.Response:
        return self._error(status, "Internal Error", "internalError")

    def rate_limit_response(
        self, request: httpx2.Request, retry_after: float | None
    ) -> httpx2.Response:
        """Drive's per-user limit: 403 with reason `userRateLimitExceeded`, no Retry-After."""
        return self._error(
            403, "User rate limit exceeded.", "userRateLimitExceeded", domain="usageLimits"
        )

    def _file_json(self, file: dict[str, Any]) -> dict[str, Any]:
        return {key: file[key] for key in PUBLIC if key in file}

    def _visible(self, file_id: str) -> dict[str, Any] | None:
        file = self.files.get(file_id)
        return None if file is None or file["trashed"] else file

    # --- token endpoint -------------------------------------------------------------------

    def _token(self, r: FakeRequest) -> httpx2.Response:
        """refresh_token grant; Google keeps the refresh token, so none comes back."""
        form = r.form()
        if self.oauth_clients.get(form.get("client_id", "")) != form.get("client_secret"):
            return _oauth_error(401, "invalid_client", "The OAuth client was not found.")
        if form.get("grant_type") != "refresh_token":
            return _oauth_error(400, "unsupported_grant_type", "Only refresh_token is faked.")
        scopes = self.refresh_tokens.get(form.get("refresh_token", ""))
        if scopes is None:
            return _oauth_error(400, "invalid_grant", "Token has been expired or revoked.")
        access = self.issue_token(self._mint("access"), sorted(scopes))
        return httpx2.Response(
            200,
            json={
                "access_token": access,
                "expires_in": 3599,
                "token_type": "Bearer",
                "scope": " ".join(sorted(scopes)),
            },
        )

    # --- routes ---------------------------------------------------------------------------

    def _list(self, r: FakeRequest) -> httpx2.Response:
        q = r.query.get("q", "")
        if search := SEARCH_QUERY.match(q):
            text = unquote(search["a"]).lower()
            hits = [
                f
                for f in self.files.values()
                if text in f["name"].lower() or text in f["content"].lower()
            ]
            hits.sort(key=lambda f: f["modifiedTime"], reverse=True)
            parent = search["parent"]
        elif listing := FOLDER_QUERY.match(q):
            hits = sorted(
                self.files.values(), key=lambda f: (f["mimeType"] != FOLDER, f["name"].lower())
            )
            parent = listing["parent"]
        else:
            return self._invalid("q")
        if parent is not None:
            parent_id = unquote(parent)
            parent_id = ROOT_ID if parent_id == "root" else parent_id
            hits = [f for f in hits if parent_id in f["parents"]]
        hits = [f for f in hits if not f["trashed"]]
        try:
            size = min(max(int(r.query.get("pageSize", "100")), 1), 1000)
        except ValueError:
            return self._invalid("pageSize")
        token = r.query.get("pageToken")
        if token is not None and not re.fullmatch(r"o\d+", token):
            return self._invalid("pageToken")
        offset = int(token[1:]) if token else 0
        body: dict[str, Any] = {"files": [self._file_json(f) for f in hits[offset : offset + size]]}
        if offset + size < len(hits):
            body["nextPageToken"] = f"o{offset + size}"
        body["incompleteSearch"] = False
        return httpx2.Response(200, json=body)

    def _get(self, r: FakeRequest) -> httpx2.Response:
        file_id = r.params["file_id"]
        file = self._visible(file_id)
        if file is None:
            return self._not_found(file_id)
        if r.query.get("alt") != "media":
            return httpx2.Response(200, json=self._file_json(file))
        if file["mimeType"].startswith("application/vnd.google-apps."):
            message = (
                "Only files with binary content can be downloaded. "
                "Use Export with Docs Editors files."
            )
            return self._error(
                403, message, "fileNotDownloadable", location="alt", locationType="parameter"
            )
        return httpx2.Response(
            200, content=file["content"].encode(), headers={"Content-Type": file["mimeType"]}
        )

    def _export(self, r: FakeRequest) -> httpx2.Response:
        file_id = r.params["file_id"]
        file = self._visible(file_id)
        if file is None:
            return self._not_found(file_id)
        export = EXPORT_TYPES.get(file["mimeType"])
        if export is None:
            return self._error(403, "Export only supports Docs Editors files.", "fileNotExportable")
        if r.query.get("mimeType") != export:
            return self._invalid("mimeType")
        text = "﻿" + file["content"].replace("\n", "\r\n")
        return httpx2.Response(200, content=text.encode(), headers={"Content-Type": export})

    def _parent_of(self, metadata: dict[str, Any]) -> str | httpx2.Response:
        parents = metadata.get("parents") or ["root"]
        parent = ROOT_ID if parents[0] == "root" else parents[0]
        folder = self._visible(parent) if parent != ROOT_ID else {"mimeType": FOLDER}
        if folder is None or folder["mimeType"] != FOLDER:
            return self._not_found(parent)
        return parent

    def _create_folder(self, r: FakeRequest) -> httpx2.Response:
        metadata = r.json() or {}
        if metadata.get("mimeType") != FOLDER or not metadata.get("name"):
            return self._invalid("mimeType")
        parent = self._parent_of(metadata)
        if isinstance(parent, httpx2.Response):
            return parent
        file = self.add_file(metadata["name"], mime_type=FOLDER, parent=parent)
        return httpx2.Response(200, json=self._file_json(file))

    def _upload(self, r: FakeRequest) -> httpx2.Response:
        if r.query.get("uploadType") != "multipart":
            return self._invalid("uploadType")
        parts = multipart_parts(r.request)
        if len(parts) != 2:
            return self._error(400, "Malformed multipart body.", "badContent")
        metadata = json.loads(parts[0])
        text = parts[1]
        if not metadata.get("name"):
            return self._invalid("name")
        parent = self._parent_of(metadata)
        if isinstance(parent, httpx2.Response):
            return parent
        mime_type = metadata.get("mimeType") or "text/plain"
        file = self.add_file(metadata["name"], mime_type=mime_type, parent=parent, content=text)
        return httpx2.Response(200, json=self._file_json(file))

    def _delete(self, r: FakeRequest) -> httpx2.Response:
        file_id = r.params["file_id"]
        if self._visible(file_id) is None:
            return self._not_found(file_id)
        doomed = [file_id]
        while doomed:
            current = doomed.pop()
            self.files.pop(current, None)
            doomed.extend(f["id"] for f in self.files.values() if current in f["parents"])
        return httpx2.Response(204)
