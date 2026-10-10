"""Which Google Drive scopes each tool needs.

Every tool works with `drive.file`: the app sees only files it created or that the user
opened with it, which is exactly the sandbox folder. The broad `drive` scope covers
`drive.file`, so `EXPAND` adds it when only `drive` was granted.
"""

from collections.abc import Iterable

from bench_core.permissions import ToolAccess, expand_implied, requires

DRIVE = "https://www.googleapis.com/auth/drive"
DRIVE_FILE = "https://www.googleapis.com/auth/drive.file"

IMPLIES = {DRIVE: (DRIVE_FILE,)}


def EXPAND(granted: Iterable[str]) -> frozenset[str]:
    return expand_implied(granted, IMPLIES)


SEARCH_FILES = ToolAccess("gdrive_search_files", requires(DRIVE_FILE))
GET_FILE_METADATA = ToolAccess("gdrive_get_file_metadata", requires(DRIVE_FILE))
READ_FILE_CONTENT = ToolAccess("gdrive_read_file_content", requires(DRIVE_FILE))
LIST_FOLDER = ToolAccess("gdrive_list_folder", requires(DRIVE_FILE))
CREATE_DOCUMENT = ToolAccess("gdrive_create_document", requires(DRIVE_FILE), write=True)
