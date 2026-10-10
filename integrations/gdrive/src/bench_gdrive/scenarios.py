"""Cassette scenarios: `bench synth gdrive` now, `bench record gdrive` later.

Each scenario drives `GdriveClient` against the sandbox. Expected tool errors are caught,
so the cassette holds the error response the tests then replay. Scenarios with `prepare`
shape the fake and can only be synthesized; a `teardown` deletes what a live run created.

Drive assigns random ids, so every scenario starts from a name: it lists My Drive or
searches, then uses the ids from that response. Replays see the same responses, so the
ids match. Search terms are whole words: live Drive matches words and word prefixes,
the fake any substring.
"""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx2

from bench_core.http import ApiClient
from bench_core.testing import CassetteSuite, FakeBackend, FakeClock, Scenario, expect_tool_error

from bench_gdrive.client import GdriveClient, make_api
from bench_gdrive.fake import SANDBOX, GdriveFake, credential, seeded_fake
from bench_gdrive.models import File
from bench_gdrive.scopes import DRIVE_FILE

CASSETTE_DIR = Path(__file__).resolve().parents[2] / "tests" / "cassettes"
SYNTHETIC_TOKEN = "synthetic-token"
FULL_ACCESS = [DRIVE_FILE]


def _fake(clock: FakeClock) -> FakeBackend:
    fake = seeded_fake(clock)
    fake.issue_token(SYNTHETIC_TOKEN, FULL_ACCESS)
    return fake


def _api(transport: httpx2.AsyncBaseTransport | None, clock: FakeClock | None) -> ApiClient:
    # The Authorization header is redacted in cassettes, so any token replays.
    return make_api(credential=credential(SYNTHETIC_TOKEN), transport=transport, clock=clock)


Run = Callable[[GdriveClient], Awaitable[Any]]
Teardown = Callable[[ApiClient, Any], Awaitable[None]]


def _scenario(
    run: Run,
    description: str,
    prepare: Callable[[GdriveFake], None] | None = None,
    teardown: Teardown | None = None,
) -> Scenario:
    async def drive(api: ApiClient) -> Any:
        return await run(GdriveClient(api))

    def shape(fake: FakeBackend) -> None:
        assert prepare is not None
        assert isinstance(fake, GdriveFake)
        prepare(fake)

    return Scenario(drive, shape if prepare else None, description, teardown)


# --- Scenarios ----------------------------------------------------------------------------

MISSING_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"


async def _child(client: GdriveClient, folder_id: str, name: str) -> File:
    page = await client.list_folder(folder_id, page_size=50)
    for file in page.items:
        if file.name == name:
            return file
    raise AssertionError(f"{name!r} not found; run `bench sandbox seed gdrive`")


async def _sandbox(client: GdriveClient) -> File:
    return await _child(client, "root", SANDBOX)


async def _find(client: GdriveClient, text: str) -> File:
    page = await client.search_files(text, page_size=1)
    if not page.items:
        raise AssertionError(f"no file matches {text!r}; run `bench sandbox seed gdrive`")
    return page.items[0]


async def list_root(client: GdriveClient) -> None:
    await client.list_folder()


async def list_folder_pages(client: GdriveClient) -> None:
    """The sandbox folder holds four entries, so a page size of 3 gives two pages."""
    sandbox = await _sandbox(client)
    first = await client.list_folder(sandbox.id, page_size=3)
    await client.list_folder(sandbox.id, page_size=3, cursor=first.next_cursor)


async def search_files(client: GdriveClient) -> None:
    await client.search_files("roadmap")


async def search_in_folder(client: GdriveClient) -> None:
    notes = await _child(client, (await _sandbox(client)).id, "Meeting notes")
    await client.search_files("weekly", folder_id=notes.id)


async def search_no_results(client: GdriveClient) -> None:
    await client.search_files("kubernetes")


async def get_file_metadata(client: GdriveClient) -> None:
    await client.get_file((await _find(client, "checklist")).id)


async def get_file_not_found(client: GdriveClient) -> None:
    await expect_tool_error(client.get_file(MISSING_ID))


async def read_document(client: GdriveClient) -> None:
    await client.read_file((await _find(client, "checklist")).id)


async def read_text_file(client: GdriveClient) -> None:
    await client.read_file((await _find(client, "glossary")).id)


async def create_document(client: GdriveClient) -> File:
    sandbox = await _sandbox(client)
    return await client.create_document(
        "Retro notes", "Went well: the importer.\nTo improve: export speed.", folder_id=sandbox.id
    )


async def _delete_created(api: ApiClient, created: File) -> None:
    await GdriveClient(api).delete_file(created.id)


async def create_document_bad_folder(client: GdriveClient) -> None:
    await expect_tool_error(client.create_document("Orphan", "text", folder_id=MISSING_ID))


async def search_forbidden(client: GdriveClient) -> None:
    await expect_tool_error(client.search_files("roadmap"))


def _no_scope_token(fake: GdriveFake) -> None:
    fake.issue_token(SYNTHETIC_TOKEN, [])


SCENARIOS: dict[str, Scenario] = {
    "list_root": _scenario(list_root, "My Drive as the app sees it: the sandbox folder"),
    "list_folder_pages": _scenario(list_folder_pages, "Sandbox folder, two pages of 3"),
    "search_files": _scenario(search_files, "Name or text contains 'roadmap'"),
    "search_in_folder": _scenario(search_in_folder, "'weekly' inside Meeting notes"),
    "search_no_results": _scenario(search_no_results, "Search without hits"),
    "get_file_metadata": _scenario(get_file_metadata, "Metadata of the release checklist"),
    "get_file_not_found": _scenario(get_file_not_found, "Unknown file id"),
    "read_document": _scenario(read_document, "Google Doc exported as text"),
    "read_text_file": _scenario(read_text_file, "Plain text file downloaded"),
    "create_document": _scenario(
        create_document, "New Google Doc in the sandbox folder", teardown=_delete_created
    ),
    "create_document_bad_folder": _scenario(
        create_document_bad_folder, "Parent folder does not exist"
    ),
    "search_forbidden": _scenario(
        search_forbidden, "Token without drive.file", prepare=_no_scope_token
    ),
}

SUITE = CassetteSuite(
    integration="gdrive",
    cassette_dir=CASSETTE_DIR,
    scenarios=SCENARIOS,
    fake=_fake,
    api=_api,
)
