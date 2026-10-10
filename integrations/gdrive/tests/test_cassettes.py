"""Client against cassettes: synthetic now, recorded from the sandbox later.

Assertions only use facts the sandbox seed guarantees (names, kinds, texts, counts), never
ids or timestamps, so they hold for both kinds.
"""

from collections.abc import AsyncIterator

import pytest
from vcr.errors import CannotOverwriteExistingCassetteException

from bench_core.permissions import NotFoundOrNoAccess, PermissionDenied
from bench_core.testing import read_manifest, stale_cassettes

from bench_gdrive.client import GdriveClient
from bench_gdrive.fake import SANDBOX
from bench_gdrive.scenarios import MISSING_ID, SCENARIOS, SUITE, _child, _find, _sandbox

pytestmark = pytest.mark.anyio


@pytest.fixture
async def client() -> AsyncIterator[GdriveClient]:
    async with SUITE.replay_api() as api:
        yield GdriveClient(api)


def test_manifest_covers_every_scenario() -> None:
    manifest = read_manifest(SUITE.cassette_dir)
    assert set(manifest) == set(SCENARIOS), "run `bench synth gdrive`"
    assert set(manifest.values()) <= {"synthetic", "recorded"}
    for name in SCENARIOS:
        assert SUITE.path(name).exists(), name


async def test_synthetic_cassettes_are_up_to_date(tmp_path) -> None:
    assert await stale_cassettes(SUITE, tmp_path) == [], "run `bench synth gdrive`"


async def test_list_root(client: GdriveClient) -> None:
    with SUITE.replay("list_root"):
        page = await client.list_folder()
    assert [(f.name, f.kind) for f in page.items if f.name == SANDBOX] == [(SANDBOX, "folder")]


async def test_list_folder_pages(client: GdriveClient) -> None:
    with SUITE.replay("list_folder_pages") as cassette:
        sandbox = await _sandbox(client)
        first = await client.list_folder(sandbox.id, page_size=3)
        second = await client.list_folder(sandbox.id, page_size=3, cursor=first.next_cursor)
    assert cassette.all_played
    assert first.next_cursor is not None
    assert second.next_cursor is None
    names = [f.name for f in first.items + second.items]
    assert names[:2] == ["Meeting notes", "Projects"]  # folders first, then by name
    assert set(names[2:]) == {"glossary.txt", "Onboarding guide"}


async def test_search_files(client: GdriveClient) -> None:
    with SUITE.replay("search_files"):
        page = await client.search_files("roadmap")
    # "Roadmap 2026" by name, the onboarding guide by its text.
    assert {f.name for f in page.items} == {"Roadmap 2026", "Onboarding guide"}
    assert all(f.kind == "document" for f in page.items)


async def test_search_in_folder(client: GdriveClient) -> None:
    with SUITE.replay("search_in_folder"):
        notes = await _child(client, (await _sandbox(client)).id, "Meeting notes")
        page = await client.search_files("weekly", folder_id=notes.id)
    assert len(page.items) == 3
    assert all(f.name.startswith("Weekly sync") and f.parents == [notes.id] for f in page.items)


async def test_search_no_results(client: GdriveClient) -> None:
    with SUITE.replay("search_no_results"):
        page = await client.search_files("kubernetes")
    assert (page.items, page.next_cursor) == ([], None)


async def test_get_file_metadata(client: GdriveClient) -> None:
    with SUITE.replay("get_file_metadata"):
        file = await client.get_file((await _find(client, "checklist")).id)
    assert (file.name, file.kind) == ("Release checklist", "document")
    assert file.web_view_link
    assert file.created_time
    assert file.modified_time
    assert len(file.parents) == 1


async def test_get_file_not_found(client: GdriveClient) -> None:
    with SUITE.replay("get_file_not_found"), pytest.raises(NotFoundOrNoAccess):
        await client.get_file(MISSING_ID)


async def test_read_document(client: GdriveClient) -> None:
    with SUITE.replay("read_document"):
        content = await client.read_file((await _find(client, "checklist")).id)
    # BOM dropped, CRLF folded to LF.
    assert content.text.startswith("Release checklist\n1. Freeze the main branch\n")
    assert "\r" not in content.text
    assert "﻿" not in content.text


async def test_read_text_file(client: GdriveClient) -> None:
    with SUITE.replay("read_text_file"):
        content = await client.read_file((await _find(client, "glossary")).id)
    assert content.file.mime_type == "text/plain"
    assert content.text.splitlines()[0] == "SSO: single sign-on"


async def test_create_document(client: GdriveClient) -> None:
    with SUITE.replay("create_document"):
        sandbox = await _sandbox(client)
        file = await client.create_document(
            "Retro notes",
            "Went well: the importer.\nTo improve: export speed.",
            folder_id=sandbox.id,
        )
    assert (file.name, file.kind, file.parents) == ("Retro notes", "document", [sandbox.id])
    assert file.web_view_link


async def test_create_document_bad_folder(client: GdriveClient) -> None:
    with SUITE.replay("create_document_bad_folder"), pytest.raises(NotFoundOrNoAccess):
        await client.create_document("Orphan", "text", folder_id=MISSING_ID)


async def test_search_forbidden(client: GdriveClient) -> None:
    with SUITE.replay("search_forbidden"), pytest.raises(PermissionDenied, match=r"drive\.file"):
        await client.search_files("roadmap")


async def test_unrecorded_request_fails_instead_of_going_out(client: GdriveClient) -> None:
    with (
        SUITE.replay("search_no_results"),
        pytest.raises(CannotOverwriteExistingCassetteException),
    ):
        await client.search_files("helm")
