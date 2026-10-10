"""`bench sandbox seed gdrive`: the fake's seed tree in the live account (user login).

With `drive.file` the app sees only what it created, so the sandbox is one folder
`bench-sandbox` in My Drive that this command creates, files in list order (search
results are newest first). A rerun finds the folder and only checks its contents. Both
commands use the login from `bench auth login gdrive`; Drive has no API key for files.
"""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

import httpx2

from bench_core.auth import default_store
from bench_core.http import ApiClient
from bench_core.sandbox import Echo, SandboxError, SmokeCall, require

from bench_gdrive.auth import LOGIN, user_token
from bench_gdrive.client import (
    FILE_FIELDS,
    UPLOAD,
    GdriveClient,
    make_api,
    multipart,
    raise_for_status,
)
from bench_gdrive.fake.seed import FILES, SANDBOX
from bench_gdrive.models import FOLDER, File

SETTINGS = ("GDRIVE_CLIENT_ID", "GDRIVE_CLIENT_SECRET")


def _api(env: Mapping[str, str], transport: httpx2.AsyncBaseTransport | None) -> ApiClient:
    require(env, *SETTINGS)
    store = default_store(env)
    if store.load(LOGIN.account) is None:
        raise SandboxError("not signed in; run `bench auth login gdrive` first")
    credential = user_token(
        LOGIN.account,
        store,
        client_id=env["GDRIVE_CLIENT_ID"],
        client_secret=env["GDRIVE_CLIENT_SECRET"],
        transport=transport,
    )
    return make_api(credential=credential, transport=transport)


async def find_sandbox(client: GdriveClient) -> File | None:
    page = await client.list_folder("root", page_size=100)
    matches = [f for f in page.items if f.name == SANDBOX and f.is_folder]
    if len(matches) > 1:
        raise SandboxError(f"My Drive has {len(matches)} folders named {SANDBOX}; keep one")
    return matches[0] if matches else None


async def tree(client: GdriveClient, folder: File, prefix: str = "") -> list[str]:
    """Paths below `folder` like the seed's `SeedFile.path`, folders before their files."""
    paths: list[str] = []
    page = await client.list_folder(folder.id, page_size=100)
    for file in page.items:
        path = prefix + file.name
        paths.append(path)
        if file.is_folder:
            paths += await tree(client, file, path + "/")
    return paths


async def seed(
    env: Mapping[str, str],
    *,
    echo: Echo = print,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> None:
    async with _api(env, transport) as api:
        client = GdriveClient(api)
        existing = await find_sandbox(client)
        expected = sorted(f.path for f in FILES)
        if existing is not None:
            actual = sorted(await tree(client, existing))
            if actual != expected:
                raise SandboxError(
                    f"{SANDBOX} holds {actual}, expected {expected}; delete the folder in "
                    "Drive and seed again"
                )
            echo(f"gdrive: {SANDBOX} already holds the {len(FILES)} seed files; nothing to do")
            return
        root = await client.create_folder(SANDBOX)
        folders = {"": root.id}
        for spec in FILES:
            parent_path, _, name = spec.path.rpartition("/")
            parent = folders[parent_path]
            if spec.mime_type == FOLDER:
                folders[spec.path] = (await client.create_folder(name, parent_id=parent)).id
            else:
                await _create_file(api, name, spec.mime_type, spec.content, parent)
            echo(f"{SANDBOX}/{spec.path}")
    echo(f"gdrive: created {SANDBOX} with {len(FILES)} entries")


async def _create_file(api: ApiClient, name: str, mime_type: str, text: str, parent: str) -> None:
    body, content_type = multipart({"name": name, "mimeType": mime_type, "parents": [parent]}, text)
    response = await api.post(
        UPLOAD,
        params={"uploadType": "multipart", "fields": FILE_FIELDS},
        content=body,
        headers={"Content-Type": content_type},
    )
    raise_for_status(response, action="create a seed file", what=SANDBOX)


@asynccontextmanager
async def live_api(env: Mapping[str, str]) -> AsyncIterator[ApiClient]:
    """`bench record gdrive`: the signed-in user's Drive, where the sandbox lives."""
    async with _api(env, None) as api:
        yield api


SMOKE = [
    SmokeCall("gdrive_search_files", {"query": "glossary"}),
    SmokeCall("gdrive_read_file_content", lambda found: {"file_id": found["files"][0]["id"]}),
    SmokeCall("gdrive_list_folder", {}),
]
