"""Cassette suites: synthesis from a fake, manifest handling, staleness check and replay."""

from pathlib import Path

import httpx2
import pytest
import yaml
from mcp.server.mcpserver.exceptions import ToolError
from vcr.errors import CannotOverwriteExistingCassetteException

from bench_core.http import ApiClient, RetryPolicy
from bench_core.testing import (
    CassetteSuite,
    FakeBackend,
    FakeClock,
    FakeRequest,
    Scenario,
    expect_tool_error,
    read_manifest,
    stale_cassettes,
    synthesize,
    write_manifest,
)

pytestmark = pytest.mark.anyio

BASE = "https://api.example.test"


def make_fake(clock: FakeClock) -> FakeBackend:
    fake = FakeBackend(require_auth=False)
    fake.route("GET", "/files/{path:path}")(lambda r: {"path": r.params["path"]})
    return fake


def make_api(transport: httpx2.AsyncBaseTransport | None, clock: FakeClock | None) -> ApiClient:
    return ApiClient(BASE, policy=RetryPolicy(), transport=transport, clock=clock)


async def read_file(api: ApiClient) -> None:
    await api.get("/files/a/b.txt")


async def read_missing(api: ApiClient) -> None:
    await api.get("/nothing")


def suite(tmp_path: Path, **scenarios: Scenario) -> CassetteSuite:
    return CassetteSuite(
        integration="example",
        cassette_dir=tmp_path / "cassettes",
        scenarios=scenarios or {"read_file": Scenario(read_file)},
        fake=make_fake,
        api=make_api,
    )


async def test_synthesize_writes_cassettes_and_manifest(tmp_path: Path) -> None:
    s = suite(tmp_path, read_file=Scenario(read_file), read_missing=Scenario(read_missing))
    result = await synthesize(s)
    assert result.written == ("read_file", "read_missing")
    assert read_manifest(s.cassette_dir) == {"read_file": "synthetic", "read_missing": "synthetic"}
    cassette = yaml.safe_load(s.path("read_file").read_text(encoding="utf-8"))
    (interaction,) = cassette["interactions"]
    assert interaction["request"]["uri"] == f"{BASE}/files/a/b.txt"
    assert "accept-encoding" not in interaction["request"]["headers"]
    assert b"\r\n" not in s.path("read_file").read_bytes()
    assert (s.cassette_dir / "MANIFEST.yaml").read_text(encoding="utf-8").startswith("# Kind")


async def test_replay_serves_the_recording_without_a_fake(tmp_path: Path) -> None:
    s = suite(tmp_path)
    await synthesize(s)
    async with s.replay_api() as api:
        with s.replay("read_file") as cassette:
            assert (await api.get("/files/a/b.txt")).json() == {"path": "a/b.txt"}
            with pytest.raises(CannotOverwriteExistingCassetteException):
                await api.get("/files/other")
        assert cassette.play_count == 1


def test_replay_errors_for_unknown_or_missing_cassettes(tmp_path: Path) -> None:
    s = suite(tmp_path)
    with pytest.raises(KeyError), s.replay("nope"):
        pass
    with pytest.raises(FileNotFoundError, match="bench synth example"), s.replay("read_file"):
        pass


async def test_recorded_cassettes_are_kept_unless_forced(tmp_path: Path) -> None:
    s = suite(tmp_path)
    s.cassette_dir.mkdir()
    s.path("read_file").write_text("recorded: true\n", encoding="utf-8")
    write_manifest(s.cassette_dir, {"read_file": "recorded"})

    result = await synthesize(s)
    assert (result.written, result.skipped_recorded) == ((), ("read_file",))
    assert s.path("read_file").read_text(encoding="utf-8") == "recorded: true\n"

    result = await synthesize(s, force=True)
    assert result.written == ("read_file",)
    assert read_manifest(s.cassette_dir) == {"read_file": "synthetic"}


async def test_obsolete_cassettes_are_removed_only_on_full_runs(tmp_path: Path) -> None:
    old = suite(tmp_path, read_file=Scenario(read_file), read_missing=Scenario(read_missing))
    await synthesize(old)
    new = suite(tmp_path)
    await synthesize(new, only=["read_file"])
    assert s_exists(new, "read_missing")
    result = await synthesize(new)
    assert result.removed == ("read_missing",)
    assert not s_exists(new, "read_missing")
    assert read_manifest(new.cassette_dir) == {"read_file": "synthetic"}


def s_exists(s: CassetteSuite, name: str) -> bool:
    return s.path(name).exists()


async def test_unknown_only_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unknown scenario"):
        await synthesize(suite(tmp_path), only=["nope"])


async def test_stale_detection(tmp_path: Path) -> None:
    s = suite(tmp_path, read_file=Scenario(read_file), read_missing=Scenario(read_missing))
    await synthesize(s)
    assert await stale_cassettes(s, tmp_path / "scratch") == []

    s.path("read_file").write_text("edited\n", encoding="utf-8")
    assert await stale_cassettes(s, tmp_path / "scratch") == ["read_file"]

    grown = suite(
        tmp_path,
        read_file=Scenario(read_file),
        read_missing=Scenario(read_missing),
        added=Scenario(read_file),
    )
    assert "added" in await stale_cassettes(grown, tmp_path / "scratch")


async def test_prepare_shapes_the_fake(tmp_path: Path) -> None:
    seen: list[FakeBackend] = []

    def prepare(fake: FakeBackend) -> None:
        seen.append(fake)
        fake.route("GET", "/extra")(lambda r: {"extra": True})

    async def run(api: ApiClient) -> None:
        assert (await api.get("/extra")).json() == {"extra": True}

    await synthesize(suite(tmp_path, extra=Scenario(run, prepare)))
    assert len(seen) == 1


async def test_path_params_span_slashes() -> None:
    fake = FakeBackend(require_auth=False)

    @fake.route("GET", "/repos/{owner}/{repo}/contents/{path:path}")
    def contents(r: FakeRequest) -> dict[str, str]:
        return dict(r.params)

    async with ApiClient(BASE, policy=RetryPolicy(), transport=fake.transport()) as api:
        body = (await api.get("/repos/o/r/contents/docs/a b.md")).json()
        assert body == {"owner": "o", "repo": "r", "path": "docs/a b.md"}
        assert (await api.get("/repos/o/r/contents")).status_code == 404


async def test_public_routes_skip_the_token_check() -> None:
    fake = FakeBackend()
    fake.route("POST", "/oauth/token", public=True)(lambda r: {"token_seen": r.token})
    fake.route("GET", "/private")(lambda r: {"ok": True})
    async with ApiClient(BASE, policy=RetryPolicy(), transport=fake.transport()) as api:
        assert (await api.post("/oauth/token")).json() == {"token_seen": None}
        assert (await api.get("/private")).status_code == 401
        assert (await api.get("/unknown")).status_code == 401  # no route leaks existence


async def test_expect_tool_error() -> None:
    async def fails() -> None:
        raise ToolError("nope")

    async def succeeds() -> None:
        return None

    await expect_tool_error(fails())
    with pytest.raises(AssertionError, match="expected a tool error"):
        await expect_tool_error(succeeds())
