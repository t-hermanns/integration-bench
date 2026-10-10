"""Spike: vcrpy records and replays httpx2 traffic (sync + async) on Python 3.14.

Records against a throwaway local HTTP server, shuts it down, then replays from the
cassette with network access blocked by pytest-recording.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx2
import pytest
import vcr
from vcr.errors import CannotOverwriteExistingCassetteException
from vcr.record_mode import RecordMode


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._reply({"path": self.path, "auth": self.headers.get("Authorization")})

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self._reply({"echo": json.loads(body)}, status=201)

    def _reply(self, payload: dict[str, object], status: int = 200) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Link", '<http://example.invalid/items?page=2>; rel="next"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        pass


def _vcr() -> vcr.VCR:
    return vcr.VCR(
        record_mode=RecordMode.ONCE,
        filter_headers=["authorization"],
        match_on=["method", "scheme", "host", "port", "path", "query", "body"],
    )


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    cassette = tmp_path_factory.mktemp("cassettes") / "spike.yaml"

    with _vcr().use_cassette(str(cassette)) as cass:
        with httpx2.Client(base_url=base, headers={"Authorization": "Bearer secret"}) as c:
            assert c.get("/items", params={"page": 1}).json()["path"] == "/items?page=1"
            assert c.post("/items", json={"title": "x"}).status_code == 201
        assert len(cass) == 2

    server.shutdown()
    server.server_close()
    return cassette, base


def test_cassette_redacts_authorization(recorded: tuple[Path, str]) -> None:
    cassette, _ = recorded
    text = cassette.read_text()
    assert "Bearer secret" not in text.split('"auth"')[0]  # request header filtered
    assert "authorization" not in text.lower().split("response:")[0]


@pytest.mark.block_network
def test_replay_sync_without_network(recorded: tuple[Path, str]) -> None:
    cassette, base = recorded
    with _vcr().use_cassette(str(cassette), record_mode=RecordMode.NONE) as cass:
        with httpx2.Client(base_url=base) as c:
            r = c.get("/items", params={"page": 1})
            assert r.json()["path"] == "/items?page=1"
            assert r.links["next"]["url"] == "http://example.invalid/items?page=2"
            r = c.post("/items", json={"title": "x"})
            assert r.status_code == 201
            assert r.json() == {"echo": {"title": "x"}}
        assert cass.all_played


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# On Windows the asyncio proactor loop builds its self-pipe via a loopback socketpair,
# so loopback must stay allowed or the event loop cannot even start.
@pytest.mark.anyio
@pytest.mark.block_network(allowed_hosts=["127.0.0.1"])
async def test_replay_async_without_network(recorded: tuple[Path, str]) -> None:
    cassette, base = recorded
    with _vcr().use_cassette(str(cassette), record_mode=RecordMode.NONE):
        async with httpx2.AsyncClient(base_url=base) as c:
            r = await c.get("/items", params={"page": 1})
            assert r.json()["path"] == "/items?page=1"


@pytest.mark.block_network
def test_unrecorded_request_fails(recorded: tuple[Path, str]) -> None:
    cassette, base = recorded
    with (
        _vcr().use_cassette(str(cassette), record_mode=RecordMode.NONE),
        httpx2.Client(base_url=base) as c,
        pytest.raises(CannotOverwriteExistingCassetteException),
    ):
        c.get("/other")


@pytest.mark.block_network
def test_live_network_is_blocked() -> None:
    with httpx2.Client() as c, pytest.raises(Exception, match=r"(?i)network|socket"):
        c.get("http://192.0.2.1/")  # TEST-NET-1, never routable
