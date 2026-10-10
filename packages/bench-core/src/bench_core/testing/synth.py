"""Cassette suites: scenarios that produce the cassettes an integration's tests replay.

A scenario is a short script against the integration's client (search, then fetch page
two, ...). The same scenarios produce both kinds of cassettes:

- `bench synth <name>` runs them against the integration's in-memory fake and records
  the traffic with vcrpy -> `synthetic` cassettes (development, before accounts exist).
- `bench record <name>` (later) runs them against the live sandbox -> `recorded`.

Tests replay a cassette with `suite.replay(name)` and never know which kind it is.
`cassettes/MANIFEST.yaml` records the kind of every cassette; `synthesize` does not
overwrite recorded cassettes unless forced.
"""

import os
import tempfile
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anyio
import httpx2
import yaml
from vcr.cassette import Cassette
from vcr.record_mode import RecordMode

from bench_core.http import ApiClient
from bench_core.testing.cassettes import make_vcr
from bench_core.testing.clock import FakeClock
from bench_core.testing.fake import FakeBackend

Source = Literal["synthetic", "recorded"]
MANIFEST = "MANIFEST.yaml"
MANIFEST_HEADER = (
    "# Kind of every cassette in this directory. `bench synth` writes `synthetic` ones from\n"
    "# the in-memory fake; `bench record` replaces them with `recorded` ones from the sandbox.\n"
)


@dataclass(frozen=True)
class Scenario:
    run: Callable[[ApiClient], Awaitable[Any]]
    """Drives the API client; expected tool errors should be caught inside."""
    prepare: Callable[[FakeBackend], None] | None = None
    """Synthetic only: shape the fake before the run (inject faults, revoke scopes)."""
    description: str = ""


@dataclass(frozen=True)
class CassetteSuite:
    integration: str
    cassette_dir: Path
    scenarios: Mapping[str, Scenario]
    fake: Callable[[FakeClock], FakeBackend]
    """A freshly seeded fake per scenario, so cassettes do not depend on run order."""
    api: Callable[[httpx2.AsyncBaseTransport | None, FakeClock | None], ApiClient]
    """API client for a transport (None = real network, patched by vcrpy on replay)."""

    def path(self, name: str) -> Path:
        return self.cassette_dir / f"{name}.yaml"

    @contextmanager
    def replay(self, name: str) -> Iterator[Cassette]:
        """Replay-only cassette context; unmatched requests raise instead of going out."""
        if name not in self.scenarios:
            raise KeyError(f"{self.integration}: no scenario {name!r}")
        path = self.path(name)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run `bench synth {self.integration}`")
        with make_vcr(record_mode=RecordMode.NONE).use_cassette(str(path)) as cassette:
            yield cassette

    def replay_api(self) -> ApiClient:
        return self.api(None, FakeClock())


def read_manifest(cassette_dir: Path) -> dict[str, Source]:
    path = cassette_dir / MANIFEST
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(data.get("cassettes") or {})


def write_manifest(cassette_dir: Path, entries: Mapping[str, Source]) -> None:
    body = yaml.safe_dump({"cassettes": dict(sorted(entries.items()))}, sort_keys=False)
    _write_text(cassette_dir / MANIFEST, MANIFEST_HEADER + body)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


async def record_scenario(suite: CassetteSuite, name: str, path: Path) -> None:
    """Run one scenario against a fresh fake and write its cassette to `path`."""
    scenario = suite.scenarios[name]
    clock = FakeClock()
    fake = suite.fake(clock)
    if scenario.prepare:
        scenario.prepare(fake)
    target = anyio.Path(path)
    await target.unlink(missing_ok=True)  # record_mode=all appends to an existing cassette
    await target.parent.mkdir(parents=True, exist_ok=True)
    async with suite.api(fake.transport(), clock) as api:
        with make_vcr(record_mode=RecordMode.ALL).use_cassette(str(path)):
            await scenario.run(api)
    # vcrpy writes with the platform newline; keep cassettes byte-identical across OSes.
    _write_text(path, await target.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class SynthResult:
    written: tuple[str, ...]
    skipped_recorded: tuple[str, ...]
    removed: tuple[str, ...]


async def synthesize(
    suite: CassetteSuite, *, only: Sequence[str] | None = None, force: bool = False
) -> SynthResult:
    """(Re)generate synthetic cassettes and update the manifest.

    Recorded cassettes are kept unless `force`. Without `only`, cassettes of scenarios
    that no longer exist are removed.
    """
    unknown = sorted(set(only or ()) - set(suite.scenarios))
    if unknown:
        raise KeyError(f"{suite.integration}: unknown scenario(s) {', '.join(unknown)}")
    manifest = read_manifest(suite.cassette_dir)
    written: list[str] = []
    skipped: list[str] = []
    for name in only or sorted(suite.scenarios):
        if manifest.get(name) == "recorded" and not force:
            skipped.append(name)
            continue
        await record_scenario(suite, name, suite.path(name))
        manifest[name] = "synthetic"
        written.append(name)
    removed: list[str] = []
    if only is None:
        for name in sorted(set(manifest) - set(suite.scenarios)):
            suite.path(name).unlink(missing_ok=True)
            del manifest[name]
            removed.append(name)
    write_manifest(suite.cassette_dir, manifest)
    return SynthResult(tuple(written), tuple(skipped), tuple(removed))


async def stale_cassettes(suite: CassetteSuite, scratch: Path) -> list[str]:
    """Synthetic cassettes that differ from a fresh synthesis (or are missing/untracked).

    Used by `bench synth --check` and the integration's own test, so a change to the fake
    or a scenario cannot silently leave old cassettes behind.
    """
    manifest = read_manifest(suite.cassette_dir)
    stale = sorted(set(suite.scenarios) - set(manifest))
    stale += sorted(set(manifest) - set(suite.scenarios))
    for name, source in sorted(manifest.items()):
        if source != "synthetic" or name not in suite.scenarios:
            continue
        fresh = scratch / f"{name}.yaml"
        await record_scenario(suite, name, fresh)
        current = suite.path(name)
        if not current.exists() or current.read_bytes() != fresh.read_bytes():
            stale.append(name)
    return stale
