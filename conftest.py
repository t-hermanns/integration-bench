"""Repo-wide pytest config: anyio backend and redacting cassettes for pytest-recording."""

from typing import Any

import pytest
from vcr import VCR

from bench_core.testing import cassette_config, register_matchers


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, Any]:
    return cassette_config()


def pytest_recording_configure(config: pytest.Config, vcr: VCR) -> None:
    register_matchers(vcr)
