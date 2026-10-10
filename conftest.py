"""Repo-wide pytest config: async tests run on anyio's pytest plugin with asyncio."""

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
