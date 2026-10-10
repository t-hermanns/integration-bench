"""Injectable time source so retry/backoff logic stays deterministic under test."""

import random
import time
from typing import Protocol

import anyio


class Clock(Protocol):
    def time(self) -> float:
        """Wall-clock seconds since the epoch (needed for `Retry-After` dates and resets)."""
        ...

    async def sleep(self, seconds: float) -> None: ...

    def random(self) -> float:
        """Uniform float in [0, 1), used for jitter."""
        ...


class SystemClock:
    def time(self) -> float:
        return time.time()

    async def sleep(self, seconds: float) -> None:
        await anyio.sleep(seconds)

    def random(self) -> float:
        return random.random()
