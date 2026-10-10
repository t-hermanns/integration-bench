"""A clock whose `sleep` advances virtual time instantly and records every wait."""


class FakeClock:
    def __init__(self, start: float = 1_800_000_000.0, rand: float = 0.5) -> None:
        self.now = start
        self.rand = rand
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def random(self) -> float:
        return self.rand

    def advance(self, seconds: float) -> None:
        """Let time pass without recording a sleep (e.g. to make a token expire)."""
        self.now += seconds
