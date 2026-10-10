"""`bench sandbox`: shared pieces of the per-integration `bench_<name>.sandbox` modules.

`bench_<name>.sandbox.seed(env, echo=...)` creates the fake's seed data in the live test
account, so scenarios and their assertions hold for synthetic and recorded cassettes
alike. It is idempotent: an already seeded sandbox is left alone, anything else that is
not empty is refused (issue numbers would not match the fake).
"""

from collections.abc import Callable, Mapping

from bench_core.doctor import missing

Echo = Callable[[str], None]


class SandboxError(Exception):
    """The sandbox is not in a state the command can work from; the message says why."""


def require(env: Mapping[str, str], *names: str) -> None:
    if absent := missing(env, *names):
        raise SandboxError(f"set {', '.join(absent)} (see `bench doctor`)")
