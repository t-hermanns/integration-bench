"""Default data for the fake, shaped like the live sandbox.

`bench sandbox seed todoist` creates the same tasks in the real test account, so the
scenarios and their assertions hold for synthetic and recorded cassettes alike. Keep both
in sync when changing this data.
"""

from datetime import UTC, datetime, timedelta

from bench_core.clock import Clock
from bench_core.testing import FakeClock

from bench_todoist.fake.backend import TodoistFake

FIRST_ID = "6Fake00000001001"
"""Fake id of the first seeded task (live ids differ)."""

_START = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)

# (title, description); Todoist lists only open tasks, so there are no closed ones.
TASKS: list[tuple[str, str | None]] = [
    ("Crash when config file is empty", "An empty config raises KeyError: 'name'."),
    ("Add dark mode to the dashboard", "Users asked for a dark theme."),
    ("Timeout when exporting large reports", "Exports over 10k rows time out."),
    ("Document the plugin API", None),
    ("Login fails with SSO", "Fixed by renewing the IdP certificate."),
    ("Upgrade to the new HTTP client", None),
    ("Crash on startup without network", "No offline fallback for the update check."),
    ("Translate the settings page", "German and French first."),
    ("Flaky export test", "Fixed the clock in the test."),
    ("Report totals off by one", "The last row is not counted."),
    ("Add CSV import", None),
    ("Crash when saving a report twice", "Second save hits a stale file handle."),
]


def at(hours: int) -> str:
    return (_START + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S.000000Z")


def seed(fake: TodoistFake) -> None:
    for hour, (title, description) in enumerate(TASKS):
        fake.add_task(title, description=description, created_at=at(hour * 5))


def seeded_fake(clock: Clock | None = None, *, require_auth: bool = True) -> TodoistFake:
    fake = TodoistFake(require_auth=require_auth, clock=clock or FakeClock())
    seed(fake)
    return fake
