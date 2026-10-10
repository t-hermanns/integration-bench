"""Default data for the fake, shaped like the live sandbox.

`bench sandbox seed slack` posts the same messages to `#bench-sandbox` in the real
workspace (as the `bench-ci` bot), so the scenarios and their assertions hold for
synthetic and recorded cassettes alike. Keep both in sync when changing this data.

Slack assigns channel ids and message timestamps itself, so scenarios look them up by
channel name and message text, never by a hard-coded id. The live workspace also has
Slack's default channels; the fake's `general` and `random` stand in for them.
"""

from dataclasses import dataclass

from bench_core.clock import Clock
from bench_core.testing import FakeClock

from bench_slack.fake.backend import SlackFake

SANDBOX = "bench-sandbox"
OTHER_CHANNELS = ("general", "random")
"""Channels the app is not a member of (reading them fails with `not_in_channel`)."""


@dataclass(frozen=True)
class SeedMessage:
    text: str
    replies: tuple[str, ...] = ()


MESSAGES: tuple[SeedMessage, ...] = (
    SeedMessage("Welcome to #bench-sandbox. integration-bench tests read and write here."),
    SeedMessage(
        "Release 2.4 is planned for Thursday. Anything still blocking?",
        (
            "QA sign-off for the export fix is still missing.",
            "I will finish QA by Wednesday noon.",
        ),
    ),
    SeedMessage("Reminder: the staging database is read-only until Monday."),
    SeedMessage(
        "Who owns the flaky export test?",
        ("That is me, the fix is in review.",),
    ),
    SeedMessage("Standup moves to 9:30 starting next week."),
)
"""Top-level messages oldest first, each with its replies (oldest first)."""

RELEASE = MESSAGES[1].text
"""The thread with two replies."""


def seed(fake: SlackFake) -> None:
    for name in OTHER_CHANNELS:
        fake.add_channel(name, is_member=False, is_general=name == "general")
    sandbox = fake.add_channel(SANDBOX, purpose="Test channel for integration-bench.")
    for message in MESSAGES:
        parent = fake.add_message(sandbox["id"], message.text)
        for reply in message.replies:
            fake.add_message(sandbox["id"], reply, thread_ts=parent["ts"])


def seeded_fake(clock: Clock | None = None, *, require_auth: bool = True) -> SlackFake:
    fake = SlackFake(require_auth=require_auth, clock=clock or FakeClock())
    seed(fake)
    return fake
