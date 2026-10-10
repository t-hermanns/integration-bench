"""In-memory Slack API for tests, and its default data."""

from bench_slack.fake.backend import SlackFake, credential
from bench_slack.fake.seed import (
    GENERAL,
    MESSAGES,
    OTHER_CHANNELS,
    RELEASE,
    SANDBOX,
    seed,
    seeded_fake,
)

__all__ = [
    "GENERAL",
    "MESSAGES",
    "OTHER_CHANNELS",
    "RELEASE",
    "SANDBOX",
    "SlackFake",
    "credential",
    "seed",
    "seeded_fake",
]
