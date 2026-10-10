"""Test helpers shared by all integrations: fake backends, cassette redaction, fake clock."""

from bench_core.testing.cassettes import (
    REDACTED,
    REDACTED_EMAIL,
    cassette_config,
    find_secrets,
    json_body,
    make_vcr,
    redact_request,
    redact_response,
    register_matchers,
    scrub,
)
from bench_core.testing.clock import FakeClock
from bench_core.testing.fake import FakeBackend, FakeRequest, Fault

__all__ = [
    "REDACTED",
    "REDACTED_EMAIL",
    "FakeBackend",
    "FakeClock",
    "FakeRequest",
    "Fault",
    "cassette_config",
    "find_secrets",
    "json_body",
    "make_vcr",
    "redact_request",
    "redact_response",
    "register_matchers",
    "scrub",
]
