"""Test helpers shared by all integrations: fakes, cassettes (redaction, synthesis), fake clock."""

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
from bench_core.testing.synth import (
    CassetteSuite,
    Scenario,
    SynthResult,
    read_manifest,
    record_scenario,
    stale_cassettes,
    synthesize,
    write_manifest,
)

__all__ = [
    "REDACTED",
    "REDACTED_EMAIL",
    "CassetteSuite",
    "FakeBackend",
    "FakeClock",
    "FakeRequest",
    "Fault",
    "Scenario",
    "SynthResult",
    "cassette_config",
    "find_secrets",
    "json_body",
    "make_vcr",
    "read_manifest",
    "record_scenario",
    "redact_request",
    "redact_response",
    "register_matchers",
    "scrub",
    "stale_cassettes",
    "synthesize",
    "write_manifest",
]
