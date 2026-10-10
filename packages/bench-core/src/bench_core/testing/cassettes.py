"""vcrpy configuration that redacts credentials and personal data from cassettes.

vcrpy applies `before_record_request` both when recording and when matching a request
during replay. A replayed request carrying a real token is therefore redacted the same
way and still matches the recording.

Redaction has two layers:
1. Structured: sensitive headers, query/form parameters and JSON fields by name.
2. Pattern-based on all remaining text: known token formats and e-mail addresses.

`find_secrets` runs the same patterns as a check (used by `bench record` before gitleaks).
"""

import copy
import json
import re
from typing import Any, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import vcr
from vcr.request import Request

REDACTED = "REDACTED"
REDACTED_EMAIL = "redacted@example.invalid"

SENSITIVE_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key"}
)
# Set by the HTTP stack, not by our code; `accept-encoding` even differs by Python version
# (zstd), which would make regenerated cassettes differ across CI jobs. Never matched on.
TRANSPORT_HEADERS = frozenset({"accept-encoding", "connection"})
# Query and form parameters (OAuth grants, Slack's `token=`, Google's `key=`).
SENSITIVE_PARAMS = frozenset(
    {
        "access_token",
        "refresh_token",
        "id_token",
        "token",
        "client_secret",
        "client_assertion",
        "assertion",
        "code",
        "code_verifier",
        "password",
        "key",
    }
)
# JSON fields. Narrower than SENSITIVE_PARAMS: `key` (Jira issue keys) and `code`
# (GitHub validation errors) are ordinary data in API bodies.
SENSITIVE_FIELDS = frozenset(
    {
        "access_token",
        "refresh_token",
        "id_token",
        "token",
        "client_secret",
        "client_assertion",
        "code_verifier",
        "password",
        "private_key",
    }
)

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    "github_pat": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    "slack_token": re.compile(r"\bxox[abeoprs]-[A-Za-z0-9-]{10,}"),
    "atlassian_api_token": re.compile(r"\bATATT[A-Za-z0-9_=-]{20,}"),
    "google_access_token": re.compile(r"\bya29\.[A-Za-z0-9_-]{20,}"),
    "google_refresh_token": re.compile(r"\b1//[A-Za-z0-9_-]{20,}"),
    "google_api_key": re.compile(r"\bAIza[A-Za-z0-9_-]{35}"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    "private_key": re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL
    ),
}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")


def scrub(text: str) -> str:
    """Replace known token formats and e-mail addresses in free text."""
    for pattern in SECRET_PATTERNS.values():
        text = pattern.sub(REDACTED, text)
    return EMAIL.sub(REDACTED_EMAIL, text)


def find_secrets(text: str) -> list[str]:
    """Names of the patterns that still match (never the values, so it is safe to print)."""
    found = [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(text)]
    if any(m.group() != REDACTED_EMAIL for m in EMAIL.finditer(text)):
        found.append("email")
    return found


def _body(request: Request) -> bytes:
    # vcrpy's setter accepts streams, but the getter always returns bytes (or None).
    return cast(bytes | None, request.body) or b""


def _redact_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: REDACTED if k.lower() in SENSITIVE_FIELDS and isinstance(v, str) else _redact_json(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_json(v) for v in value]
    return value


def _redact_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(k, REDACTED if k.lower() in SENSITIVE_PARAMS else v) for k, v in pairs]


def _redact_body(body: bytes | str | None, content_type: str) -> bytes | str | None:
    if not body:
        return body
    is_bytes = isinstance(body, bytes)
    try:
        text = body.decode() if isinstance(body, bytes) else body
    except UnicodeDecodeError:
        return body  # binary (file downloads): leave untouched
    if "x-www-form-urlencoded" in content_type:
        pairs = parse_qsl(text, keep_blank_values=True)
        redacted = _redact_pairs(pairs)
        if redacted != pairs:
            text = urlencode(redacted)
    else:
        try:
            data = json.loads(text)
        except ValueError:
            pass
        else:
            redacted = _redact_json(data)
            if redacted != data:
                text = json.dumps(redacted, ensure_ascii=False, separators=(",", ":"))
    text = scrub(text)
    return text.encode() if is_bytes else text


def redact_request(request: Request) -> Request:
    request = copy.deepcopy(request)
    for name in list(request.headers):
        if name.lower() in TRANSPORT_HEADERS:
            del request.headers[name]
        elif name.lower() in SENSITIVE_HEADERS:
            request.headers[name] = REDACTED
        else:
            request.headers[name] = scrub(request.headers[name])

    parts = urlsplit(request.uri)
    if parts.query:
        pairs = parse_qsl(parts.query, keep_blank_values=True)
        redacted = _redact_pairs(pairs)
        if redacted != pairs:
            request.uri = urlunsplit(parts._replace(query=urlencode(redacted)))

    content_type = request.headers.get("content-type", "") or ""
    request.body = _redact_body(_body(request), content_type.lower())
    return request


def redact_response(response: dict[str, Any]) -> dict[str, Any]:
    response = copy.deepcopy(response)
    headers: dict[str, list[str]] = response["headers"]
    for name, values in headers.items():
        if name.lower() in SENSITIVE_HEADERS:
            headers[name] = [REDACTED for _ in values]
        else:
            headers[name] = [scrub(v) for v in values]

    content_type = next((v[0] for k, v in headers.items() if k.lower() == "content-type" and v), "")
    body = _redact_body(response["body"]["string"], content_type.lower())
    response["body"]["string"] = body
    for name in headers:
        if name.lower() == "content-length" and body is not None:
            raw = body.encode() if isinstance(body, str) else body
            headers[name] = [str(len(raw))]
    return response


def json_body(r1: Request, r2: Request) -> None:
    """vcrpy matcher: bodies equal as JSON when both parse, else byte-equal."""
    b1, b2 = _body(r1), _body(r2)
    try:
        if json.loads(b1) == json.loads(b2):
            return
    except ValueError:
        pass
    assert b1 == b2, "request bodies differ"


MATCH_ON = ("method", "scheme", "host", "port", "path", "query", "json_body")


def cassette_config() -> dict[str, Any]:
    """Keyword arguments for `vcr.VCR` / the value of pytest-recording's `vcr_config`."""
    return {
        "before_record_request": redact_request,
        "before_record_response": redact_response,
        "decode_compressed_response": True,
        "match_on": MATCH_ON,
    }


def register_matchers(instance: vcr.VCR) -> None:
    instance.register_matcher("json_body", json_body)


def make_vcr(**overrides: Any) -> vcr.VCR:
    instance = vcr.VCR(**{**cassette_config(), **overrides})
    register_matchers(instance)
    return instance
