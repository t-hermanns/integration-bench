"""Todoist credentials.

**Static token** (API key or bot token): never refreshed; a 401 means it was revoked.
The granted scopes are unknown, so every tool stays registered.
"""

from bench_core.auth import Credential, StaticToken
from bench_core.permissions import Granted


def static_token(account: str, token: str) -> StaticToken:
    return StaticToken(account, token)


async def granted_scopes(credential: Credential) -> Granted:
    """Static tokens do not report their scopes: None keeps every tool registered."""
    return None
