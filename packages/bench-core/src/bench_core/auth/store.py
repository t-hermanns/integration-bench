"""Token sets and where they live.

Backends:
- `MemoryTokenStore`  tests and short-lived processes
- `FileTokenStore`    one JSON file per account, written atomically
- `KeyringTokenStore` OS credential store (Windows Credential Manager), chunked
- `EnvTokenStore`     read-only, for CI: non-rotating machine credentials from env vars
"""

import contextlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, Self

import keyring
import keyring.errors


@dataclass(frozen=True)
class TokenSet:
    access_token: str | None = None
    refresh_token: str | None = None
    expires_at: float | None = None
    """Epoch seconds; None means "does not expire" (or unknown)."""
    scopes: frozenset[str] = field(default_factory=frozenset)

    def expires_within(self, now: float, seconds: float) -> bool:
        return self.expires_at is not None and self.expires_at - now <= seconds

    def to_json(self) -> str:
        data: dict[str, Any] = {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "scopes": sorted(self.scopes),
        }
        return json.dumps(data, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> Self:
        data = json.loads(raw)
        return cls(
            access_token=data.get("access_token"),
            refresh_token=data.get("refresh_token"),
            expires_at=data.get("expires_at"),
            scopes=frozenset(data.get("scopes") or ()),
        )


class TokenStore(Protocol):
    def load(self, account: str) -> TokenSet | None: ...

    def save(self, account: str, tokens: TokenSet) -> None:
        """Persist durably before returning; callers rely on this for rotating tokens."""
        ...


class ReadOnlyStoreError(RuntimeError):
    pass


class MemoryTokenStore:
    def __init__(self, initial: Mapping[str, TokenSet] | None = None) -> None:
        self.tokens: dict[str, TokenSet] = dict(initial or {})
        self.saves = 0

    def load(self, account: str) -> TokenSet | None:
        return self.tokens.get(account)

    def save(self, account: str, tokens: TokenSet) -> None:
        self.saves += 1
        self.tokens[account] = tokens


_SAFE_ACCOUNT = re.compile(r"^[A-Za-z0-9_.-]+$")


def _check_account(account: str) -> str:
    if not _SAFE_ACCOUNT.match(account):
        raise ValueError(f"invalid account name {account!r}")
    return account


class FileTokenStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, account: str) -> Path:
        return self.directory / f"{_check_account(account)}.json"

    def load(self, account: str) -> TokenSet | None:
        path = self._path(account)
        if not path.exists():
            return None
        return TokenSet.from_json(path.read_text(encoding="utf-8"))

    def save(self, account: str, tokens: TokenSet) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self._path(account)
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=f".{account}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(tokens.to_json())
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


class KeyringTokenStore:
    """Windows Credential Manager caps a secret at 2560 bytes (UTF-16, ~1280 chars) and
    keyring does not split, while Atlassian JWT access tokens alone can exceed that.

    Layout: `<account>` holds a pointer `<generation>:<count>`; chunks live under
    `<account>@<generation>#<i>`. A save writes the other generation's chunks first and
    flips the pointer last, so a crash mid-save leaves the previous value intact.
    """

    CHUNK = 1000

    def __init__(self, service: str = "integration-bench") -> None:
        self.service = service

    def _pointer(self, account: str) -> tuple[str, int] | None:
        raw = keyring.get_password(self.service, _check_account(account))
        if raw is None:
            return None
        generation, _, count = raw.partition(":")
        return generation, int(count)

    def load(self, account: str) -> TokenSet | None:
        pointer = self._pointer(account)
        if pointer is None:
            return None
        generation, count = pointer
        parts: list[str] = []
        for i in range(count):
            part = keyring.get_password(self.service, f"{account}@{generation}#{i}")
            if part is None:
                return None
            parts.append(part)
        return TokenSet.from_json("".join(parts))

    def save(self, account: str, tokens: TokenSet) -> None:
        raw = tokens.to_json()
        chunks = [raw[i : i + self.CHUNK] for i in range(0, len(raw), self.CHUNK)]
        old = self._pointer(account)
        generation = "b" if old is not None and old[0] == "a" else "a"
        for i, chunk in enumerate(chunks):
            keyring.set_password(self.service, f"{account}@{generation}#{i}", chunk)
        keyring.set_password(self.service, account, f"{generation}:{len(chunks)}")
        if old is not None:
            for i in range(old[1]):
                with contextlib.suppress(keyring.errors.PasswordDeleteError):
                    keyring.delete_password(self.service, f"{account}@{old[0]}#{i}")


class EnvTokenStore:
    """Read-only store for CI. `mapping` = account -> {TokenSet field: env var name}.

    Fresh access tokens obtained via a (non-rotating) refresh token are cached in memory.
    A rotated refresh token cannot be persisted in CI, so saving one is an error.
    """

    def __init__(
        self,
        mapping: Mapping[str, Mapping[str, str]],
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.mapping = mapping
        self.environ = os.environ if environ is None else environ
        self._cache: dict[str, TokenSet] = {}

    def _from_env(self, account: str) -> TokenSet | None:
        fields = self.mapping.get(account)
        if not fields:
            return None
        values = {name: self.environ.get(var) for name, var in fields.items()}
        if not any(values.values()):
            return None
        scopes = values.pop("scopes", None)
        return TokenSet(
            access_token=values.get("access_token"),
            refresh_token=values.get("refresh_token"),
            scopes=frozenset(re.split(r"[\s,]+", scopes.strip())) if scopes else frozenset(),
        )

    def load(self, account: str) -> TokenSet | None:
        return self._cache.get(account) or self._from_env(account)

    def save(self, account: str, tokens: TokenSet) -> None:
        current = self._from_env(account)
        if current is None:
            raise ReadOnlyStoreError(f"no env credentials configured for {account!r}")
        if tokens.refresh_token != current.refresh_token:
            raise ReadOnlyStoreError(
                f"{account}: the provider rotated the refresh token, but env credentials are "
                "read-only. Use a non-rotating credential in CI."
            )
        self._cache[account] = replace(tokens, refresh_token=current.refresh_token)


TOKEN_DIR_VAR = "BENCH_TOKEN_DIR"


def default_store(env: Mapping[str, str]) -> TokenStore:
    """Where `bench auth login` keeps user tokens: the OS keyring, or JSON files in
    `BENCH_TOKEN_DIR` when set (tests, machines without a keyring backend)."""
    directory = env.get(TOKEN_DIR_VAR)
    return FileTokenStore(Path(directory)) if directory else KeyringTokenStore()
