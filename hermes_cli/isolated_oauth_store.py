from __future__ import annotations

import base64
import os
import stat
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, ClassVar, assert_never

from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr

from hermes_cli import __version__
from hermes_cli.isolated_oauth_schema import CapabilityError, Principal, strict_json
from hermes_cli.isolated_oauth_wire import JSON_ADAPTER, MAX_BODY, OpenAIWire


class Tokens(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore", strict=True, frozen=True
    )
    access_token: SecretStr
    refresh_token: SecretStr


class AuthClaims(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore", strict=True, frozen=True
    )
    chatgpt_account_id: str = Field(min_length=1)


class Claims(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore", strict=True, frozen=True
    )
    sub: str = Field(min_length=1)
    exp: int
    auth: AuthClaims = Field(alias="https://api.openai.com/auth")


def _string_leaves(value: JsonValue) -> Generator[str, None, None]:
    match value:
        case str():
            yield value
        case list():
            for item in value:
                yield from _string_leaves(item)
        case dict():
            for key, item in value.items():
                yield key
                yield from _string_leaves(item)
        case int() | float() | None:
            return
        case _:
            assert_never(value)


def principal_for(tokens: Tokens) -> tuple[Principal, int]:
    pieces = tokens.access_token.get_secret_value().split(".")
    if len(pieces) != 3 or not tokens.refresh_token.get_secret_value():
        raise CapabilityError("principal_unavailable")
    claims = Claims.model_validate_json(
        strict_json(base64.urlsafe_b64decode(pieces[1] + "=" * (-len(pieces[1]) % 4)))
    )
    return Principal(
        account_id=claims.auth.chatgpt_account_id, subject=claims.sub
    ), claims.exp


def _kernel_lock(handle: BinaryIO, acquire: bool) -> None:
    if os.name == "nt":
        import msvcrt

        _ = handle.seek(0)
        msvcrt.locking(
            handle.fileno(), msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK, 1
        )
    else:
        import fcntl

        fcntl.flock(
            handle.fileno(),
            (fcntl.LOCK_EX | fcntl.LOCK_NB) if acquire else fcntl.LOCK_UN,
        )


@contextmanager
def store_lock(path: Path) -> Generator[None, None, None]:
    with path.with_suffix(".lock").open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            _ = handle.write(b" ")
            handle.flush()
        deadline = time.monotonic() + 30
        while True:
            try:
                _kernel_lock(handle, True)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise CapabilityError("auth_lock_timeout") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            _kernel_lock(handle, False)


def read_store(path: Path) -> dict[str, JsonValue]:
    with path.open("rb") as handle:
        body = handle.read(MAX_BODY + 1)
    if len(body) > MAX_BODY:
        raise CapabilityError("auth_store_too_large")
    return JSON_ADAPTER.validate_json(
        strict_json(body.decode("utf-8-sig")), strict=True
    )


def write_store(path: Path, store: dict[str, JsonValue]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix="auth.json.tmp.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
            _ = handle.write(JSON_ADAPTER.dump_json(store))
            handle.flush()
            os.fsync(handle.fileno())
        except OSError:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class AuthSession:
    def __init__(self, path: Path, wire: OpenAIWire) -> None:
        self.path: Path = path
        self.wire: OpenAIWire = wire
        self.disclosure_secrets: tuple[SecretStr, ...] = ()

    def reject_disclosure(self, response: str) -> None:
        payload = JSON_ADAPTER.validate_json(response, strict=True)
        for leaf in _string_leaves(payload):
            if any(
                secret.get_secret_value() in leaf for secret in self.disclosure_secrets
            ):
                raise CapabilityError("credential_in_response")

    def credentials(
        self, expected: Principal | None, *, allow_refresh: bool
    ) -> tuple[Principal, dict[str, str]]:
        store = read_store(self.path)
        providers = JSON_ADAPTER.validate_python(store.get("providers"), strict=True)
        state = JSON_ADAPTER.validate_python(providers.get("openai-codex"), strict=True)
        if state.get("auth_mode") != "chatgpt":
            raise CapabilityError("oauth_required")
        token_fields = JSON_ADAPTER.validate_python(state.get("tokens"), strict=True)
        tokens = Tokens.model_validate(token_fields)
        self.disclosure_secrets = (tokens.access_token, tokens.refresh_token)
        principal, expires = principal_for(tokens)
        if expected is not None and principal != expected:
            raise CapabilityError("principal_mismatch")
        if expires <= time.time() + 120:
            if not allow_refresh:
                raise CapabilityError("refresh_required_read_only")
            refreshed = self.wire.refresh(tokens.refresh_token.get_secret_value())
            _ = refreshed.setdefault(
                "refresh_token", tokens.refresh_token.get_secret_value()
            )
            updated = Tokens.model_validate(refreshed)
            self.disclosure_secrets += (updated.access_token, updated.refresh_token)
            next_principal, next_expires = principal_for(updated)
            if next_principal != principal or next_expires <= time.time() + 120:
                raise CapabilityError("refresh_principal_or_expiry_mismatch")
            token_fields.update(
                access_token=updated.access_token.get_secret_value(),
                refresh_token=updated.refresh_token.get_secret_value(),
            )
            state["tokens"] = token_fields
            state["last_refresh"] = datetime.now(timezone.utc).isoformat()
            providers["openai-codex"] = state
            store["providers"] = providers
            write_store(self.path, store)
            tokens = updated
        return principal, {
            "Authorization": "Bearer " + tokens.access_token.get_secret_value(),
            "ChatGPT-Account-Id": principal.account_id,
            "Accept": "text/event-stream",
            "User-Agent": f"HermesAgent/{__version__}",
            "originator": "hermes-agent",
        }
