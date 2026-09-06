from __future__ import annotations

import binascii
import json
import logging
import sys
from pathlib import Path
from typing import assert_never

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from pydantic import ValidationError

from hermes_cli.isolated_oauth_schema import (
    CapabilityError,
    Complete,
    Completion,
    Describe,
    DescribeReadOnly,
    Description,
    REQUEST_ADAPTER,
    Request,
    strict_json,
)
from hermes_cli.isolated_oauth_store import AuthSession, store_lock
from hermes_cli.isolated_oauth_wire import MAX_BODY, OpenAIWire


def execute(home: Path, request: Request, wire: OpenAIWire) -> Description | Completion:
    if not home.is_absolute() or not home.is_dir():
        raise CapabilityError("explicit_home_required")
    path = home.resolve(strict=True) / "auth.json"
    if path.is_symlink() or not path.is_file():
        raise CapabilityError("auth_store_unavailable")
    session = AuthSession(path, wire)
    match request.operation:
        case "describe_read_only" | "complete_read_only":
            return _perform(session, request, allow_refresh=False)
        case "describe" | "complete":
            with store_lock(path):
                return _perform(session, request, allow_refresh=True)
        case _:
            assert_never(request.operation)


def _perform(
    session: AuthSession, request: Request, *, allow_refresh: bool
) -> Description | Completion:
    match request:
        case Complete():
            expected = request.principal
        case Describe() | DescribeReadOnly():
            expected = None
        case _:
            assert_never(request)
    principal, headers = session.credentials(expected, allow_refresh=allow_refresh)
    models = session.wire.catalog(headers)
    match request:
        case Describe() | DescribeReadOnly():
            description = Description(principal=principal, models=models)
            session.reject_disclosure(description.model_dump_json())
            return description
        case Complete():
            selected = next(
                (entry for entry in models if entry.model == request.model), None
            )
            if selected is None:
                raise CapabilityError("unsupported_model")
            if request.reasoning_effort not in selected.reasoning_efforts:
                raise CapabilityError("unsupported_reasoning")
            reply = session.wire.complete(request, headers)
            completion = Completion(
                principal=principal,
                model=request.model,
                reasoning_effort=request.reasoning_effort,
                attempt_id=request.attempt_id,
                message=reply.message,
                reasoning_items=reply.reasoning_items,
                finish_reason="tool_calls" if reply.message.tool_calls else "stop",
            )
            session.reject_disclosure(completion.model_dump_json())
            return completion
        case _:
            assert_never(request)


def main() -> int:
    logging.disable(logging.CRITICAL)
    try:
        if len(sys.argv) != 3 or sys.argv[1] != "--home":
            raise CapabilityError("usage_requires_home")
        raw = sys.stdin.buffer.read(MAX_BODY + 1)
        if len(raw) > MAX_BODY:
            raise CapabilityError("request_too_large")
        request = REQUEST_ADAPTER.validate_json(strict_json(raw), strict=True)
        with httpx.Client(
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(120, connect=10),
            transport=httpx.HTTPTransport(retries=0, trust_env=False),
        ) as client:
            result = execute(Path(sys.argv[2]), request, OpenAIWire(client))
        _ = sys.stdout.write(result.model_dump_json() + "\n")
        return 0
    except CapabilityError as error:
        _ = sys.stdout.write(json.dumps({"error": str(error)}) + "\n")
        return 1
    except (
        ValidationError,
        OSError,
        ValueError,
        binascii.Error,
        RecursionError,
        httpx.HTTPError,
    ):
        _ = sys.stdout.write('{"error":"isolated_oauth_failed"}\n')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
