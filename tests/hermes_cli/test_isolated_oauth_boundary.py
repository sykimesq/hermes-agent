import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from hermes_cli.isolated_oauth_schema import REQUEST_ADAPTER, CapabilityError
from tests.hermes_cli.conftest_isolated_oauth import (
    auth_home,
    catalog,
    completed,
    invoke,
    request,
    token,
)


def test_read_only_catalog_never_refreshes_or_writes(tmp_path):
    home = auth_home(tmp_path)
    original = (home / "auth.json").read_bytes()
    seen = []
    attempt = REQUEST_ADAPTER.validate_json(
        '{"operation":"describe_read_only","provider":"openai_oauth"}'
    )

    def handler(incoming):
        seen.append(incoming.method)
        return httpx.Response(200, json=catalog())

    result = invoke(home, attempt, handler)
    assert result.models[0].model == "model-a"
    assert seen == ["GET"] and (home / "auth.json").read_bytes() == original
    assert not (home / "auth.lock").exists()


def test_read_only_catalog_fails_closed_when_refresh_needed(tmp_path):
    home = auth_home(tmp_path, token(expires=1))
    original = (home / "auth.json").read_bytes()
    seen = []
    attempt = REQUEST_ADAPTER.validate_json(
        '{"operation":"describe_read_only","provider":"openai_oauth"}'
    )
    with pytest.raises(CapabilityError, match="refresh_required_read_only"):
        invoke(home, attempt, lambda incoming: seen.append(incoming))
    assert seen == [] and (home / "auth.json").read_bytes() == original
    assert not (home / "auth.lock").exists()


def test_function_tool_roundtrip_has_no_auxiliary_inference(tmp_path):
    seen = []
    tool = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    }
    output = [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "read_file",
            "arguments": '{"path":"a"}',
        }
    ]

    def handler(incoming):
        seen.append(incoming)
        return (
            httpx.Response(200, json=catalog())
            if incoming.method == "GET"
            else completed(output=output)
        )

    result = invoke(auth_home(tmp_path), request(tools=[tool]), handler)
    assert (
        result.finish_reason == "tool_calls"
        and result.message.tool_calls[0].id == "call-1"
    )
    assert len(seen) == 2
    body = json.loads(seen[1].content)
    assert body["tools"] == [{"type": "function", **tool["function"]}]


def test_unrequested_provider_tool_is_rejected(tmp_path):
    with pytest.raises(CapabilityError, match="unsupported_response_item"):
        invoke(
            auth_home(tmp_path),
            request(),
            lambda incoming: (
                httpx.Response(200, json=catalog())
                if incoming.method == "GET"
                else completed(output=[{"type": "web_search_call"}])
            ),
        )


def test_credential_echo_is_not_exposed(tmp_path):
    with pytest.raises(CapabilityError, match="credential_in_response"):
        invoke(
            auth_home(tmp_path),
            request(),
            lambda incoming: (
                httpx.Response(200, json=catalog())
                if incoming.method == "GET"
                else completed(text=token())
            ),
        )


def test_lock_interoperates_with_existing_hermes_auth_lock(tmp_path):
    from hermes_cli.auth import _auth_store_lock
    from hermes_cli.auth import _kernel_lock as old_kernel_lock
    from hermes_cli.isolated_oauth_store import _kernel_lock, store_lock

    path = auth_home(tmp_path) / "auth.json"
    with _auth_store_lock(target_path=path):
        with path.with_suffix(".lock").open("r+b") as handle:
            with pytest.raises(OSError):
                _kernel_lock(handle, True)
    with store_lock(path):
        with path.with_suffix(".lock").open("r+b") as handle:
            with pytest.raises(OSError):
                old_kernel_lock(handle, True)


def test_duplicate_auth_fields_rejected_before_network(tmp_path):
    home = auth_home(tmp_path)
    path = home / "auth.json"
    original = path.read_text()
    path.write_text(
        original[:-1]
        + ',"providers":'
        + json.dumps(json.loads(original)["providers"])
        + "}"
    )
    seen = []
    with pytest.raises(CapabilityError, match="invalid_json"):
        invoke(home, request(), lambda incoming: seen.append(incoming))
    assert seen == []


def test_provider_subject_format_is_preserved_exactly(tmp_path):
    home = auth_home(tmp_path, token(subject="auth0|synthetic"))
    result = invoke(
        home,
        request(principal={"account_id": "account-a", "subject": "auth0|synthetic"}),
        lambda incoming: (
            httpx.Response(200, json=catalog())
            if incoming.method == "GET"
            else completed()
        ),
    )
    assert result.principal.subject == "auth0|synthetic"


def test_concurrent_refresh_uses_one_same_principal_rotation(tmp_path):
    home = auth_home(tmp_path, token(expires=1))
    refreshed = []

    def handler(incoming):
        if incoming.url.host == "auth.openai.com":
            refreshed.append(True)
            return httpx.Response(
                200,
                json={
                    "access_token": token(nonce="fresh"),
                    "refresh_token": "fresh-refresh",
                },
            )
        if incoming.method == "GET":
            return httpx.Response(200, json=catalog())
        return completed()

    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(
            workers.map(
                lambda index: invoke(
                    home, request(attempt_id=f"call-{index}"), handler
                ),
                range(8),
            )
        )
    assert len(refreshed) == 1
    assert {result.attempt_id for result in results} == {
        f"call-{index}" for index in range(8)
    }


def test_concurrent_accounts_use_only_their_selected_home(tmp_path):
    homes = [tmp_path / "a", tmp_path / "b"]
    for index, home in enumerate(homes):
        home.mkdir()
        auth_home(home, token(account=f"account-{index}"))

    def run(index):
        account = f"account-{index % 2}"

        def handler(incoming):
            assert incoming.headers["ChatGPT-Account-Id"] == account
            assert incoming.headers["Authorization"] == "Bearer " + token(
                account=account
            )
            return (
                httpx.Response(200, json=catalog())
                if incoming.method == "GET"
                else completed()
            )

        return invoke(
            homes[index % 2],
            request(principal={"account_id": account, "subject": "user-a"}),
            handler,
        )

    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(run, range(16)))
    assert [result.principal.account_id for result in results] == [
        f"account-{index % 2}" for index in range(16)
    ]


def test_read_only_completion_never_refreshes_or_writes(tmp_path):
    home = auth_home(tmp_path)
    original = (home / "auth.json").read_bytes()
    seen = []

    def handler(incoming):
        seen.append((incoming.method, incoming.url.host))
        return (
            httpx.Response(200, json=catalog())
            if incoming.method == "GET"
            else completed()
        )

    result = invoke(home, request(operation="complete_read_only"), handler)
    assert result.message.content == "done"
    assert seen == [("GET", "chatgpt.com"), ("POST", "chatgpt.com")]
    assert (home / "auth.json").read_bytes() == original and not (
        home / "auth.lock"
    ).exists()


def test_read_only_completion_rejects_expiry_without_any_write(tmp_path):
    home = auth_home(tmp_path, token(expires=1))
    original = (home / "auth.json").read_bytes()
    seen = []
    with pytest.raises(CapabilityError, match="refresh_required_read_only"):
        invoke(
            home,
            request(operation="complete_read_only"),
            lambda incoming: seen.append(incoming),
        )
    assert seen == []
    assert (home / "auth.json").read_bytes() == original and not (
        home / "auth.lock"
    ).exists()
