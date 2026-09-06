import json

import httpx
import pytest
from pydantic import ValidationError

from hermes_cli.isolated_oauth_schema import CapabilityError, responses_payload
from tests.hermes_cli.conftest_isolated_oauth import (
    auth_home,
    catalog,
    completed,
    invoke,
    request,
    token,
)


def reason():
    return {
        "type": "reasoning",
        "id": "rs-1",
        "summary": [],
        "encrypted_content": "opaque-reasoning",
    }


def tool():
    return {
        "type": "function",
        "function": {"name": "read_file", "parameters": {"type": "object"}},
    }


def history_message():
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ],
    }


def test_reasoning_state_is_returned_for_function_continuation(tmp_path):
    seen = []

    def handler(incoming):
        seen.append(incoming)
        if incoming.method == "GET":
            return httpx.Response(200, json=catalog())
        return completed(
            output=[
                reason(),
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "read_file",
                    "arguments": "{}",
                },
            ]
        )

    result = invoke(auth_home(tmp_path), request(tools=[tool()]), handler)
    assert result.reasoning_items[0].encrypted_content == "opaque-reasoning"
    assert json.loads(seen[1].content)["include"] == ["reasoning.encrypted_content"]


@pytest.mark.parametrize(
    "content", [None, [{"type": "reasoning_text", "text": "synthetic reasoning"}]]
)
def test_optional_reasoning_content_is_preserved(tmp_path, content):
    item = {**reason(), "content": content}

    def handler(incoming):
        if incoming.method == "GET":
            return httpx.Response(200, json=catalog())
        return completed(
            output=[
                item,
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "read_file",
                    "arguments": "{}",
                },
            ]
        )

    result = invoke(auth_home(tmp_path), request(tools=[tool()]), handler)
    returned = json.loads(result.model_dump_json())["reasoning_items"][0]
    assert returned["content"] == content
    attempt = request(
        messages=[history_message()],
        reasoning_history=[{"before_call_id": "call-1", "items": [returned]}],
    )
    replay = responses_payload(attempt)["input"][0]
    assert replay.get("content") == content


def test_reasoning_state_is_inserted_at_exact_function_anchor():
    attempt = request(
        messages=[
            {"role": "user", "content": "test"},
            history_message(),
            {"role": "tool", "tool_call_id": "call-1", "content": "data"},
        ],
        reasoning_history=[{"before_call_id": "call-1", "items": [reason()]}],
    )
    body = responses_payload(attempt)
    assert body["input"] == [
        {"role": "user", "content": "test"},
        reason(),
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "read_file",
            "arguments": "{}",
        },
        {"type": "function_call_output", "call_id": "call-1", "output": "data"},
    ]


@pytest.mark.parametrize(
    "history,error",
    [
        (
            [{"before_call_id": "missing", "items": [reason()]}],
            "missing_reasoning_anchor",
        ),
        (
            [{"before_call_id": "call-1", "items": [reason()]}] * 2,
            "duplicate_reasoning_anchor",
        ),
    ],
)
def test_reasoning_anchor_must_be_unique_and_present(history, error):
    attempt = request(messages=[history_message()], reasoning_history=history)
    with pytest.raises(CapabilityError, match=error):
        responses_payload(attempt)


def test_duplicate_function_call_cannot_rebind_reasoning():
    attempt = request(messages=[history_message(), history_message()])
    with pytest.raises(CapabilityError, match="duplicate_tool_call"):
        responses_payload(attempt)


@pytest.mark.parametrize(
    "field", ["access_token", "refresh_token", "provider", "model"]
)
def test_reasoning_objects_cannot_carry_unrelated_fields(field):
    with pytest.raises(ValidationError):
        request(
            reasoning_history=[
                {"before_call_id": "call-1", "items": [{**reason(), field: "value"}]}
            ]
        )


def test_ordinary_text_response_has_no_reasoning_history(tmp_path):
    output = [
        reason(),
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "done"}],
        },
    ]
    result = invoke(
        auth_home(tmp_path),
        request(),
        lambda incoming: (
            httpx.Response(200, json=catalog())
            if incoming.method == "GET"
            else completed(output=output)
        ),
    )
    assert result.reasoning_items == ()


@pytest.mark.parametrize("echo", ["access", "refresh", "old-access", "old-refresh"])
def test_selected_tokens_before_and_after_refresh_cannot_escape(tmp_path, echo):
    old = token(expires=1)
    fresh = token(nonce="new")
    home = auth_home(tmp_path, old)
    values = {
        "access": fresh,
        "refresh": "new-refresh",
        "old-access": old,
        "old-refresh": "synthetic-refresh",
    }

    def handler(incoming):
        if incoming.url.host == "auth.openai.com":
            return httpx.Response(
                200, json={"access_token": fresh, "refresh_token": "new-refresh"}
            )
        if incoming.method == "GET":
            return httpx.Response(200, json=catalog())
        return completed(text=values[echo])

    with pytest.raises(CapabilityError, match="credential_in_response"):
        invoke(home, request(), handler)


def test_refresh_token_cannot_escape_in_encrypted_reasoning(tmp_path):
    output = [
        {**reason(), "encrypted_content": "synthetic-refresh"},
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "read_file",
            "arguments": "{}",
        },
    ]
    with pytest.raises(CapabilityError, match="credential_in_response"):
        invoke(
            auth_home(tmp_path),
            request(tools=[tool()]),
            lambda incoming: (
                httpx.Response(200, json=catalog())
                if incoming.method == "GET"
                else completed(output=output)
            ),
        )


@pytest.mark.parametrize(
    "refresh",
    [
        'synthetic-"quoted"-refresh',
        "synthetic-유니코드-refresh",
        "synthetic-\\slash-refresh",
    ],
)
def test_json_escaping_cannot_hide_selected_refresh_token(tmp_path, refresh):
    home = auth_home(tmp_path)
    path = home / "auth.json"
    state = json.loads(path.read_text())
    state["providers"]["openai-codex"]["tokens"]["refresh_token"] = refresh
    path.write_text(json.dumps(state))
    with pytest.raises(CapabilityError, match="credential_in_response"):
        invoke(
            home,
            request(),
            lambda incoming: (
                httpx.Response(200, json=catalog())
                if incoming.method == "GET"
                else completed(text=refresh)
            ),
        )
