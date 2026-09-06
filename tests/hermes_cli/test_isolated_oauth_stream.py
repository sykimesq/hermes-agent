import json

import httpx
import pytest

from hermes_cli.isolated_oauth_schema import CapabilityError
from tests.hermes_cli.conftest_isolated_oauth import auth_home, catalog, invoke, request


def stream_events(events):
    return httpx.Response(
        200, text="".join("data: " + json.dumps(event) + "\n\n" for event in events)
    )


def message_event(index=0):
    return {
        "type": "response.output_item.done",
        "output_index": index,
        "item": {
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "done"}],
        },
    }


def terminal(output):
    return {
        "type": "response.completed",
        "response": {"status": "completed", "model": "model-a", "output": output},
    }


@pytest.mark.parametrize("output", [[], None])
def test_done_items_survive_empty_terminal_output(tmp_path, output):
    def handler(incoming):
        return (
            httpx.Response(200, json=catalog())
            if incoming.method == "GET"
            else stream_events([message_event(), terminal(output)])
        )

    result = invoke(auth_home(tmp_path), request(), handler)
    assert result.message.content == "done"


def test_done_items_require_successful_terminal_frame(tmp_path):
    with pytest.raises(CapabilityError, match="inference_incomplete"):
        invoke(
            auth_home(tmp_path),
            request(),
            lambda incoming: (
                httpx.Response(200, json=catalog())
                if incoming.method == "GET"
                else stream_events([message_event()])
            ),
        )


@pytest.mark.parametrize(
    "events",
    [
        [message_event(), message_event(), terminal([])],
        [message_event(1), terminal([])],
        [terminal([]), message_event()],
    ],
)
def test_ambiguous_done_items_are_rejected(tmp_path, events):
    with pytest.raises(CapabilityError, match="ambiguous_response"):
        invoke(
            auth_home(tmp_path),
            request(),
            lambda incoming: (
                httpx.Response(200, json=catalog())
                if incoming.method == "GET"
                else stream_events(events)
            ),
        )
