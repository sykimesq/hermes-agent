import base64
import json

import httpx

from hermes_cli.isolated_oauth import execute
from hermes_cli.isolated_oauth_schema import REQUEST_ADAPTER
from hermes_cli.isolated_oauth_wire import OpenAIWire


def token(account="account-a", subject="user-a", expires=4102444800, nonce="first"):
    claims = {
        "sub": subject,
        "exp": expires,
        "nonce": nonce,
        "https://api.openai.com/auth": {"chatgpt_account_id": account},
    }
    return (
        "synthetic."
        + base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
        + ".signature"
    )


def auth_home(tmp_path, access=None):
    path = tmp_path / "auth.json"
    path.write_text(
        json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "auth_mode": "chatgpt",
                    "tokens": {
                        "access_token": access or token(),
                        "refresh_token": "synthetic-refresh",
                    },
                },
                "unrelated": {"preserve": "unchanged"},
            },
            "credential_pool": {"openai-codex": [{"access_token": "forbidden-pool"}]},
        })
    )
    return tmp_path


def request(**updates):
    raw = {
        "operation": "complete",
        "provider": "openai_oauth",
        "model": "model-a",
        "reasoning_effort": "high",
        "attempt_id": "attempt-a",
        "principal": {"account_id": "account-a", "subject": "user-a"},
        "messages": [{"role": "user", "content": "test"}],
    }
    raw.update(updates)
    return REQUEST_ADAPTER.validate_json(json.dumps(raw), strict=True)


def catalog():
    return {
        "models": [
            {
                "slug": "model-a",
                "visibility": "list",
                "supported_reasoning_levels": [{"effort": "high"}],
            },
            {
                "slug": "model-b",
                "visibility": "list",
                "supported_reasoning_levels": [{"effort": "xhigh"}],
            },
        ]
    }


def completed(model="model-a", text="done", output=None):
    data = {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "model": model,
            "output": output
            if output is not None
            else [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
        },
    }
    return httpx.Response(200, text="data: " + json.dumps(data) + "\n\n")


def invoke(home, attempt, handler):
    with httpx.Client(
        transport=httpx.MockTransport(handler), trust_env=False, follow_redirects=False
    ) as client:
        return execute(home, attempt, OpenAIWire(client))
