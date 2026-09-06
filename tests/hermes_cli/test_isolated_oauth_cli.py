import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.hermes_cli.conftest_isolated_oauth import (
    auth_home,
    catalog,
    completed,
    request,
    token,
)


def test_actual_cli_rejects_invalid_provider_without_secret_output(tmp_path):
    script = Path(__file__).resolve().parents[2] / "hermes_cli" / "isolated_oauth.py"
    process = subprocess.run(
        [sys.executable, "-I", str(script), "--home", str(tmp_path)],
        input=json.dumps({
            "operation": "describe",
            "provider": "openai",
            "token": "synthetic-secret",
        }),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 1
    assert json.loads(process.stdout) == {"error": "isolated_oauth_failed"}
    assert process.stderr == "" and "synthetic-secret" not in process.stdout


def test_real_imports_do_not_initialize_general_runtime(tmp_path):
    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; sys.path.insert(0, " + repr(str(root)) + "); "
        "import hermes_cli.isolated_oauth; "
        "assert not any(m.startswith(('agent.', 'run_agent', 'hermes_cli.config', "
        "'hermes_cli.main', 'hermes_cli.plugins', 'model_tools')) for m in sys.modules); "
        "assert 'hermes_cli.auth' not in sys.modules"
    )
    process = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=30
    )
    assert process.returncode == 0, process.stderr


def test_actual_cli_completes_with_synthetic_transport_only(tmp_path):
    home = auth_home(tmp_path)
    root = Path(__file__).resolve().parents[2]
    script = root / "hermes_cli" / "isolated_oauth.py"
    harness = tmp_path / "cli_harness.py"
    harness.write_text(
        "\n".join([
            "import httpx, json, runpy, sys",
            "calls = []",
            "def handler(request):",
            "    calls.append(request)",
            "    assert request.url.host == 'chatgpt.com'",
            "    assert request.headers['ChatGPT-Account-Id'] == 'account-a'",
            "    if request.method == 'GET':",
            "        return httpx.Response(200, json=" + repr(catalog()) + ")",
            "    payload = json.loads(request.content)",
            "    assert payload['model'] == 'model-a' and payload['reasoning'] == {'effort': 'high'}",
            "    return httpx.Response(200, text=" + repr(completed().text) + ")",
            "def transport(**kwargs):",
            "    assert kwargs == {'retries': 0, 'trust_env': False}",
            "    return httpx.MockTransport(handler)",
            "httpx.HTTPTransport = transport",
            "sys.argv = [" + repr(str(script)) + ", '--home', " + repr(str(home)) + "]",
            "try:",
            "    runpy.run_path(sys.argv[0], run_name='__main__')",
            "finally:",
            "    assert len(calls) == 2",
            "    assert not any(m.startswith(('agent.', 'hermes_cli.config', 'hermes_cli.main', 'run_agent')) for m in sys.modules)",
        ])
    )
    environment = dict(
        os.environ,
        OPENAI_API_KEY="forbidden-key",
        HERMES_CODEX_BASE_URL="https://example.invalid",
        HTTPS_PROXY="https://example.invalid",
        CODEX_HOME="Z:/forbidden",
        HERMES_HOME="Z:/forbidden",
    )
    process = subprocess.run(
        [sys.executable, "-I", str(harness)],
        input=request().model_dump_json(),
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["message"]["content"] == "done" and result["model"] == "model-a"
    assert (
        process.stderr == ""
        and token() not in process.stdout
        and "forbidden" not in process.stdout
    )


@pytest.mark.parametrize(
    "payload",
    [
        '{"operation":"describe","provider":"openai_oauth","provider":"openai_oauth"}',
        '{"operation":"describe","provider":"openai_oauth","extra":NaN}',
        '{"operation":"describe","provider":"openai_oauth","extra":1e999}',
    ],
)
def test_actual_cli_rejects_ambiguous_or_nonfinite_json(tmp_path, payload):
    script = Path(__file__).resolve().parents[2] / "hermes_cli" / "isolated_oauth.py"
    process = subprocess.run(
        [sys.executable, "-I", str(script), "--home", str(tmp_path)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 1 and json.loads(process.stdout) == {
        "error": "invalid_json"
    }
