import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from pydantic import ValidationError

from hermes_cli.isolated_oauth_schema import REQUEST_ADAPTER, CapabilityError
from tests.hermes_cli.conftest_isolated_oauth import (
    auth_home,
    catalog,
    completed,
    invoke,
    request,
    token,
)


def test_exact_binding_when_inference_is_sent(tmp_path):
    home = auth_home(tmp_path)
    seen = []

    def handler(incoming):
        seen.append(incoming)
        return (
            httpx.Response(200, json=catalog())
            if incoming.method == "GET"
            else completed()
        )

    result = invoke(home, request(), handler)
    assert result.model == "model-a" and result.reasoning_effort == "high"
    assert result.attempt_id == "attempt-a" and result.principal.subject == "user-a"
    assert [(r.method, r.url.host, r.url.path) for r in seen] == [
        ("GET", "chatgpt.com", "/backend-api/codex/models"),
        ("POST", "chatgpt.com", "/backend-api/codex/responses"),
    ]
    wire = json.loads(seen[1].content)
    assert wire["model"] == "model-a" and wire["reasoning"] == {"effort": "high"}
    assert wire["input"] == [{"role": "user", "content": "test"}]
    assert wire["store"] is False and wire["stream"] is True
    assert all(r.headers["ChatGPT-Account-Id"] == "account-a" for r in seen)
    assert all(r.headers["Authorization"] == "Bearer " + token() for r in seen)


@pytest.mark.parametrize(
    "updates",
    [
        {"provider": "openai"},
        {"provider": "ollama_cloud"},
        {"api_key": "forbidden"},
        {"base_url": "https://example.invalid"},
        {"model": ""},
        {"reasoning_effort": None},
        {"principal": {"account_id": "account-a"}},
        {"reasoning_effort": "high\n"},
    ],
)
def test_invalid_contract_when_parsing(updates):
    with pytest.raises((ValidationError, CapabilityError)):
        request(**updates)


@pytest.mark.parametrize(
    "updates,error",
    [
        ({"model": "model-not-in-catalog"}, "unsupported_model"),
        ({"reasoning_effort": "xhigh"}, "unsupported_reasoning"),
        ({"reasoning_effort": "none"}, "unsupported_reasoning"),
    ],
)
def test_unsupported_selection_when_catalog_disagrees(tmp_path, updates, error):
    seen = []

    def handler(incoming):
        seen.append(incoming.method)
        return httpx.Response(200, json=catalog())

    with pytest.raises(CapabilityError, match=error):
        invoke(auth_home(tmp_path), request(**updates), handler)
    assert seen == ["GET"]


@pytest.mark.parametrize("status", [301, 400, 401, 403, 429, 500])
def test_fail_closed_without_retry_when_provider_fails(tmp_path, status):
    seen = []

    def handler(incoming):
        seen.append(incoming.method)
        return httpx.Response(
            status,
            headers={"Location": "https://example.invalid"},
            text="synthetic-secret",
        )

    with pytest.raises(CapabilityError, match="catalog_failed"):
        invoke(auth_home(tmp_path), request(), handler)
    assert seen == ["GET"]


def test_no_alternate_store_when_singleton_is_missing(tmp_path, monkeypatch):
    home = auth_home(tmp_path)
    store = json.loads((home / "auth.json").read_text())
    del store["providers"]["openai-codex"]
    (home / "auth.json").write_text(json.dumps(store))
    monkeypatch.setenv("OPENAI_API_KEY", "forbidden-env")
    monkeypatch.setenv("CODEX_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    seen = []
    with pytest.raises(ValidationError):
        invoke(home, request(), lambda incoming: seen.append(incoming))
    assert seen == []


@pytest.mark.parametrize(
    "principal",
    [
        {"account_id": "account-b", "subject": "user-a"},
        {"account_id": "account-a", "subject": "user-b"},
    ],
)
def test_cross_principal_rejected_before_network(tmp_path, principal):
    seen = []
    with pytest.raises(CapabilityError, match="principal_mismatch"):
        invoke(
            auth_home(tmp_path),
            request(principal=principal),
            lambda incoming: seen.append(incoming),
        )
    assert seen == []


def test_same_principal_refresh_is_owned_and_persisted_by_hermes(tmp_path):
    home = auth_home(tmp_path, token(expires=1))
    seen = []
    fresh = token(nonce="new-grant")

    def handler(incoming):
        seen.append(incoming)
        if incoming.url.host == "auth.openai.com":
            return httpx.Response(
                200, json={"access_token": fresh, "refresh_token": "new-refresh"}
            )
        if incoming.method == "GET":
            return httpx.Response(200, json=catalog())
        return completed()

    result = invoke(home, request(), handler)
    assert result.message.content == "done"
    assert [r.url.host for r in seen] == [
        "auth.openai.com",
        "chatgpt.com",
        "chatgpt.com",
    ]
    assert all(r.headers["Authorization"] == "Bearer " + fresh for r in seen[1:])
    stored = json.loads((home / "auth.json").read_text())
    assert (
        stored["providers"]["openai-codex"]["tokens"]["refresh_token"] == "new-refresh"
    )
    assert stored["providers"]["unrelated"] == {"preserve": "unchanged"}
    assert stored["credential_pool"]["openai-codex"] == [
        {"access_token": "forbidden-pool"}
    ]
    assert (
        "access_token" not in result.model_dump_json()
        and "refresh_token" not in result.model_dump_json()
    )


def test_new_same_principal_grant_is_accepted_when_store_changes(tmp_path):
    home = auth_home(tmp_path, token(nonce="new-grant"))
    result = invoke(
        home,
        request(),
        lambda incoming: (
            httpx.Response(200, json=catalog())
            if incoming.method == "GET"
            else completed()
        ),
    )
    assert result.principal.subject == "user-a"


def test_cross_principal_refresh_rejected_without_persistence(tmp_path):
    home = auth_home(tmp_path, token(expires=1))
    original = (home / "auth.json").read_bytes()
    seen = []

    def handler(incoming):
        seen.append(incoming)
        return httpx.Response(
            200,
            json={"access_token": token(account="account-b"), "refresh_token": "other"},
        )

    with pytest.raises(CapabilityError, match="refresh_principal_or_expiry_mismatch"):
        invoke(home, request(), handler)
    assert len(seen) == 1 and seen[0].url.host == "auth.openai.com"
    assert (home / "auth.json").read_bytes() == original


def test_failed_refresh_never_recovers_from_another_store(tmp_path):
    home = auth_home(tmp_path, token(expires=1))
    seen = []

    def handler(incoming):
        seen.append(incoming)
        return httpx.Response(401, json={"error": "invalid_grant"})

    with pytest.raises(CapabilityError, match="oauth_refresh_failed"):
        invoke(home, request(), handler)
    assert len(seen) == 1


def test_catalog_reports_only_actual_raw_entries(tmp_path):
    describe = REQUEST_ADAPTER.validate_json(
        '{"operation":"describe","provider":"openai_oauth"}'
    )
    result = invoke(
        auth_home(tmp_path),
        describe,
        lambda incoming: httpx.Response(200, json=catalog()),
    )
    assert [(m.model, m.reasoning_efforts) for m in result.models] == [
        ("model-a", ("high",)),
        ("model-b", ("xhigh",)),
    ]


@pytest.mark.parametrize(
    "catalog_data",
    [
        {"models": []},
        {"models": [{"slug": "model-a", "visibility": "list"}]},
        {"models": catalog()["models"] * 2},
    ],
)
def test_incomplete_catalog_cannot_supply_guessed_models(tmp_path, catalog_data):
    with pytest.raises((CapabilityError, ValidationError)):
        invoke(
            auth_home(tmp_path),
            request(),
            lambda incoming: httpx.Response(200, json=catalog_data),
        )


def test_response_model_substitution_is_rejected(tmp_path):
    with pytest.raises(CapabilityError, match="response_model_mismatch"):
        invoke(
            auth_home(tmp_path),
            request(),
            lambda incoming: (
                httpx.Response(200, json=catalog())
                if incoming.method == "GET"
                else completed(model="model-b")
            ),
        )


def test_concurrent_roles_cannot_cross_bind(tmp_path):
    home = auth_home(tmp_path)

    def run(index):
        model, effort = ("model-a", "high") if index % 2 == 0 else ("model-b", "xhigh")

        def handler(incoming):
            if incoming.method == "GET":
                return httpx.Response(200, json=catalog())
            payload = json.loads(incoming.content)
            assert payload["model"] == model and payload["reasoning"] == {
                "effort": effort
            }
            assert incoming.headers["Authorization"] == "Bearer " + token()
            return completed(model=model, text=str(index))

        return invoke(
            home,
            request(
                model=model, reasoning_effort=effort, attempt_id=f"attempt-{index}"
            ),
            handler,
        )

    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(run, range(16)))
    assert [(r.attempt_id, r.message.content) for r in results] == [
        (f"attempt-{i}", str(i)) for i in range(16)
    ]
