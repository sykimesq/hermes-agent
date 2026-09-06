from __future__ import annotations

from typing import ClassVar, Final

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from hermes_cli.auth_constants import (
    CODEX_OAUTH_CLIENT_ID,
    CODEX_OAUTH_TOKEN_URL,
    DEFAULT_CODEX_BASE_URL,
)
from hermes_cli.isolated_oauth_schema import (
    AssistantMessage,
    CapabilityError,
    Complete,
    ModelCapability,
    ProviderReply,
    ReasoningItem,
    ToolCall,
    responses_payload,
    strict_json,
)

MAX_BODY: Final = 8 * 1024 * 1024
JSON_ADAPTER: Final = TypeAdapter(dict[str, JsonValue])
INDEX_ADAPTER: Final = TypeAdapter(int)


class CatalogReasoning(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore", strict=True, frozen=True
    )
    effort: str = Field(min_length=1)


class CatalogModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore", strict=True, frozen=True
    )
    slug: str = Field(min_length=1)
    supported_reasoning_levels: list[CatalogReasoning]
    visibility: str


class Catalog(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore", strict=True, frozen=True
    )
    models: list[CatalogModel]


def bounded_body(response: httpx.Response) -> bytes:
    body = bytearray()
    for chunk in response.iter_bytes():
        body.extend(chunk)
        if len(body) > MAX_BODY:
            raise CapabilityError("response_too_large")
    return bytes(body)


class OpenAIWire:
    def __init__(self, client: httpx.Client) -> None:
        self.client: httpx.Client = client

    def refresh(self, refresh_token: str) -> dict[str, JsonValue]:
        with self.client.stream(
            "POST",
            CODEX_OAUTH_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CODEX_OAUTH_CLIENT_ID,
            },
        ) as response:
            if response.status_code != 200:
                raise CapabilityError("oauth_refresh_failed")
            return JSON_ADAPTER.validate_json(
                strict_json(bounded_body(response)), strict=True
            )

    def catalog(self, headers: dict[str, str]) -> tuple[ModelCapability, ...]:
        with self.client.stream(
            "GET",
            DEFAULT_CODEX_BASE_URL + "/models?client_version=1.0.0",
            headers=headers,
        ) as response:
            if response.status_code != 200:
                raise CapabilityError("catalog_failed")
            catalog = Catalog.model_validate_json(strict_json(bounded_body(response)))
        result: list[ModelCapability] = []
        seen: set[str] = set()
        for entry in catalog.models:
            if entry.slug in seen:
                raise CapabilityError("ambiguous_catalog")
            seen.add(entry.slug)
            if entry.visibility not in {"list", "show"}:
                continue
            efforts = tuple(level.effort for level in entry.supported_reasoning_levels)
            if len(set(efforts)) != len(efforts):
                raise CapabilityError("ambiguous_catalog")
            result.append(ModelCapability(model=entry.slug, reasoning_efforts=efforts))
        if not result:
            raise CapabilityError("catalog_empty")
        return tuple(result)

    def complete(self, request: Complete, headers: dict[str, str]) -> ProviderReply:
        with self.client.stream(
            "POST",
            DEFAULT_CODEX_BASE_URL + "/responses",
            headers=headers,
            json=responses_payload(request),
        ) as response:
            if response.status_code != 200:
                raise CapabilityError("inference_failed")
            body = bounded_body(response).decode("utf-8")
        completed: dict[str, JsonValue] | None = None
        done_items: dict[int, dict[str, JsonValue]] = {}
        done_ids: set[str] = set()
        for block in body.replace("\r\n", "\n").split("\n\n"):
            data = "\n".join(
                line[5:].lstrip()
                for line in block.splitlines()
                if line.startswith("data:")
            )
            if not data or data == "[DONE]":
                continue
            event = JSON_ADAPTER.validate_json(strict_json(data), strict=True)
            if event.get("type") == "response.output_item.done":
                index = INDEX_ADAPTER.validate_python(
                    event.get("output_index"), strict=True
                )
                item = JSON_ADAPTER.validate_python(event.get("item"), strict=True)
                item_id = item.get("id")
                if (
                    completed is not None
                    or index < 0
                    or index in done_items
                    or not isinstance(item_id, str)
                    or not item_id
                    or item_id in done_ids
                ):
                    raise CapabilityError("ambiguous_response")
                done_items[index] = item
                done_ids.add(item_id)
            if event.get("type") in {"error", "response.failed", "response.incomplete"}:
                raise CapabilityError("inference_failed")
            if event.get("type") == "response.completed":
                if completed is not None:
                    raise CapabilityError("ambiguous_response")
                completed = JSON_ADAPTER.validate_python(
                    event.get("response"), strict=True
                )
        if completed is None or completed.get("status") != "completed":
            raise CapabilityError("inference_incomplete")
        if completed.get("model") != request.model:
            raise CapabilityError("response_model_mismatch")
        output = completed.get("output")
        if done_items:
            if set(done_items) != set(range(len(done_items))):
                raise CapabilityError("ambiguous_response")
            assembled: list[JsonValue] = [
                done_items[index] for index in range(len(done_items))
            ]
            if output is not None and output != [] and output != assembled:
                raise CapabilityError("ambiguous_response")
            output = assembled
        if not isinstance(output, list):
            raise CapabilityError("invalid_response")
        text: list[str] = []
        calls: list[ToolCall] = []
        reasoning: list[ReasoningItem] = []
        for raw in output:
            item = JSON_ADAPTER.validate_python(raw, strict=True)
            kind = item.get("type")
            if kind == "reasoning":
                reasoning.append(
                    ReasoningItem.model_validate_json(JSON_ADAPTER.dump_json(item))
                )
                continue
            if kind == "function_call":
                call = ToolCall.model_validate({
                    "id": item.get("call_id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "arguments": item.get("arguments"),
                    },
                })
                if call.function.name not in {
                    tool.function.name for tool in request.tools
                }:
                    raise CapabilityError("unrequested_tool")
                calls.append(call)
                continue
            if kind != "message" or item.get("role") != "assistant":
                raise CapabilityError("unsupported_response_item")
            contents = item.get("content")
            if not isinstance(contents, list):
                raise CapabilityError("invalid_response")
            for content in contents:
                part = JSON_ADAPTER.validate_python(content, strict=True)
                value = part.get("text")
                if part.get("type") != "output_text" or not isinstance(value, str):
                    raise CapabilityError("unsupported_response_content")
                text.append(value)
        if not text and not calls:
            raise CapabilityError("empty_response")
        return ProviderReply(
            message=AssistantMessage(content="".join(text), tool_calls=tuple(calls)),
            reasoning_items=tuple(reasoning) if calls else (),
        )
