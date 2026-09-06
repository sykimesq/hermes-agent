from __future__ import annotations

import json
import math
from typing import Annotated, ClassVar, Final, Literal, assert_never

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    model_validator,
)


class CapabilityError(Exception):
    pass


def _unique_fields(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    fields: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in fields:
            raise CapabilityError("invalid_json")
        fields[key] = value
    return fields


def _finite_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise CapabilityError("invalid_json")
    return value


def strict_json(raw: str | bytes) -> str | bytes:
    json.loads(
        raw,
        object_pairs_hook=_unique_fields,
        parse_constant=_finite_float,
        parse_float=_finite_float,
    )
    return raw


class StrictModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )


Identifier = Annotated[
    str, Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:@/-]+$")
]
PrincipalName = Annotated[str, Field(min_length=1, max_length=512, pattern=r"^[!-~]+$")]


class Principal(StrictModel):
    account_id: PrincipalName
    subject: PrincipalName


class FunctionCall(StrictModel):
    name: Identifier
    arguments: str


class ToolCall(StrictModel):
    id: Identifier
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(StrictModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: Identifier | None = None

    @model_validator(mode="after")
    def check_role_fields(self) -> Message:
        if self.tool_calls and self.role != "assistant":
            raise CapabilityError("invalid_request")
        if (self.role == "tool") != (self.tool_call_id is not None):
            raise CapabilityError("invalid_request")
        if self.content is None and not self.tool_calls:
            raise CapabilityError("invalid_request")
        return self


class FunctionDefinition(StrictModel):
    name: Identifier
    description: str = ""
    parameters: dict[str, JsonValue]


class Tool(StrictModel):
    type: Literal["function"]
    function: FunctionDefinition


class Describe(StrictModel):
    operation: Literal["describe"]
    provider: Literal["openai_oauth"]


class DescribeReadOnly(StrictModel):
    operation: Literal["describe_read_only"]
    provider: Literal["openai_oauth"]


class ReasoningSummary(StrictModel):
    type: Literal["summary_text"]
    text: str


class ReasoningContent(StrictModel):
    type: Literal["reasoning_text"]
    text: str


class ReasoningItem(StrictModel):
    type: Literal["reasoning"]
    id: Identifier
    summary: tuple[ReasoningSummary, ...]
    content: tuple[ReasoningContent, ...] | None = None
    encrypted_content: str = Field(min_length=1)
    status: Literal["in_progress", "completed", "incomplete"] | None = None


class ReasoningHistory(StrictModel):
    before_call_id: Identifier
    items: tuple[ReasoningItem, ...] = Field(min_length=1)


class Complete(StrictModel):
    operation: Literal["complete", "complete_read_only"]
    provider: Literal["openai_oauth"]
    principal: Principal
    model: Identifier
    reasoning_effort: Identifier
    attempt_id: Identifier
    messages: tuple[Message, ...] = Field(min_length=1)
    tools: tuple[Tool, ...] = ()
    reasoning_history: tuple[ReasoningHistory, ...] = ()


Request = Annotated[
    Describe | DescribeReadOnly | Complete, Field(discriminator="operation")
]
REQUEST_ADAPTER: Final[TypeAdapter[Describe | DescribeReadOnly | Complete]] = (
    TypeAdapter(Request)
)


class ModelCapability(StrictModel):
    model: Identifier
    reasoning_efforts: tuple[Identifier, ...]


class Description(StrictModel):
    provider: Literal["openai_oauth"] = "openai_oauth"
    principal: Principal
    models: tuple[ModelCapability, ...]


class AssistantMessage(StrictModel):
    role: Literal["assistant"] = "assistant"
    content: str
    tool_calls: tuple[ToolCall, ...] = ()


class ProviderReply(StrictModel):
    message: AssistantMessage
    reasoning_items: tuple[ReasoningItem, ...]


class Completion(StrictModel):
    provider: Literal["openai_oauth"] = "openai_oauth"
    principal: Principal
    model: str
    reasoning_effort: str
    attempt_id: str
    message: AssistantMessage
    reasoning_items: tuple[ReasoningItem, ...]
    finish_reason: Literal["stop", "tool_calls"]


def responses_payload(request: Complete) -> dict[str, JsonValue]:
    instructions: list[str] = []
    inputs: list[JsonValue] = []
    history = {entry.before_call_id: entry.items for entry in request.reasoning_history}
    if len(history) != len(request.reasoning_history):
        raise CapabilityError("duplicate_reasoning_anchor")
    calls_seen: set[str] = set()
    for message in request.messages:
        match message.role:
            case "system" | "developer":
                if inputs:
                    raise CapabilityError("instruction_order_unsupported")
                instructions.append(message.content or "")
            case "user" | "assistant":
                if message.tool_calls:
                    reasoning = history.pop(message.tool_calls[0].id, ())
                    inputs.extend(
                        item.model_dump(mode="json", exclude_none=True)
                        for item in reasoning
                    )
                if message.content:
                    inputs.append({"role": message.role, "content": message.content})
                for call in message.tool_calls:
                    if call.id in calls_seen:
                        raise CapabilityError("duplicate_tool_call")
                    calls_seen.add(call.id)
                    inputs.append({
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    })
            case "tool":
                inputs.append({
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                })
            case _:
                assert_never(message.role)
    if history:
        raise CapabilityError("missing_reasoning_anchor")
    if not inputs:
        raise CapabilityError("invalid_request")
    tool_payloads: list[JsonValue] = [
        {
            "type": "function",
            "name": tool.function.name,
            "description": tool.function.description,
            "parameters": tool.function.parameters,
        }
        for tool in request.tools
    ]
    return {
        "model": request.model,
        "reasoning": {"effort": request.reasoning_effort},
        "instructions": "\n\n".join(instructions),
        "input": inputs,
        "tools": tool_payloads,
        "store": False,
        "stream": True,
        "include": ["reasoning.encrypted_content"],
    }
