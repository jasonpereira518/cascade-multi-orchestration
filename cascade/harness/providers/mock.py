"""Scriptable mock model source: deterministic, offline, $0. The default.

Built on the Agents SDK's own ``ScriptedModel`` (agents.testing). Each role gets one model that
lives for the whole source, so a plan can span repeated invocations (e.g. repair rounds). A plan is
either a sequence of steps consumed in order, or a function of ``(ModelCall, call_index)``. With no
plan, or once a sequence is exhausted (unless ``strict``), the model answers with a minimal valid
instance of the requested output schema and no tool calls.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any

from agents import ModelSettings
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from agents.models.interface import Model, ModelTracing
from agents.testing import ModelCall, ModelStep, ScriptedModel, assistant_message, function_call
from agents.testing.model import ModelStepResult
from agents.tool import Tool
from agents.usage import Usage as SdkUsage
from openai.types.responses.response_prompt_param import ResponsePromptParam
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from pydantic import BaseModel

from cascade.errors import ProviderError
from cascade.harness.providers.base import ModelSource

PlanFn = Callable[[ModelCall, int], ModelStepResult]
Plan = Sequence[ModelStep] | PlanFn


class MockPlanExhaustedError(ProviderError):
    """A strict plan ran out of steps."""


def sdk_usage(input_tokens: int, output_tokens: int, reasoning_tokens: int = 0) -> SdkUsage:
    """SDK usage as a real provider reports it: output_tokens INCLUDES reasoning tokens."""
    total_out = output_tokens + reasoning_tokens
    return SdkUsage(
        requests=1,
        input_tokens=input_tokens,
        output_tokens=total_out,
        total_tokens=input_tokens + total_out,
        input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning_tokens),
    )


def tool_call_step(
    name: str, arguments: Mapping[str, Any], *, call_id: str, input_tokens: int = 200,
    output_tokens: int = 40, reasoning_tokens: int = 0,
) -> ModelStep:  # fmt: skip
    return ModelStep(
        output=[function_call(name, arguments, call_id=call_id)],
        usage=sdk_usage(input_tokens, output_tokens, reasoning_tokens),
    )


def final_step(
    output: BaseModel | str, *, input_tokens: int = 300, output_tokens: int = 60,
    reasoning_tokens: int = 0,
) -> ModelStep:  # fmt: skip
    text = output if isinstance(output, str) else output.model_dump_json()
    return ModelStep(
        output=[assistant_message(text)],
        usage=sdk_usage(input_tokens, output_tokens, reasoning_tokens),
    )


def minimal_instance(schema: Mapping[str, Any], root: Mapping[str, Any] | None = None) -> Any:
    """The smallest value that satisfies a (pydantic-style) JSON schema."""
    root = root or schema
    if "$ref" in schema:
        return minimal_instance(root["$defs"][str(schema["$ref"]).rsplit("/", 1)[-1]], root)
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        options = [o for o in schema["anyOf"] if o.get("type") != "null"] or schema["anyOf"]
        return minimal_instance(options[0], root)
    kind = schema.get("type")
    if kind == "object":
        return {k: minimal_instance(v, root) for k, v in schema.get("properties", {}).items()}
    if kind == "array":
        return [] if schema.get("minItems", 0) == 0 else [minimal_instance(schema["items"], root)]
    if kind == "string":
        return "x" * int(schema.get("minLength", 0))
    if kind in ("integer", "number"):
        return schema.get("minimum", 0)
    if kind == "boolean":
        return False
    return None


class MockModel(Model):
    def __init__(
        self, role: str, plan: Plan | None, *, strict: bool, default_input_tokens: int,
        default_output_tokens: int,
    ) -> None:  # fmt: skip
        self.role = role
        self._plan = plan
        self._strict = strict
        self._usage = (default_input_tokens, default_output_tokens)
        self._index = 0
        self._scripted = ScriptedModel()

    @property
    def calls(self) -> tuple[ModelCall, ...]:
        return self._scripted.calls

    def _default_step(self, call: ModelCall) -> ModelStep:
        schema = call.output_schema
        if schema is None or schema.is_plain_text():
            text = "ok"
        else:
            text = json.dumps(minimal_instance(schema.json_schema()))
        return ModelStep(output=[assistant_message(text)], usage=sdk_usage(*self._usage))

    def _resolve(self, call: ModelCall) -> ModelStepResult:
        index = self._index
        self._index += 1
        plan = self._plan
        if callable(plan):
            return plan(call, index)
        if plan is not None and index < len(plan):
            return plan[index]
        if plan is not None and self._strict:
            raise MockPlanExhaustedError(f"{self.role}: plan exhausted at call {index}")
        return self._default_step(call)

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> ModelResponse:
        self._scripted.enqueue(ModelStep.respond(self._resolve))
        return await self._scripted.get_response(
            system_instructions, input, model_settings, tools, output_schema, handoffs, tracing,
            previous_response_id=previous_response_id, conversation_id=conversation_id,
            prompt=prompt,
        )  # fmt: skip

    def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[TResponseStreamEvent]:
        raise NotImplementedError("MockModel supports non-streaming calls only")


class MockModelSource(ModelSource):
    name = "mock"
    is_real = False

    def __init__(
        self, plans: Mapping[str, Plan] | None = None, *, strict: bool = False,
        default_input_tokens: int = 200, default_output_tokens: int = 50,
    ) -> None:  # fmt: skip
        self._plans = dict(plans or {})
        self._strict = strict
        self._defaults = (default_input_tokens, default_output_tokens)
        self.models: dict[str, MockModel] = {}

    def model_for(self, model_name: str, role: str) -> MockModel:
        if role not in self.models:
            self.models[role] = MockModel(
                role, self._plans.get(role), strict=self._strict,
                default_input_tokens=self._defaults[0], default_output_tokens=self._defaults[1],
            )  # fmt: skip
        return self.models[role]

    def calls(self, role: str) -> tuple[ModelCall, ...]:
        model = self.models.get(role)
        return model.calls if model else ()
