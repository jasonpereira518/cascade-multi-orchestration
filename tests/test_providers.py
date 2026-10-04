from __future__ import annotations

import asyncio
from typing import Any

import pytest
from agents import Agent, ModelSettings, RunConfig, Runner
from agents.models.interface import ModelTracing
from agents.models.openai_responses import OpenAIResponsesModel
from agents.testing import ModelCall, ModelStep
from pydantic import BaseModel

from cascade.agents.schemas import (
    LocalizationOutput,
    PatchOutput,
    ReproOutput,
    ReviewOutput,
    TriageOutput,
)
from cascade.errors import SpendNotAllowedError
from cascade.harness.providers import get_model_source, resolve_provider_name
from cascade.harness.providers.mock import (
    MockModelSource,
    MockPlanExhaustedError,
    final_step,
    minimal_instance,
    sdk_usage,
    tool_call_step,
)
from cascade.harness.providers.openai import API_KEY_ENV, MissingApiKeyError, OpenAIModelSource


def one_call(model: Any, text: str = "hi") -> Any:
    return asyncio.run(
        model.get_response(
            "sys", text, ModelSettings(max_tokens=50), [], None, [], ModelTracing.DISABLED,
            previous_response_id=None, conversation_id=None, prompt=None,
        )
    )  # fmt: skip


# --- the mock source -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "output_type", [TriageOutput, ReproOutput, LocalizationOutput, PatchOutput, ReviewOutput]
)
def test_the_default_mock_answers_every_role_schema_with_a_valid_instance(
    output_type: type[BaseModel],
) -> None:
    source = MockModelSource()
    agent = Agent(
        name="x", instructions="x", model=source.model_for("m", "x"), output_type=output_type
    )
    result = Runner.run_sync(agent, "go", run_config=RunConfig(tracing_disabled=True))
    assert isinstance(result.final_output, output_type)


def test_minimal_instance_covers_refs_enums_optionals_and_arrays() -> None:
    schema = {
        "$defs": {
            "Item": {"type": "object", "properties": {"n": {"type": "integer", "minimum": 3}}}
        },
        "type": "object",
        "properties": {
            "kind": {"enum": ["a", "b"]},
            "fixed": {"const": "yes"},
            "maybe": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "items": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/Item"}},
            "none": {"type": "array"},
            "flag": {"type": "boolean"},
        },
    }
    assert minimal_instance(schema) == {
        "kind": "a", "fixed": "yes", "maybe": "", "items": [{"n": 3}], "none": [], "flag": False,
    }  # fmt: skip


def test_plans_are_consumed_in_order_then_fall_back_to_the_default() -> None:
    source = MockModelSource({"r": [final_step("first"), final_step("second")]})
    model = source.model_for("m", "r")
    texts = [one_call(model).output[0].content[0].text for _ in range(3)]
    assert texts == ["first", "second", "ok"]


def test_a_strict_plan_refuses_to_improvise() -> None:
    source = MockModelSource({"r": [final_step("only")]}, strict=True)
    model = source.model_for("m", "r")
    one_call(model)
    with pytest.raises(MockPlanExhaustedError, match="exhausted at call 1"):
        one_call(model)


def test_a_function_plan_sees_each_call_and_its_index() -> None:
    seen: list[tuple[int, str]] = []

    def plan(call: ModelCall, index: int) -> ModelStep:
        seen.append((index, str(call.input)))
        return final_step(f"reply {index}")

    source = MockModelSource({"r": plan})
    model = source.model_for("m", "r")
    assert one_call(model, "alpha").output[0].content[0].text == "reply 0"
    assert one_call(model, "beta").output[0].content[0].text == "reply 1"
    assert [i for i, _ in seen] == [0, 1] and "alpha" in seen[0][1] and "beta" in seen[1][1]


def test_one_model_per_role_persists_across_invocations_and_records_calls() -> None:
    source = MockModelSource({"patcher": [final_step("a"), final_step("b")]})
    first = source.model_for("m", "patcher")
    assert source.model_for("other-model", "patcher") is first
    assert source.model_for("m", "reviewer") is not first
    one_call(first, "round one")
    one_call(first, "round two")
    assert len(source.calls("patcher")) == 2 and source.calls("reviewer") == ()
    assert "round two" in str(source.calls("patcher")[1].input)


def test_mock_usage_follows_the_real_convention_of_counting_reasoning_inside_output() -> None:
    step = tool_call_step(
        "t", {}, call_id="c", input_tokens=10, output_tokens=5, reasoning_tokens=7
    )
    assert step.usage.output_tokens == 12 and step.usage.output_tokens_details.reasoning_tokens == 7
    assert sdk_usage(1, 2).total_tokens == 3


# --- the OpenAI source (never contacts the network here) ---------------------------------------


def test_the_openai_source_refuses_without_the_spend_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "sk-not-a-real-key")
    with pytest.raises(SpendNotAllowedError):
        OpenAIModelSource()


@pytest.mark.parametrize("value", ["", "0", "true", "yes", "2"])
def test_the_gate_requires_exactly_one(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("CASCADE_ALLOW_SPEND", value)
    monkeypatch.setenv(API_KEY_ENV, "sk-not-a-real-key")
    with pytest.raises(SpendNotAllowedError):
        OpenAIModelSource()


def test_the_openai_source_needs_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CASCADE_ALLOW_SPEND", "1")
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(MissingApiKeyError):
        OpenAIModelSource()


def test_the_openai_source_builds_an_sdk_responses_model_and_rechecks_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CASCADE_ALLOW_SPEND", "1")
    monkeypatch.setenv(API_KEY_ENV, "sk-not-a-real-key")
    source = OpenAIModelSource()
    model = source.model_for("whatever-model", "patcher")
    assert isinstance(model, OpenAIResponsesModel) and source.is_real
    monkeypatch.delenv("CASCADE_ALLOW_SPEND")
    with pytest.raises(SpendNotAllowedError):  # closing the gate stops later models
        source.model_for("whatever-model", "reviewer")


def test_default_provider_is_mock_and_env_and_argument_override_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert resolve_provider_name() == "mock" and isinstance(get_model_source(), MockModelSource)
    monkeypatch.setenv("CASCADE_PROVIDER", "openai")
    assert resolve_provider_name() == "openai" and resolve_provider_name("mock") == "mock"
    with pytest.raises(SpendNotAllowedError):
        get_model_source()
    with pytest.raises(ValueError, match="unknown provider"):
        get_model_source("nope")
