from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from agents import ModelSettings
from agents.items import ModelResponse
from agents.models.interface import ModelTracing
from agents.testing import ModelStep

from cascade.config import ModelsConfig, ResolvedBudget
from cascade.errors import (
    BudgetExceededError,
    CascadeError,
    KillSwitchEngagedError,
    SpendNotAllowedError,
)
from cascade.harness.ledger import Ledger
from cascade.harness.models import AttemptRef, MeteredModel, estimate_input_tokens, usage_from_sdk
from cascade.harness.providers.mock import MockModelSource, final_step, sdk_usage
from cascade.harness.trace import TraceWriter

ATTEMPT = AttemptRef("run-1", "task-1", "attempt-1", "arm-1")


def metered(
    ledger: Ledger, models: ModelsConfig, steps: list[ModelStep] | None = None, *,
    model_index: int = 0, real: bool = False, trace: TraceWriter | None = None,
) -> tuple[MeteredModel, MockModelSource]:  # fmt: skip
    source = MockModelSource({"triager": steps if steps is not None else [final_step("hi")]})
    inner = source.model_for("m", "triager")
    model = MeteredModel(
        inner, ledger=ledger, attempt=ATTEMPT, role="triager",
        model_name=models.models[model_index].name, provider_name="mock", real=real, trace=trace,
    )  # fmt: skip
    return model, source


def call(model: MeteredModel, text: str = "hello", max_tokens: int | None = 100) -> ModelResponse:
    return asyncio.run(
        model.get_response(
            "system", text, ModelSettings(max_tokens=max_tokens), [], None, [], ModelTracing.DISABLED,
            previous_response_id=None, conversation_id=None, prompt=None,
        )
    )  # fmt: skip


def test_usage_from_sdk_separates_reasoning_from_output() -> None:
    u = usage_from_sdk(sdk_usage(100, 50, 25))
    assert (u.input_tokens, u.output_tokens, u.reasoning_tokens) == (100, 50, 25)
    # The SDK reports output INCLUDING reasoning, so the raw object carries 75 output tokens.
    assert sdk_usage(100, 50, 25).output_tokens == 75


def test_a_call_is_recorded_with_provider_usage_and_priced_from_models_yaml(
    ledger: Ledger, models: ModelsConfig
) -> None:
    model, _ = metered(
        ledger, models, [final_step("hi", input_tokens=100, output_tokens=50, reasoning_tokens=25)]
    )
    call(model)
    [row] = ledger.rows()
    p = models.prices_for(models.models[0].name)
    expected = (
        100 * p.input_usd_per_mtok + 50 * p.output_usd_per_mtok + 25 * p.reasoning_usd_per_mtok
    ) / 1e6
    assert (row.input_tokens, row.output_tokens, row.reasoning_tokens) == (100, 50, 25)
    assert row.cost_usd == pytest.approx(expected) and row.cost_usd > 0
    assert (row.task_id, row.attempt_id, row.arm_id, row.role, row.provider) == (
        "task-1", "attempt-1", "arm-1", "triager", "mock",
    )  # fmt: skip
    assert row.status == "ok" and row.latency_s >= 0


def test_calls_get_distinct_deterministic_ids_and_all_land_in_the_ledger(
    ledger: Ledger, models: ModelsConfig
) -> None:
    model, _ = metered(ledger, models, [final_step("a"), final_step("b"), final_step("c")])
    for _ in range(3):
        call(model)
    ids = [r.call_id for r in ledger.rows()]
    assert len(set(ids)) == 3 and ledger.call_count(task_id="task-1", attempt_id="attempt-1") == 3


def test_over_cap_calls_are_refused_before_anything_is_sent(
    ledger: Ledger, models: ModelsConfig
) -> None:
    # mock-large worst case: 3000 output tokens at 50/Mtok = $0.15, above the $0.10 attempt cap.
    model, source = metered(ledger, models, model_index=1)
    with pytest.raises(BudgetExceededError, match="attempt cap"):
        call(model, max_tokens=3000)
    assert source.calls("triager") == () and ledger.rows() == []


def test_the_attempt_token_cap_is_enforced(
    ledger: Ledger, models: ModelsConfig, budget: ResolvedBudget
) -> None:
    model, source = metered(ledger, models)
    with pytest.raises(BudgetExceededError, match="token cap"):
        call(model, max_tokens=budget.per_attempt_max_tokens)  # request alone fills the cap
    assert source.calls("triager") == ()


def test_the_kill_switch_stops_calls(
    ledger: Ledger, models: ModelsConfig, budget: ResolvedBudget
) -> None:
    model, source = metered(ledger, models, [final_step("a"), final_step("b")])
    call(model)
    budget.kill_switch_path.touch()
    with pytest.raises(KillSwitchEngagedError):
        call(model)
    assert len(source.calls("triager")) == 1 and len(ledger.rows()) == 1


def test_real_models_need_the_spend_gate_even_with_budget_room(
    ledger: Ledger, models: ModelsConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, source = metered(ledger, models, real=True)
    with pytest.raises(SpendNotAllowedError):
        call(model)
    assert source.calls("triager") == () and ledger.rows() == []
    monkeypatch.setenv("CASCADE_ALLOW_SPEND", "1")
    call(model)  # gate open and under every cap: allowed
    assert len(ledger.rows()) == 1


def test_a_failing_call_is_logged_as_a_zero_cost_error_and_reraised(
    ledger: Ledger, models: ModelsConfig
) -> None:
    model, _ = metered(ledger, models, [ModelStep(error=RuntimeError("upstream 500"))])
    with pytest.raises(RuntimeError, match="upstream 500"):
        call(model)
    [row] = ledger.rows()
    assert row.status == "error" and "upstream 500" in str(row.error) and row.cost_usd == 0.0


def test_a_call_without_max_tokens_is_refused(ledger: Ledger, models: ModelsConfig) -> None:
    model, source = metered(ledger, models)
    with pytest.raises(CascadeError, match="max_tokens"):
        call(model, max_tokens=None)
    assert source.calls("triager") == ()


def test_the_estimate_is_pessimistic_and_counts_instructions_and_input() -> None:
    assert estimate_input_tokens(None, "x" * 300) == 100
    assert estimate_input_tokens("y" * 30, "x" * 300) == 110
    assert estimate_input_tokens(None, [{"role": "user", "content": "hello"}]) > 10


def test_every_call_is_traced_in_full(ledger: Ledger, models: ModelsConfig, tmp_path: Path) -> None:
    trace = TraceWriter(tmp_path / "t.jsonl")
    model, _ = metered(ledger, models, trace=trace)
    call(model, text="the issue says clamp is broken")
    [line] = [json.loads(ln) for ln in trace.text().splitlines()]
    assert (
        line["kind"] == "llm"
        and line["role"] == "triager"
        and line["model"] == models.models[0].name
    )
    assert line["system"] == "system" and "clamp is broken" in line["input"]
    assert line["output"][0]["content"][0]["text"] == "hi"
    assert line["usage"]["input_tokens"] == 300 and line["cost_usd"] > 0


def test_streaming_is_not_supported_because_it_would_bypass_authorization(
    ledger: Ledger, models: ModelsConfig
) -> None:
    model, _ = metered(ledger, models)
    with pytest.raises(NotImplementedError):
        model.stream_response(
            None, "x", ModelSettings(max_tokens=10), [], None, [], ModelTracing.DISABLED,
            previous_response_id=None, conversation_id=None, prompt=None,
        )  # fmt: skip
