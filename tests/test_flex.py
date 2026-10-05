"""Flex tier plumbing and cached-token usage. No network: the inner model is a scripted fake.

Flex is requested per call (``service_tier="flex"``) and a ``429 Resource Unavailable`` is retried
with backoff, never charged, and never answered by falling back to standard (one matrix, one tier).
Whether a given real model supports flex cannot be tested offline; see DECISIONS.md.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx2 as httpx  # the openai SDK's HTTP layer (a transitive dependency)
import openai
import pytest
from agents import ModelSettings
from agents.items import ModelResponse
from agents.models.interface import Model, ModelTracing

from cascade.agents.roles import model_settings_for
from cascade.config import ArmSpec
from cascade.errors import CascadeError
from cascade.harness.ledger import Ledger
from cascade.harness.models import (
    AttemptRef,
    FlexRetryPolicy,
    MeteredModel,
    is_resource_unavailable,
    usage_from_sdk,
)
from cascade.harness.providers.mock import MockModelSource, sdk_usage
from cascade.harness.trace import TraceWriter
from tests.matrix_helpers import CHEAP, matrix_models, new_ledger


def rate_limit(message: str) -> openai.RateLimitError:
    req = httpx.Request("POST", "https://api.example.invalid/v1/responses")
    return openai.RateLimitError(message, response=httpx.Response(429, request=req), body=None)


class FlakyModel(Model):
    """Raises the queued errors, one per call, then answers like the mock."""

    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = list(errors)
        self.calls = 0
        self._ok = MockModelSource().model_for("m", "triager")

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return await self._ok.get_response(*args, **kwargs)

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def metered(
    tmp_path: Path,
    inner: Model,
    *,
    tier: str,
    bucket: str = "pilot",
    model_name: str = "mock-luna",
    policy: FlexRetryPolicy | None = None,
    trace: TraceWriter | None = None,
) -> tuple[MeteredModel, list[float], Ledger]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    ledger = new_ledger(tmp_path)
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    model = MeteredModel(
        inner, ledger=ledger, attempt=AttemptRef("r", "t1", "a1", CHEAP, bucket, tier),
        role="triager", model_name=model_name, provider_name="mock", real=False, trace=trace,
        flex_retry=policy, sleep=sleep,
    )  # fmt: skip
    return model, slept, ledger


def call(model: MeteredModel) -> ModelResponse:
    return asyncio.run(
        model.get_response(
            "system", "hello", ModelSettings(max_tokens=100), [], None, [], ModelTracing.DISABLED,
            previous_response_id=None, conversation_id=None, prompt=None,
        )
    )  # fmt: skip


# --- request settings ---------------------------------------------------------------------------


def test_standard_sends_no_service_tier_and_flex_sends_flex() -> None:
    arm = ArmSpec(model="mock-luna", reasoning_effort="low")
    standard = model_settings_for(arm, 1000)
    flex = model_settings_for(arm, 1000, "flex")
    assert standard.extra_args is None  # the default path is unchanged by the existence of tiers
    assert flex.extra_args == {"service_tier": "flex"}
    assert flex.max_tokens == 1000 and flex.store is False
    assert flex.reasoning is not None and flex.reasoning.effort == "low"


# --- retry --------------------------------------------------------------------------------------


def test_is_resource_unavailable_matches_only_the_flex_capacity_429() -> None:
    assert is_resource_unavailable(rate_limit("429 Resource Unavailable"))
    assert is_resource_unavailable(rate_limit("Resource unavailable: try again later"))
    assert not is_resource_unavailable(rate_limit("Rate limit reached for requests"))
    assert not is_resource_unavailable(ValueError("resource unavailable"))


def test_flex_retries_resource_unavailable_with_exponential_capped_backoff(tmp_path: Path) -> None:
    inner = FlakyModel([rate_limit("Resource Unavailable")] * 3)
    policy = FlexRetryPolicy(max_retries=5, base_s=2.0, cap_s=5.0)
    model, slept, ledger = metered(tmp_path, inner, tier="flex", policy=policy)
    call(model)
    assert inner.calls == 4 and slept == [2.0, 4.0, 5.0]  # 2, 4, then capped at 5
    rows = ledger.rows()
    assert len(rows) == 1 and rows[0].status == "ok"  # the uncharged 429s leave no rows
    assert rows[0].processing_tier == "flex" and rows[0].bucket == "pilot"
    ledger.close()


def test_flex_cost_is_the_discounted_cost(tmp_path: Path) -> None:
    model, _, ledger = metered(tmp_path, FlakyModel([]), tier="flex")
    call(model)
    [rec] = ledger.rows()
    standard = ledger.cost_of("mock-luna", usage_from_sdk(sdk_usage(200, 50)))
    assert rec.cost_usd == pytest.approx(standard * 0.5)
    ledger.close()


def test_exhausted_flex_retries_become_one_error_row_and_the_error_surfaces(tmp_path: Path) -> None:
    inner = FlakyModel([rate_limit("Resource Unavailable")] * 10)
    policy = FlexRetryPolicy(max_retries=2, base_s=1.0, cap_s=60.0)
    model, slept, ledger = metered(tmp_path, inner, tier="flex", policy=policy)
    with pytest.raises(openai.RateLimitError):
        call(model)
    assert inner.calls == 3 and slept == [1.0, 2.0]  # the original try plus 2 retries
    [rec] = ledger.rows()
    assert rec.status == "error" and rec.cost_usd == 0 and rec.processing_tier == "flex"
    ledger.close()


def test_standard_tier_never_retries_even_a_resource_unavailable_error(tmp_path: Path) -> None:
    inner = FlakyModel([rate_limit("Resource Unavailable")])
    model, slept, ledger = metered(tmp_path, inner, tier="standard")
    with pytest.raises(openai.RateLimitError):
        call(model)
    assert inner.calls == 1 and slept == []
    ledger.close()


@pytest.mark.parametrize("error", [rate_limit("Rate limit reached"), RuntimeError("boom")])
def test_flex_does_not_retry_other_429s_or_other_errors(tmp_path: Path, error: Exception) -> None:
    inner = FlakyModel([error])
    model, slept, ledger = metered(tmp_path, inner, tier="flex")
    with pytest.raises(type(error)):
        call(model)
    assert inner.calls == 1 and slept == []
    ledger.close()


def test_retries_are_traced(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    inner = FlakyModel([rate_limit("Resource Unavailable")])
    model, _, ledger = metered(tmp_path, inner, tier="flex", trace=TraceWriter(trace_path))
    call(model)
    assert "flex_retry" in trace_path.read_text()
    ledger.close()


def test_a_model_without_a_flex_multiplier_cannot_run_on_flex(tmp_path: Path) -> None:
    inner = FlakyModel([])
    model, _, ledger = metered(tmp_path, inner, tier="flex", model_name="mock-noflex")
    with pytest.raises(CascadeError, match="no flex_multiplier"):
        call(model)
    assert inner.calls == 0  # refused before anything was sent
    ledger.close()


def test_the_retry_policy_delay_doubles_and_caps() -> None:
    p = FlexRetryPolicy(max_retries=9, base_s=5.0, cap_s=120.0)
    assert [p.delay(i) for i in range(6)] == [5.0, 10.0, 20.0, 40.0, 80.0, 120.0]


# --- cached tokens ------------------------------------------------------------------------------


def test_cached_and_cache_write_tokens_are_read_from_the_sdk_usage() -> None:
    u = usage_from_sdk(sdk_usage(1000, 50, 10, cached_tokens=300, cache_write_tokens=40))
    assert (u.input_tokens, u.output_tokens, u.reasoning_tokens) == (1000, 50, 10)
    assert (u.cached_tokens, u.cache_write_tokens) == (300, 40)


def test_a_metered_call_logs_cached_tokens_and_prices_them(tmp_path: Path) -> None:
    class Cached(FlakyModel):
        async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
            r = await super().get_response(*args, **kwargs)
            r.usage = sdk_usage(1000, 0, 0, cached_tokens=600)
            return r

    model, _, ledger = metered(tmp_path, Cached([]), tier="standard")
    call(model)
    [rec] = ledger.rows()
    assert rec.cached_tokens == 600
    assert rec.cost_usd == pytest.approx((400 * 1.0 + 600 * 0.25) / 1e6)  # synthetic luna prices
    ledger.close()


def test_the_synthetic_models_file_marks_flex_support_per_model() -> None:
    m = matrix_models()
    assert m.model_spec("mock-luna").flex_multiplier == 0.5
    assert m.model_spec("mock-noflex").flex_multiplier is None


# --- the real source's client timeout -------------------------------------------------------------


def test_the_openai_client_timeout_is_longer_for_flex(monkeypatch: pytest.MonkeyPatch) -> None:
    from cascade.harness.providers.openai import (
        FLEX_TIMEOUT_S,
        STANDARD_TIMEOUT_S,
        OpenAIModelSource,
    )

    monkeypatch.setenv("CASCADE_ALLOW_SPEND", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")  # building the client is offline
    standard, flex = OpenAIModelSource(), OpenAIModelSource(flex=True)
    assert standard.timeout_s == STANDARD_TIMEOUT_S == 300.0
    assert flex.timeout_s == FLEX_TIMEOUT_S == 900.0 > standard.timeout_s


def test_the_cli_builds_a_long_timeout_source_only_for_flex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cascade import matrix_cli
    from cascade.harness.providers.openai import OpenAIModelSource

    monkeypatch.setenv("CASCADE_ALLOW_SPEND", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    flex = matrix_cli._make_source("openai", "flex")  # pyright: ignore[reportPrivateUsage]
    std = matrix_cli._make_source("openai", "standard")  # pyright: ignore[reportPrivateUsage]
    assert isinstance(flex, OpenAIModelSource) and isinstance(std, OpenAIModelSource)
    assert flex.timeout_s == 900.0 and std.timeout_s == 300.0
