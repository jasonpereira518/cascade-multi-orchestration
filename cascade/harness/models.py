"""Metering for Agents SDK models: every LLM call is authorized, costed, and logged.

``MeteredModel`` wraps any SDK ``Model`` (real or mock). Around each call it runs, in order:
  1. ledger.authorize: kill switch, spend gate (real models only), global/task/attempt dollar caps
     and the attempt token cap, all on worst-case figures, before anything is sent;
  2. the wrapped model call;
  3. ledger.record: cost computed from the provider's reported usage and models.yaml prices, or a
     zero-cost error row if the call raised.

Because it sits at the ``Model`` boundary, every role's model calls, including each turn of a
tool-using loop, go through here; nothing in the agents can bypass the ledger.

Usage normalization: the SDK reports ``output_tokens`` INCLUDING reasoning tokens; the ledger wants
them separate (see providers.base.Usage), so reasoning is subtracted from output. Cached input
tokens are billed as ordinary input (models.yaml has no cached price), which can only
overstate cost.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from agents import ModelSettings
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from agents.models.interface import Model, ModelTracing
from agents.tool import Tool
from agents.usage import Usage as SdkUsage
from openai.types.responses.response_prompt_param import ResponsePromptParam

from cascade.errors import CascadeError
from cascade.harness.ledger import CallContext, Ledger
from cascade.harness.providers.base import Usage
from cascade.harness.trace import TraceWriter

# Pessimistic on purpose (see Ledger.authorize): errs toward refusing a call near a cap.
_CHARS_PER_TOKEN_ESTIMATE = 3


@dataclass(frozen=True)
class AttemptRef:
    """Identifies one attempt: a pipeline execution of one arm on one task with one seed."""

    run_id: str
    task_id: str
    attempt_id: str
    arm_id: str


def usage_from_sdk(usage: SdkUsage) -> Usage:
    reasoning = usage.output_tokens_details.reasoning_tokens or 0
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=max(usage.output_tokens - reasoning, 0),
        reasoning_tokens=reasoning,
    )


def estimate_input_tokens(
    system_instructions: str | None, input: str | list[TResponseInputItem]
) -> int:
    body = input if isinstance(input, str) else json.dumps(input, default=str)
    return math.ceil((len(system_instructions or "") + len(body)) / _CHARS_PER_TOKEN_ESTIMATE)


class MeteredModel(Model):
    def __init__(
        self,
        inner: Model,
        *,
        ledger: Ledger,
        attempt: AttemptRef,
        role: str,
        model_name: str,
        provider_name: str,
        real: bool,
        trace: TraceWriter | None = None,
    ) -> None:
        self._inner = inner
        self._ledger = ledger
        self._attempt = attempt
        self._role = role
        self._model_name = model_name
        self._provider = provider_name
        self._real = real
        self._trace = trace or TraceWriter(None)

    def _next_context(self) -> CallContext:
        a = self._attempt
        index = self._ledger.call_count(task_id=a.task_id, attempt_id=a.attempt_id)
        raw = f"{a.task_id}|{a.attempt_id}|{index}"
        return CallContext(
            call_id=hashlib.sha256(raw.encode()).hexdigest()[:32],
            run_id=a.run_id,
            task_id=a.task_id,
            attempt_id=a.attempt_id,
            arm_id=a.arm_id,
            role=self._role,
        )

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
        max_out = model_settings.max_tokens
        if max_out is None:
            raise CascadeError("model_settings.max_tokens must be set so caps can be enforced")
        ctx = self._next_context()
        # Raises BudgetExceeded / KillSwitchEngaged / SpendNotAllowed before anything is sent.
        self._ledger.authorize(
            ctx,
            model=self._model_name,
            est_input_tokens=estimate_input_tokens(system_instructions, input),
            max_output_tokens=max_out,
            real=self._real,
        )
        start = time.monotonic()
        try:
            response = await self._inner.get_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                prompt=prompt,
            )
        except Exception as e:
            self._ledger.record(
                ctx,
                provider=self._provider,
                model=self._model_name,
                usage=Usage(input_tokens=0, output_tokens=0),
                latency_s=time.monotonic() - start,
                status="error",
                error=f"{type(e).__name__}: {e}"[:500],
            )
            self._trace.write(
                "llm", role=self._role, call_id=ctx.call_id, system=system_instructions,
                input=input, error=f"{type(e).__name__}: {e}",
            )  # fmt: skip
            raise
        usage = usage_from_sdk(response.usage)
        record = self._ledger.record(
            ctx,
            provider=self._provider,
            model=self._model_name,
            usage=usage,
            latency_s=time.monotonic() - start,
        )
        self._trace.write(
            "llm", role=self._role, call_id=ctx.call_id, model=self._model_name,
            system=system_instructions, input=input,
            output=[_dump(item) for item in response.output],
            usage=usage.model_dump(), cost_usd=record.cost_usd,
        )  # fmt: skip
        return response

    def stream_response(
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
    ) -> AsyncIterator[TResponseStreamEvent]:
        # Streaming would make usage arrive after the fact and bypass pre-call authorization.
        raise NotImplementedError("MeteredModel supports non-streaming calls only")


def _dump(item: Any) -> Any:
    dump = getattr(item, "model_dump", None)
    return dump(mode="json") if callable(dump) else item
