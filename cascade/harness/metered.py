"""Provider wrapper that routes every call through the ledger: authorize, call, record."""

from __future__ import annotations

import math

from cascade.errors import DuplicateCallError, ProviderError
from cascade.harness.ledger import CallContext, Ledger
from cascade.harness.providers.base import LLMRequest, LLMResponse, Provider, Usage

# Deliberately pessimistic (code and non-English text tokenize denser than 4 chars/token) so the
# pre-call cap check errs toward refusing. The recorded usage always comes from the provider.
_CHARS_PER_TOKEN_ESTIMATE = 3


def estimate_input_tokens(request: LLMRequest) -> int:
    chars = sum(len(m.content) for m in request.messages)
    return math.ceil(chars / _CHARS_PER_TOKEN_ESTIMATE)


class MeteredClient:
    def __init__(self, provider: Provider, ledger: Ledger) -> None:
        self._provider = provider
        self._ledger = ledger

    def complete(self, ctx: CallContext, request: LLMRequest) -> LLMResponse:
        # A real call repeated under the same id would never show up twice in the ledger.
        # Resumable jobs must skip completed calls using their own persisted outputs.
        if self._ledger.has_call(ctx.call_id):
            raise DuplicateCallError(f"call_id already recorded: {ctx.call_id}")
        self._ledger.authorize(
            ctx,
            model=request.model,
            est_input_tokens=estimate_input_tokens(request),
            max_output_tokens=request.max_output_tokens,
            real=self._provider.is_real,
        )
        try:
            response = self._provider.complete(request)
        except ProviderError as e:
            self._ledger.record(
                ctx,
                provider=self._provider.name,
                model=request.model,
                usage=Usage(input_tokens=0, output_tokens=0),
                latency_s=0.0,
                status="error",
                error=str(e),
            )
            raise
        self._ledger.record(
            ctx,
            provider=self._provider.name,
            model=request.model,
            usage=response.usage,
            latency_s=response.latency_s,
        )
        return response
