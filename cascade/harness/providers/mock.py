"""Scriptable mock provider: deterministic, offline, $0. The default provider."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from cascade.errors import ProviderError
from cascade.harness.providers.base import LLMRequest, LLMResponse, Provider, Usage


class MockScriptExhaustedError(ProviderError):
    """The scripted replies ran out and no default reply was configured."""


@dataclass(frozen=True)
class MockReply:
    text: str = "mock reply"
    # None means "estimate from the text length" (about 4 chars per token).
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int = 0
    latency_s: float = 0.0
    # If set, the call raises ProviderError(error) instead of returning.
    error: str | None = None


ScriptFn = Callable[[LLMRequest, int], MockReply]


def _estimate(chars: int) -> int:
    return max(1, chars // 4)


class MockProvider(Provider):
    """Replies come from ``script`` (a sequence consumed in order, or a function of
    ``(request, call_index)``). Once a sequence is exhausted the ``default`` reply is used, or
    MockScriptExhaustedError is raised if there is no default. With neither script nor default,
    every call returns ``MockReply()``.
    """

    name = "mock"
    is_real = False

    def __init__(
        self,
        script: Sequence[MockReply] | ScriptFn | None = None,
        *,
        default: MockReply | None = None,
    ) -> None:
        if script is None and default is None:
            default = MockReply()
        self._script = script
        self._default = default
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        index = len(self.calls)
        self.calls.append(request)
        reply = self._next_reply(request, index)
        if reply.error is not None:
            raise ProviderError(reply.error)

        input_tokens = reply.input_tokens
        if input_tokens is None:
            input_tokens = _estimate(sum(len(m.content) for m in request.messages))
        output_tokens = reply.output_tokens
        if output_tokens is None:
            output_tokens = _estimate(len(reply.text))

        reasoning = min(reply.reasoning_tokens, request.max_output_tokens)
        finish = "stop"
        if reasoning + output_tokens > request.max_output_tokens:
            output_tokens = request.max_output_tokens - reasoning
            finish = "length"
        return LLMResponse(
            text=reply.text,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning,
            ),
            latency_s=reply.latency_s,
            model=request.model,
            finish_reason=finish,
        )

    def _next_reply(self, request: LLMRequest, index: int) -> MockReply:
        script = self._script
        if callable(script):
            return script(request, index)
        if script is not None and index < len(script):
            return script[index]
        if self._default is not None:
            return self._default
        raise MockScriptExhaustedError(f"mock script exhausted at call {index}")
