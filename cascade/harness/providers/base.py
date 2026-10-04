"""Provider interface. Providers do I/O only; metering and caps live in the ledger layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Message(_Strict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(_Strict):
    model: str
    messages: list[Message]
    reasoning_effort: str | None = None
    # Hard ceiling on generated tokens, including reasoning tokens (billed as output).
    max_output_tokens: Annotated[int, Field(gt=0)]


class Usage(_Strict):
    """Non-overlapping token counts. Providers must normalize to this shape.

    ``output_tokens`` EXCLUDES ``reasoning_tokens``. Some APIs report reasoning tokens as a subset
    of output tokens; the provider adapter subtracts them so the ledger never double-counts.
    """

    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    reasoning_tokens: Annotated[int, Field(ge=0)] = 0


class LLMResponse(_Strict):
    text: str
    usage: Usage
    latency_s: Annotated[float, Field(ge=0)]
    model: str
    finish_reason: str = "stop"


class Provider(ABC):
    name: ClassVar[str]
    # Real providers bill money. The metered client refuses them unless the spend gate is open.
    is_real: ClassVar[bool]

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse: ...
