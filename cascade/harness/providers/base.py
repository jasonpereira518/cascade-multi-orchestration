"""Model sources: where an arm's model comes from. Metering and caps live in harness.models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, ClassVar

from agents.models.interface import Model
from pydantic import BaseModel, ConfigDict, Field


class Usage(BaseModel):
    """Non-overlapping token counts, the shape the ledger prices.

    ``output_tokens`` EXCLUDES ``reasoning_tokens``. The Agents SDK reports reasoning tokens as a
    subset of output tokens; harness.models subtracts them so nothing is billed twice.
    ``cached_tokens`` and ``cache_write_tokens`` are SUBSETS of ``input_tokens`` (the provider
    reports total input and says how much of it was a cache hit or a cache write).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    reasoning_tokens: Annotated[int, Field(ge=0)] = 0
    cached_tokens: Annotated[int, Field(ge=0)] = 0
    cache_write_tokens: Annotated[int, Field(ge=0)] = 0


class ModelSource(ABC):
    name: ClassVar[str]
    # Real sources bill money. MeteredModel refuses them unless the spend gate is open.
    is_real: ClassVar[bool]

    @abstractmethod
    def model_for(self, model_name: str, role: str) -> Model:
        """Return the (unmetered) SDK model that serves ``role`` for the named model."""
