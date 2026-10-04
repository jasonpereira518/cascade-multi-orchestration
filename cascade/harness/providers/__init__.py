"""Model sources. The default is the mock; the real one sits behind the spend gate."""

from __future__ import annotations

import os

from cascade.harness.providers.base import ModelSource
from cascade.harness.providers.mock import MockModelSource
from cascade.harness.providers.openai import OpenAIModelSource

PROVIDER_ENV = "CASCADE_PROVIDER"


def resolve_provider_name(name: str | None = None) -> str:
    return name or os.environ.get(PROVIDER_ENV) or "mock"


def get_model_source(name: str | None = None) -> ModelSource:
    """Build a source by name, falling back to $CASCADE_PROVIDER, then to "mock"."""
    chosen = resolve_provider_name(name)
    if chosen == "mock":
        return MockModelSource()
    if chosen == "openai":
        return OpenAIModelSource()
    raise ValueError(f"unknown provider {chosen!r}; expected 'mock' or 'openai'")
