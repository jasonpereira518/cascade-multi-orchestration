"""LLM providers. The default is the mock provider; real providers sit behind the spend gate."""

from __future__ import annotations

import os

from cascade.harness.providers.base import Provider
from cascade.harness.providers.mock import MockProvider
from cascade.harness.providers.openai import OpenAIProvider

PROVIDER_ENV = "CASCADE_PROVIDER"


def get_provider(name: str | None = None) -> Provider:
    """Build a provider by name, falling back to $CASCADE_PROVIDER, then to "mock"."""
    chosen = name or os.environ.get(PROVIDER_ENV) or "mock"
    if chosen == "mock":
        return MockProvider()
    if chosen == "openai":
        return OpenAIProvider()
    raise ValueError(f"unknown provider {chosen!r}; expected 'mock' or 'openai'")
