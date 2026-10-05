"""Real OpenAI models through the OpenAI Agents SDK. Refuses to run without CASCADE_ALLOW_SPEND=1.

The SDK's Responses model (``OpenAIResponsesModel``) does the HTTP; this module only builds it. The
API was checked against the installed openai-agents 0.23.1 / openai 3.24.0: the constructor takes
``(model, openai_client)`` and building it makes no network call. Metering is not done here:
``harness.models.MeteredModel`` wraps whatever this returns, so cost comes from the usage fields
the API reports and every call lands in the ledger.
"""

from __future__ import annotations

import os

from agents.models.interface import Model
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncOpenAI

from cascade.errors import CascadeError
from cascade.harness.providers.base import ModelSource
from cascade.harness.spend import require_spend_allowed

API_KEY_ENV = "OPENAI_API_KEY"
STANDARD_TIMEOUT_S = 300.0
# Flex requests can queue; OpenAI documents a 10-minute default timeout and advises raising it.
FLEX_TIMEOUT_S = 900.0


class MissingApiKeyError(CascadeError):
    """OPENAI_API_KEY is not set."""


class OpenAIModelSource(ModelSource):
    name = "openai"
    is_real = True

    def __init__(self, *, flex: bool = False) -> None:
        require_spend_allowed("constructing OpenAIModelSource")
        key = os.environ.get(API_KEY_ENV)
        if not key:
            raise MissingApiKeyError(f"{API_KEY_ENV} is not set")
        # Constructing the client does not contact the network.
        self.timeout_s = FLEX_TIMEOUT_S if flex else STANDARD_TIMEOUT_S
        self._client = AsyncOpenAI(api_key=key, max_retries=2, timeout=self.timeout_s)

    def model_for(self, model_name: str, role: str) -> Model:
        # Re-checked per model: the env can change after construction.
        require_spend_allowed(f"real model {model_name!r} for role {role}")
        return OpenAIResponsesModel(model=model_name, openai_client=self._client)
