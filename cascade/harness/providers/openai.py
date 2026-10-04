"""OpenAI provider STUB. It refuses to run without CASCADE_ALLOW_SPEND=1.

No network code exists here yet. When the live adapter is written it must (a) check the installed
SDK version and docs first (hard rule 8), (b) normalize usage so output_tokens excludes
reasoning_tokens, and (c) read OPENAI_API_KEY from the environment only.
"""

from __future__ import annotations

from cascade.harness.providers.base import LLMRequest, LLMResponse, Provider
from cascade.harness.spend import require_spend_allowed


class OpenAIProvider(Provider):
    name = "openai"
    is_real = True

    def __init__(self) -> None:
        require_spend_allowed("constructing OpenAIProvider")

    def complete(self, request: LLMRequest) -> LLMResponse:
        # Re-checked per call: the env can change after construction.
        require_spend_allowed("OpenAIProvider.complete")
        raise NotImplementedError(
            "OpenAI adapter is not implemented yet; this stub only enforces the spend gate"
        )
