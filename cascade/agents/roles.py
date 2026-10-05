"""The five LLM roles as Agents SDK agents (the sixth, the Verifier, is deterministic code).

Each role is one SDK ``Agent`` with a pydantic ``output_type`` (structured output), its own tool
list, a turn budget, and a model that is the arm's model wrapped in ``MeteredModel``. Tracing is
disabled on every run: the SDK's default tracing uploads prompts to a third-party backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from agents import Agent, ModelSettings, RunConfig, Runner
from agents.models.interface import Model
from openai.types.shared import Reasoning
from pydantic import BaseModel

from cascade.agents import prompts
from cascade.agents.schemas import (
    LOCALIZER,
    PATCHER,
    REPRODUCER,
    REVIEWER,
    TRIAGER,
    LocalizationOutput,
    PatchOutput,
    ReproOutput,
    ReviewOutput,
    TriageOutput,
)
from cascade.agents.tools import Toolbox
from cascade.config import ArmSpec


@dataclass(frozen=True)
class RoleSpec:
    name: str
    output_type: type[BaseModel]
    tools: tuple[str, ...]
    max_turns: int

    @property
    def instructions(self) -> str:
        return prompts.INSTRUCTIONS[self.name]


ROLE_SPECS: dict[str, RoleSpec] = {
    TRIAGER: RoleSpec(TRIAGER, TriageOutput, (), 2),
    REPRODUCER: RoleSpec(
        REPRODUCER,
        ReproOutput,
        ("list_files", "read_file", "search_code", "grep", "write_reproducer", "run_tests"),
        14,
    ),
    LOCALIZER: RoleSpec(
        LOCALIZER, LocalizationOutput, ("list_files", "read_file", "search_code", "grep"), 10
    ),
    PATCHER: RoleSpec(
        PATCHER,
        PatchOutput,
        ("list_files", "read_file", "search_code", "grep", "edit_file", "create_file", "run_tests"),
        24,
    ),
    REVIEWER: RoleSpec(REVIEWER, ReviewOutput, ("read_file", "grep"), 6),
}


def model_settings_for(
    arm: ArmSpec, max_output_tokens: int, tier: str = "standard"
) -> ModelSettings:
    """Settings for every call of an arm: the arm's reasoning effort and a hard output ceiling.

    ``store=False`` keeps the provider from retaining responses server-side. ``max_tokens`` bounds
    reasoning plus visible output, and is also what the ledger prices as the worst case.
    """
    reasoning = None
    if arm.reasoning_effort is not None:
        reasoning = Reasoning(effort=cast(Any, arm.reasoning_effort))
    # service_tier goes out as an extra request argument; standard sends nothing, so the default
    # path is byte-for-byte what it was before tiers existed.
    extra = {"service_tier": "flex"} if tier == "flex" else None
    return ModelSettings(
        max_tokens=max_output_tokens, reasoning=reasoning, store=False, extra_args=extra
    )


def build_agent(
    spec: RoleSpec,
    *,
    model: Model,
    toolbox: Toolbox | None,
    arm: ArmSpec,
    max_output_tokens: int,
    tier: str = "standard",
) -> Agent[Any]:
    tools = toolbox.tools(spec.tools) if toolbox is not None else []
    return Agent(
        name=spec.name,
        instructions=spec.instructions,
        tools=tools,
        model=model,
        model_settings=model_settings_for(arm, max_output_tokens, tier),
        output_type=spec.output_type,
    )


def run_role(agent: Agent[Any], spec: RoleSpec, input_text: str) -> BaseModel:
    """Run one role to completion and return its validated structured output."""
    result = Runner.run_sync(
        agent, input_text, max_turns=spec.max_turns, run_config=RunConfig(tracing_disabled=True)
    )
    return result.final_output_as(spec.output_type, raise_if_incorrect_type=True)
