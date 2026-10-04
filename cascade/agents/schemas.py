"""Structured outputs for the LLM roles, and records for the deterministic Verifier.

The LLM schemas are written for the Agents SDK's strict JSON-schema mode, which requires every
field to be present (no defaults) and forbids free-form dicts. Numeric ranges are therefore
described in the field text and enforced in code (see ``clamp_difficulty``), not by constraints.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

TRIAGER = "triager"
REPRODUCER = "reproducer"
LOCALIZER = "localizer"
PATCHER = "patcher"
REVIEWER = "reviewer"
LLM_ROLES = (TRIAGER, REPRODUCER, LOCALIZER, PATCHER, REVIEWER)
VERIFIER = "verifier"  # the sixth role: no LLM, it executes tests


class _LlmOut(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TriageOutput(_LlmOut):
    summary: str = Field(description="One or two sentences: what is wrong, in your own words.")
    issue_type: Literal["bug", "regression", "feature_request", "question", "other"]
    suspected_areas: list[str] = Field(
        description="Modules, classes, or functions the issue text points at. May be empty."
    )
    search_terms: list[str] = Field(
        description="Identifiers or phrases likely to appear in the relevant code."
    )
    difficulty: int = Field(description="Estimated fix difficulty from 1 (trivial) to 5 (hard).")
    reproducible_with_unit_test: bool = Field(
        description="True if a short pytest test could demonstrate the problem."
    )


def clamp_difficulty(value: int) -> int:
    return max(1, min(5, value))


class ReproOutput(_LlmOut):
    summary: str = Field(description="What the reproducer test checks and why it should fail now.")
    test_name: str = Field(description="Name of the test function you wrote.")
    expected_failure: str = Field(description="The failure you expect to see before any fix.")


class FileRef(_LlmOut):
    path: str = Field(description="Path relative to the repository root.")
    reason: str = Field(description="Why this file is likely involved.")


class LocalizationOutput(_LlmOut):
    files: list[FileRef] = Field(description="At most 5 files, most likely first.")
    functions: list[str] = Field(description="Functions or methods most likely to need changes.")
    notes: str = Field(description="Anything the patcher should know.")


class PatchOutput(_LlmOut):
    summary: str = Field(description="What you changed and why it fixes the issue.")
    files_changed: list[str] = Field(description="Paths of files you edited or created.")
    confidence: Literal["low", "medium", "high"]


class ReviewIssue(_LlmOut):
    severity: Literal["blocker", "major", "minor"]
    description: str


class ReviewOutput(_LlmOut):
    verdict: Literal["approve", "request_changes"]
    issues: list[ReviewIssue] = Field(description="Empty if you approve.")
    summary: str


# --- deterministic records (never produced by an LLM) -------------------------------------------


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VerifierReport(_Record):
    """Whether the agent's own evidence supports its patch (not whether hidden tests pass).

    ``claimed`` is verifier_claimed_success: the reproducer fails before the patch and passes after,
    and no test of the visible suite regresses. With the Reproducer disabled (an ablation) the
    reproducer condition is dropped and ``claimed`` rests on the regression check alone.
    """

    patch_nonempty: bool
    baseline_ok: bool
    repro_enabled: bool
    repro_failed_before: bool
    repro_passes_after: bool
    regressions: list[str]
    suite_passed_before: int
    suite_passed_after: int
    claimed: bool
    reasons: list[str]


class StageRecord(_Record):
    stage: str
    ok: bool
    detail: str = ""
    wall_s: Annotated[float, Field(ge=0)] = 0.0
