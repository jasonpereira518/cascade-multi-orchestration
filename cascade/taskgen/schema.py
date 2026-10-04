"""Task schema (docs/SPEC.md) and the restricted view agents are allowed to see."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SHA40 = re.compile(r"^[0-9a-f]{40}$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Features(_Strict):
    src_files: Annotated[int, Field(ge=0)]
    src_lines: Annotated[int, Field(ge=0)]
    issue_chars: Annotated[int, Field(ge=0)]
    test_runtime_s: Annotated[float, Field(ge=0)]


class Audit(_Strict):
    status: Literal["pending", "ok", "rejected"] = "pending"
    notes: str = ""


class Task(_Strict):
    task_id: str
    repo: str
    base_commit: str
    issue_title: str
    issue_body: str
    gold_patch: str
    test_patch: str
    fail_to_pass: Annotated[list[str], Field(min_length=1)]
    pass_to_pass: list[str]
    image: str
    features: Features
    # None until the frozen split (by hash and by repo) assigns it.
    split: Literal["train", "test"] | None = None
    audit: Audit = Field(default_factory=Audit)

    @field_validator("base_commit")
    @classmethod
    def _full_sha(cls, v: str) -> str:
        if not _SHA40.fullmatch(v):
            raise ValueError("base_commit must be a full 40-char lowercase hex sha")
        return v

    def agent_view(self) -> AgentView:
        return AgentView(
            task_id=self.task_id,
            repo=self.repo,
            base_commit=self.base_commit,
            issue_title=self.issue_title,
            issue_body=self.issue_body,
        )


class AgentView(_Strict):
    """Everything an agent may see about a task: the issue text and where the repo is pinned.

    This model is the only task data passed into agent prompts. It deliberately has no gold
    patch, test patch, test ids, features, split, or audit fields (hard rule 3).
    """

    task_id: str
    repo: str
    base_commit: str
    issue_title: str
    issue_body: str
