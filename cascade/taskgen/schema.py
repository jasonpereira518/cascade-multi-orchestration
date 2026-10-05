"""Task schema (docs/SPEC.md) and the restricted view agents are allowed to see."""

from __future__ import annotations

import datetime as dt
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SHA40 = re.compile(r"^[0-9a-f]{40}$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


Count = Annotated[int, Field(ge=0)]


class Features(_Strict):
    """Task features. The first four are the SPEC's; the rest are model-free difficulty features
    computed by ``taskgen.features`` from the patches, issue text, and test lists (no LLM, no repo
    access). They default to zero so manifests written before they existed still load."""

    src_files: Count
    src_lines: Count
    issue_chars: Count
    test_runtime_s: Annotated[float, Field(ge=0)]
    src_hunks: Count = 0
    src_added: Count = 0
    src_removed: Count = 0
    gold_files: Count = 0
    test_files: Count = 0
    test_added: Count = 0
    f2p_count: Count = 0
    p2p_count: Count = 0
    issue_words: Count = 0
    issue_code_blocks: Count = 0
    issue_has_traceback: bool = False
    issue_identifiers: Count = 0
    changed_symbols: Count = 0
    changed_symbols_in_issue: Count = 0
    new_public_symbols: Count = 0


AuditStatus = Literal["pending", "ok", "underspecified", "leaks_fix", "arbitrary_test"]


class Audit(_Strict):
    """Human audit result. The verdict log (tasks/audit.jsonl) is the source of truth; this field is
    filled from it by `cascade tasks finalize`. Only status "ok" tasks enter experiments."""

    status: AuditStatus = "pending"
    notes: str = ""


class MinedWindow(_Strict):
    """The date window a task was mined under (recorded so windows are never mixed)."""

    start: dt.date
    end: dt.date
    kind: Literal["standard", "relaxed"]


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
    # Provenance, all optional so older manifests still load. Never shown to agents (AgentView).
    mined_window: MinedWindow | None = None
    merged_at: dt.datetime | None = None
    # arm id -> True if the PR was merged AFTER that arm's knowledge cutoff, False if not, None if
    # the cutoff is not filled in models.yaml. Restamped by `cascade tasks finalize`.
    cutoff_clearance: dict[str, bool | None] = Field(default_factory=lambda: {})

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
