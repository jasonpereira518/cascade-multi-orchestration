"""Trivial pipeline STUB used by the smoke test.

It runs a single triage call through the metered client. The real pipeline (triage, reproduce,
localize, patch, review, verify, bounded repair loop) replaces this later; the contract it already
honours is that agents receive only an AgentView and a workspace built from the base commit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from cascade.config import ArmSpec
from cascade.harness.ledger import CallContext
from cascade.harness.metered import MeteredClient
from cascade.harness.providers.base import LLMRequest, Message
from cascade.taskgen.schema import AgentView

_MAX_LISTED_FILES = 50


@dataclass(frozen=True)
class StubResult:
    task_id: str
    arm_id: str
    triage_text: str
    calls: int


def list_python_files(workspace: Path) -> list[str]:
    """Deterministic tool (no LLM): sorted relative paths of .py files, capped for prompt size."""
    files = sorted(p.relative_to(workspace).as_posix() for p in workspace.rglob("*.py"))
    return files[:_MAX_LISTED_FILES]


def call_id_for(
    run_id: str, task_id: str, arm_id: str, attempt_id: str, role: str, turn: int
) -> str:
    """Deterministic idempotency key: a resumed run reproduces the same ids."""
    raw = f"{run_id}|{task_id}|{arm_id}|{attempt_id}|{role}|{turn}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def run_stub_pipeline(
    view: AgentView,
    workspace: Path,
    client: MeteredClient,
    arm: ArmSpec,
    *,
    run_id: str,
    attempt_id: str = "attempt-0",
    max_output_tokens: int = 256,
) -> StubResult:
    files = "\n".join(list_python_files(workspace))
    request = LLMRequest(
        model=arm.model,
        reasoning_effort=arm.reasoning_effort,
        max_output_tokens=max_output_tokens,
        messages=[
            Message(role="system", content="You triage GitHub issues. Name the likely area."),
            Message(
                role="user",
                content=f"Issue: {view.issue_title}\n\n{view.issue_body}\n\nPython files:\n{files}",
            ),
        ],
    )
    ctx = CallContext(
        call_id=call_id_for(run_id, view.task_id, arm.arm_id, attempt_id, "triager", 0),
        run_id=run_id,
        task_id=view.task_id,
        attempt_id=attempt_id,
        arm_id=arm.arm_id,
        role="triager",
    )
    response = client.complete(ctx, request)
    return StubResult(view.task_id, arm.arm_id, response.text, calls=1)
