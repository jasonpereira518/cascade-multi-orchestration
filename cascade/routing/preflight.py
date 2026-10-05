"""Pre-flight leakage check over a real manifest, one result per audited task. No spend.

tests/test_leakage.py proves the machinery on one synthetic task. This runs the same checks on every
real task before any model sees it, because a leak is a property of the task's own repo and issue:

  workspace  the base-commit export the agent works in: no ``.git``, no file identical to or named
             like a gold/test patch artifact (``LeakGuard``), no hidden test name, no commit hash
             from after the base commit, and no verbatim gold or test patch in any file;
  prompts    every role's system instructions and every input builder, rendered from the task's
             ``AgentView`` (the only task data prompts can receive), audited for the same things;
  view       ``AgentView`` still has exactly the allowed fields.

A hit is reported, not auto-fixed: a hidden test name in an issue body, for example, may be a
harmless coincidence or a real leak, and that is a human call (it is also what the audit's
``leaks_fix`` verdict is for). The network-off and read-only properties are Docker-level and are
covered by tests/test_sandbox.py and tests/test_grading.py, not here.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from cascade.agents import prompts
from cascade.agents.audit import audit_text, post_base_commits
from cascade.agents.schemas import TriageOutput
from cascade.errors import CascadeError
from cascade.harness.sandbox import LeakGuard, prepare_workspace
from cascade.taskgen.schema import AgentView, Task

ALLOWED_VIEW_FIELDS = frozenset({"task_id", "repo", "base_commit", "issue_title", "issue_body"})
_MAX_FILE_BYTES = 2_000_000
_REPRO_PATH = "tests/test_cascade_repro.py"


@dataclass(frozen=True)
class TaskPreflight:
    task_id: str
    violations: list[str] = field(default_factory=lambda: [])
    checks: int = 0  # how many distinct checks ran, so "0 violations" is never "0 checks"

    @property
    def ok(self) -> bool:
        return not self.violations and self.checks > 0


def _placeholder_triage() -> TriageOutput:
    return TriageOutput(
        summary="placeholder", issue_type="other", suspected_areas=[], search_terms=[],
        difficulty=3, reproducible_with_unit_test=True,
    )  # fmt: skip


def rendered_prompts(view: AgentView) -> dict[str, str]:
    """Every agent-visible prompt for a task: instructions plus each role's input text."""
    triage = _placeholder_triage()
    out: dict[str, str] = {f"instructions:{k}": v for k, v in prompts.INSTRUCTIONS.items()}
    out["input:triager"] = prompts.triage_input(view)
    out["input:reproducer"] = prompts.repro_input(view, triage, _REPRO_PATH)
    out["input:localizer"] = prompts.localize_input(view, triage)
    out["input:patcher"] = prompts.patch_input(view, triage, None, None, None, _REPRO_PATH, None)
    return out


def _workspace_text(root: Path) -> str:
    parts: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                continue
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(parts)


def check_task(task: Task, clone: Path, *, scratch: Path) -> TaskPreflight:
    violations: list[str] = []
    checks = 1
    fields = frozenset(AgentView.model_fields)
    if fields != ALLOWED_VIEW_FIELDS:
        violations.append(f"view: AgentView fields are {sorted(fields)}, not the allowed set")

    post_base = post_base_commits(clone, task.base_commit)
    workspace = scratch / "workspace"
    try:
        prepare_workspace(clone, task.base_commit, workspace)
    except CascadeError as e:
        violations.append(f"workspace: cannot export the base commit: {e}")
        return TaskPreflight(task.task_id, violations, checks)
    try:
        checks += 2
        violations += [f"workspace: {v}" for v in LeakGuard.from_task(task).scan(workspace)]
        text = _workspace_text(workspace)
        violations += [f"workspace: {v}" for v in audit_text(text, task, post_base_shas=post_base)]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    for name, text in rendered_prompts(task.agent_view()).items():
        checks += 1
        found = audit_text(text, task, post_base_shas=post_base)
        violations += [f"prompt {name}: {v}" for v in found]
    return TaskPreflight(task.task_id, violations, checks)


def run_preflight(
    tasks: Sequence[Task], clone_for: Callable[[Task], Path | None]
) -> list[TaskPreflight]:
    """One result per task, in the given order. A task whose clone is missing fails, not skips."""
    results: list[TaskPreflight] = []
    for task in tasks:
        clone = clone_for(task)
        if clone is None or not clone.exists():
            results.append(TaskPreflight(task.task_id, [f"no clone available for {task.repo}"]))
            continue
        scratch = Path(tempfile.mkdtemp(prefix="cascade-preflight-"))
        try:
            results.append(check_task(task, clone, scratch=scratch))
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    return results


def format_preflight(results: Sequence[TaskPreflight]) -> str:
    bad = [r for r in results if not r.ok]
    lines = [f"PREFLIGHT: {len(results)} audited task(s) checked, {len(bad)} failed"]
    for r in bad:
        lines.append(f"- {r.task_id}")
        lines += [f"    {v}" for v in (r.violations or ["no checks ran"])]
    if not bad and results:
        lines.append("all checks passed: no gold patch, test patch, hidden test name, or post-base")
        lines.append("commit is reachable in any workspace or prompt")
    return "\n".join(lines)
