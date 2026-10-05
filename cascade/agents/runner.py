"""Run one attempt end to end: workspaces, pipeline, hidden-test grading, matrix row.

Roles of the two kinds of check, kept apart on purpose:
  - verifier_claimed: the agent pipeline's own evidence (reproducer + visible suite). Produced
    inside the pipeline, which never sees hidden data.
  - true_success: graded here, after the pipeline is finished, by applying the agent's SOURCE patch
    to a fresh base export, adding the hidden test patch, and requiring every fail_to_pass and
    pass_to_pass test to pass. Nothing from grading flows back into any prompt.

Idempotent and resumable: a (task, arm, seed, config) with a matrix row is skipped unless forced.
An interrupted attempt leaves no row; rerunning it starts a new attempt id (fresh per-attempt cap),
while its earlier spend stays in the ledger and in the task and global totals.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from cascade.agents.audit import audit_trace, post_base_commits
from cascade.agents.patching import patch_sha
from cascade.agents.pipeline import AgentPipeline, AttemptOutcome, PipelineConfig
from cascade.agents.verifier import usable
from cascade.config import ArmSpec
from cascade.harness.grading import GradingRunner
from cascade.harness.ledger import DEFAULT_BUCKET, STANDARD_TIER, Ledger
from cascade.harness.models import AttemptRef
from cascade.harness.providers.base import ModelSource
from cascade.harness.sandbox import ContainerRunner, LeakGuard, SandboxRunner, prepare_workspace
from cascade.harness.trace import TraceWriter
from cascade.routing.matrix import MatrixRow, MatrixStore, config_id_of
from cascade.taskgen.build_env import DockerTestExecutor
from cascade.taskgen.gitrepo import PatchApplyError, apply_patch
from cascade.taskgen.results import PASSED
from cascade.taskgen.schema import Task
from cascade.taskgen.validate import TestExecutor

# Builds a test executor around a container runner (agent-side or grading).
ExecutorFactory = Callable[[ContainerRunner], TestExecutor]


def docker_executor_factory(task: Task, test_cmd: str) -> ExecutorFactory:
    return lambda runner: DockerTestExecutor(task.image, runner, test_cmd)


@dataclass(frozen=True)
class GradeResult:
    true_success: bool
    reason: str  # "passed" or why not
    failed: list[str]  # fail_to_pass / pass_to_pass tests that did not pass (capped)


@dataclass(frozen=True)
class RunReport:
    row: MatrixRow
    outcome: AttemptOutcome | None  # None when an existing row was reused
    grade: GradeResult | None
    leak_violations: list[str]
    attempt_dir: Path | None
    reused: bool


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def grade_patch(
    *,
    task: Task,
    patch: str,
    clone: Path,
    executor: TestExecutor,
    workdir: Path,
    timeout_s: float,
) -> GradeResult:
    """True success: all hidden fail_to_pass and pass_to_pass tests pass with the agent's patch."""
    if not patch.strip():
        return GradeResult(False, "empty_patch", [])
    workspace = workdir / f"grade-{uuid.uuid4().hex[:8]}"
    try:
        prepare_workspace(clone, task.base_commit, workspace)
        try:
            apply_patch(workspace, patch)
        except PatchApplyError:
            return GradeResult(False, "patch_apply_failed", [])
        try:
            apply_patch(workspace, task.test_patch)
        except PatchApplyError:
            return GradeResult(False, "test_patch_conflict", [])
        run = executor.run(workspace=workspace, pytest_args=[], timeout_s=timeout_s)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    if run.timed_out:
        return GradeResult(False, "grading_timeout", [])
    if not usable(run):
        return GradeResult(False, "grading_run_error", [])
    required = [*task.fail_to_pass, *task.pass_to_pass]
    failed = [t for t in required if run.outcomes.get(t) != PASSED]
    if failed:
        return GradeResult(False, "tests_failed", failed[:20])
    return GradeResult(True, "passed", [])


def next_attempt_id(ledger: Ledger, task_id: str, base: str) -> str:
    """``base`` plus a try counter, so a retry gets a fresh per-attempt cap."""
    tries = [
        a for a in ledger.attempt_ids(task_id=task_id) if a == base or a.startswith(base + "|try")
    ]
    return base if not tries else f"{base}|try{len(tries)}"


def run_attempt(
    *,
    task: Task,
    arm: ArmSpec,
    seed: int,
    config: PipelineConfig,
    ledger: Ledger,
    matrix: MatrixStore,
    source: ModelSource,
    clone: Path,
    runs_dir: Path,
    executor_factory: ExecutorFactory,
    forbidden_host_paths: Sequence[Path] = (),
    grading_timeout_s: float = 120.0,
    force: bool = False,
    bucket: str = DEFAULT_BUCKET,
    tier: str = STANDARD_TIER,
) -> RunReport:
    cfg_json = config.canonical()
    cfg_id = config_id_of(cfg_json)
    existing = matrix.get(task.task_id, arm.arm_id, seed, cfg_id)
    if existing is not None and not force:
        return RunReport(existing, None, None, [], None, reused=True)

    attempt_dir = (
        runs_dir / "attempts" / _safe(task.task_id) / _safe(arm.arm_id) / f"seed{seed}-{cfg_id}"
    )
    shutil.rmtree(attempt_dir, ignore_errors=True)  # a previous incomplete attempt left no row
    base_dir = prepare_workspace(clone, task.base_commit, attempt_dir / "base")
    work_dir = prepare_workspace(clone, task.base_commit, attempt_dir / "work")
    scratch = attempt_dir / "scratch"
    scratch.mkdir()

    base_id = f"{arm.arm_id}|seed{seed}|{cfg_id}"
    attempt = AttemptRef(
        run_id=uuid.uuid4().hex[:12],
        task_id=task.task_id,
        attempt_id=next_attempt_id(ledger, task.task_id, base_id),
        arm_id=arm.arm_id,
        bucket=bucket,
        tier=tier,
    )
    trace = TraceWriter(attempt_dir / "trace.jsonl")
    guard = LeakGuard.from_task(task, host_paths=forbidden_host_paths)
    agent_executor = executor_factory(SandboxRunner(guard, sandbox_root=attempt_dir))
    grading_executor = executor_factory(GradingRunner(sandbox_root=attempt_dir))

    pipeline = AgentPipeline(
        source=source, arm=arm, ledger=ledger, executor=agent_executor, config=config,
        attempt=attempt, trace=trace, scratch=scratch,
    )  # fmt: skip
    started_at = dt.datetime.now(dt.UTC).isoformat()
    started = time.monotonic()
    outcome = pipeline.run(task.agent_view(), base_dir, work_dir)
    wall_s = time.monotonic() - started

    grade = grade_patch(
        task=task, patch=outcome.src_patch, clone=clone, executor=grading_executor,
        workdir=attempt_dir, timeout_s=grading_timeout_s,
    )  # fmt: skip
    leak_violations = audit_trace(
        trace.text(), task, post_base_shas=post_base_commits(clone, task.base_commit)
    )

    totals = ledger.attempt_totals(task_id=task.task_id, attempt_id=attempt.attempt_id)
    row = MatrixRow(
        task_id=task.task_id,
        arm_id=arm.arm_id,
        seed=seed,
        true_success=grade.true_success,
        verifier_claimed=outcome.claimed,
        cost_usd=totals.cost_usd,
        input_tokens=totals.input_tokens,
        output_tokens=totals.output_tokens,
        reasoning_tokens=totals.reasoning_tokens,
        wall_s=wall_s,
        turns=totals.calls,
        patch_sha=patch_sha(outcome.src_patch),
        pipeline_config=cfg_json,
        error=outcome.error,
        model=arm.model,
        reasoning_effort=arm.reasoning_effort,
        processing_tier=tier,
        cached_tokens=totals.cached_tokens,
        cache_write_tokens=totals.cache_write_tokens,
        started_at=started_at,
    )
    (attempt_dir / "patch.diff").write_text(outcome.src_patch, encoding="utf-8")
    (attempt_dir / "result.json").write_text(
        json.dumps(
            {
                "attempt_id": attempt.attempt_id,
                "row": row.model_dump(),
                "grade": {"reason": grade.reason, "failed": grade.failed},
                "verifier": None if outcome.report is None else outcome.report.model_dump(),
                "stages": [s.model_dump() for s in outcome.stages],
                "rounds": outcome.rounds,
                "review_verdicts": outcome.review_verdicts,
                "repro_valid": outcome.repro_valid,
                "leak_violations": leak_violations,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    matrix.put(row)  # written last: a row exists only for a fully finished attempt
    return RunReport(row, outcome, grade, leak_violations, attempt_dir, reused=False)
