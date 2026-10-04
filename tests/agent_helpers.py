"""Helpers for the agent tests. Everything here is SYNTHETIC (see tests/fixtures/README.md).

``NativeExecutor`` runs pytest on the host. It exists only so tests can drive the whole pipeline
without Docker, against the tiny fixture repo and scripted (never real-model) edits. It still routes
every run through the real container runner's ``validate``, so the leak guard and mount rules that
Docker mode enforces are exercised here too. Real attempts always use the Docker executor.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.testing import ModelStep

from cascade.agents.pipeline import AgentPipeline, PipelineConfig
from cascade.agents.runner import ExecutorFactory, RunReport, run_attempt
from cascade.agents.schemas import (
    FileRef,
    LocalizationOutput,
    PatchOutput,
    ReproOutput,
    ReviewIssue,
    ReviewOutput,
    TriageOutput,
)
from cascade.config import ArmSpec, ModelsConfig, ResolvedBudget
from cascade.harness.ledger import Ledger
from cascade.harness.models import AttemptRef
from cascade.harness.providers.mock import MockModelSource, final_step, tool_call_step
from cascade.harness.sandbox import ContainerRunner, SandboxSpec, prepare_workspace
from cascade.harness.trace import TraceWriter
from cascade.routing.matrix import MatrixStore
from cascade.taskgen.results import RunResult, parse_report_full
from cascade.taskgen.schema import Task
from tests.conftest import ROOT, FixtureRepo

REPRO_PATH = "tests/test_cascade_repro.py"
REPRO_CODE = (
    "from tinylib import clamp\n\n\n"
    "def test_repro_value_above_range_is_limited_to_high():\n"
    "    assert clamp(99, 0, 10) == 10\n"
)
BROKEN_REPRO_CODE = (  # fails, but for the wrong reason: an import error
    "from tinylib import does_not_exist\n\n\ndef test_repro_broken():\n    assert does_not_exist()\n"
)
PASSING_REPRO_CODE = (  # proves nothing: passes on the unpatched code
    "from tinylib import clamp\n\n\ndef test_repro_passes_already():\n    assert clamp(5, 0, 10) == 5\n"
)
CLAMP = "tinylib/clamp.py"
BUGGY = "    if value > high:\n        return low\n"
FIXED = "    if value > high:\n        return high\n"
# Passes the reproducer (99) and every visible test, but fails the hidden test (42): a Verifier
# false positive, by construction.
OVERFIT = "    if value > high:\n        return high if value == 99 else low\n"
BREAKS_BELOW = "    if value < low:\n        return high\n"  # breaks the visible test_below_range


class NativeExecutor:
    """TEST-ONLY. Runs pytest on the host in the given workspace (never use for real models)."""

    __test__ = False

    def __init__(self, runner: ContainerRunner | None = None) -> None:
        self._runner = runner
        self.calls: list[tuple[Path, list[str]]] = []

    def run(self, *, workspace: Path, pytest_args: Sequence[str], timeout_s: float) -> RunResult:
        if self._runner is not None:  # same mount / leak checks the Docker path performs
            self._runner.validate(SandboxSpec(image="native-test", workspace=workspace))
        self.calls.append((workspace, list(pytest_args)))
        report = workspace / ".cascade_results.jsonl"
        report.unlink(missing_ok=True)
        env = {
            **os.environ,
            "PYTHONPATH": str(ROOT / "docker"),
            "CASCADE_REPORT_PATH": str(report),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        start = time.monotonic()
        proc = subprocess.run(
            [
                sys.executable, "-m", "pytest", "-p", "cascade_pytest_report",
                "--continue-on-collection-errors", "-p", "no:cacheprovider", "-q", "--tb=short",
                *pytest_args,
            ],
            cwd=workspace, env=env, capture_output=True, text=True, timeout=timeout_s, check=False,
        )  # fmt: skip
        outcomes: dict[str, str] = {}
        details: dict[str, str] = {}
        if report.exists():
            outcomes, details = parse_report_full(report.read_text())
            report.unlink()
        return RunResult(
            proc.returncode, time.monotonic() - start, False, outcomes,
            (proc.stdout + proc.stderr)[-800:], details,
        )  # fmt: skip


# --- scripted plans ----------------------------------------------------------------------------


def triage_step() -> ModelStep:
    return final_step(
        TriageOutput(
            summary="clamp returns the lower bound for values above the range",
            issue_type="bug",
            suspected_areas=["tinylib.clamp"],
            search_terms=["clamp", "high", "low"],
            difficulty=2,
            reproducible_with_unit_test=True,
        )
    )


def repro_steps(code: str = REPRO_CODE) -> list[ModelStep]:
    return [
        tool_call_step("write_reproducer", {"code": code}, call_id="repro-1"),
        tool_call_step("run_tests", {"test_paths": [REPRO_PATH]}, call_id="repro-2"),
        final_step(ReproOutput(summary="clamp(99, 0, 10) should be 10", test_name="test_repro",
                               expected_failure="assert 0 == 10")),
    ]  # fmt: skip


def localize_steps() -> list[ModelStep]:
    return [
        tool_call_step(
            "search_code", {"query": "clamp value high low", "top_k": 3}, call_id="loc-1"
        ),
        final_step(
            LocalizationOutput(
                files=[
                    FileRef(path=CLAMP, reason="defines clamp"),
                    FileRef(path="nope.py", reason="?"),
                ],
                functions=["clamp"],
                notes="the branch for values above high returns low",
            )
        ),
    ]


def patch_steps(new_branch: str = FIXED, *, call: str = "p") -> list[ModelStep]:
    return [
        tool_call_step("read_file", {"path": CLAMP, "start_line": 1, "end_line": 0}, call_id=f"{call}-1"),
        tool_call_step("edit_file", {"path": CLAMP, "old_text": BUGGY, "new_text": new_branch},
                       call_id=f"{call}-2"),
        final_step(PatchOutput(summary="return high for values above the range",
                               files_changed=[CLAMP], confidence="high")),
    ]  # fmt: skip


def noop_patch_steps() -> list[ModelStep]:
    """A patcher that changes nothing."""
    return [final_step(PatchOutput(summary="no change", files_changed=[], confidence="low"))]


def approve_step() -> ModelStep:
    return final_step(ReviewOutput(verdict="approve", issues=[], summary="looks right"))


def request_changes_step() -> ModelStep:
    return final_step(
        ReviewOutput(
            verdict="request_changes",
            issues=[ReviewIssue(severity="major", description="handle values below low too")],
            summary="needs another look",
        )
    )


def good_plans() -> dict[str, list[ModelStep]]:
    return {
        "triager": [triage_step()],
        "reproducer": repro_steps(),
        "localizer": localize_steps(),
        "patcher": patch_steps(),
        "reviewer": [approve_step()],
    }


def request_text(call: Any) -> str:
    """Everything a model call carried: system instructions plus the full input."""
    import json

    return (call.system_instructions or "") + "\n" + json.dumps(call.input, default=str)


# --- an attempt environment ----------------------------------------------------------------------


def pipeline_config(**over: Any) -> PipelineConfig:
    base: dict[str, Any] = {"max_output_tokens": 512, "max_repairs": 2, "test_timeout_s": 60.0}
    return PipelineConfig(**{**base, **over})


def native_factory(_runner: ContainerRunner) -> NativeExecutor:
    return NativeExecutor(_runner)


@dataclass
class AttemptEnv:
    tmp: Path
    fx: FixtureRepo
    models: ModelsConfig
    budget: ResolvedBudget
    ledger: Ledger
    matrix: MatrixStore
    runs_dir: Path
    executors: list[NativeExecutor] = field(default_factory=lambda: [])

    @property
    def task(self) -> Task:
        return self.fx.task

    @property
    def arm(self) -> ArmSpec:
        return self.models.arms[0]

    def source(
        self, plans: Mapping[str, Sequence[ModelStep]] | None = None, **kw: Any
    ) -> MockModelSource:
        return MockModelSource(dict(plans if plans is not None else good_plans()), **kw)

    def factory(self) -> ExecutorFactory:
        def make(runner: ContainerRunner) -> NativeExecutor:
            executor = NativeExecutor(runner)
            self.executors.append(executor)
            return executor

        return make

    def run(
        self,
        plans: Mapping[str, Sequence[ModelStep]] | None = None,
        *,
        source: MockModelSource | None = None,
        config: PipelineConfig | None = None,
        seed: int = 0,
        force: bool = False,
        arm: ArmSpec | None = None,
        executor_factory: ExecutorFactory | None = None,
    ) -> RunReport:
        return run_attempt(
            task=self.task,
            arm=arm or self.arm,
            seed=seed,
            config=config or pipeline_config(),
            ledger=self.ledger,
            matrix=self.matrix,
            source=source or self.source(plans),
            clone=self.fx.path,
            runs_dir=self.runs_dir,
            executor_factory=executor_factory or self.factory(),
            forbidden_host_paths=[],
            grading_timeout_s=60.0,
            force=force,
        )

    def pipeline(
        self,
        source: MockModelSource,
        *,
        config: PipelineConfig | None = None,
        executor: NativeExecutor | None = None,
        attempt_id: str = "attempt-0",
    ) -> tuple[AgentPipeline, Path, Path]:
        """A pipeline over fresh base/work exports, for tests that call it directly."""
        base = prepare_workspace(self.fx.path, self.task.base_commit, self.tmp / "p-base")
        work = prepare_workspace(self.fx.path, self.task.base_commit, self.tmp / "p-work")
        scratch = self.tmp / "p-scratch"
        scratch.mkdir(exist_ok=True)
        pipeline = AgentPipeline(
            source=source,
            arm=self.arm,
            ledger=self.ledger,
            executor=executor or NativeExecutor(),
            config=config or pipeline_config(),
            attempt=AttemptRef("run-x", self.task.task_id, attempt_id, self.arm.arm_id),
            trace=TraceWriter(self.tmp / "trace.jsonl"),
            scratch=scratch,
        )
        return pipeline, base, work


Plans = Callable[[], dict[str, list[ModelStep]]]
