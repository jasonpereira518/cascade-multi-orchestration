"""run_attempt end to end: workspaces, pipeline, hidden-test grading, ledger-derived matrix row."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from cascade.agents.pipeline import AgentPipeline
from cascade.agents.runner import GradeResult, grade_patch, next_attempt_id
from cascade.agents.schemas import PatchOutput
from cascade.errors import SpendNotAllowedError
from cascade.harness.providers.mock import MockModelSource, final_step
from cascade.taskgen.results import PASSED, RunResult
from cascade.taskgen.validate import TestExecutor
from tests.agent_helpers import (
    BUGGY,
    CLAMP,
    FIXED,
    OVERFIT,
    AttemptEnv,
    NativeExecutor,
    approve_step,
    good_plans,
    noop_patch_steps,
    patch_steps,
    repro_steps,
)
from tests.conftest import FixtureRepo


def ledger_cost(env: AttemptEnv, attempt_id: str) -> float:
    return sum(r.cost_usd for r in env.ledger.rows() if r.attempt_id == attempt_id)


def good_patch_text(fx: FixtureRepo) -> str:
    return fx.task.gold_patch


def test_a_correct_attempt_records_both_outcomes_and_ledger_derived_numbers(
    attempt_env: AttemptEnv,
) -> None:
    report = attempt_env.run()
    row = report.row
    assert row.true_success and row.verifier_claimed and row.error is None
    assert (
        report.grade is not None
        and report.grade.reason == "passed"
        and report.leak_violations == []
    )

    rows = attempt_env.ledger.rows()
    attempt_id = rows[0].attempt_id
    assert row.turns == len(rows) == 10  # one ledger row per model call
    assert row.cost_usd == pytest.approx(ledger_cost(attempt_env, attempt_id)) and row.cost_usd > 0
    assert (row.input_tokens, row.output_tokens, row.reasoning_tokens) == (
        sum(r.input_tokens for r in rows), sum(r.output_tokens for r in rows),
        sum(r.reasoning_tokens for r in rows),
    )  # fmt: skip
    assert row.patch_sha is not None and row.wall_s > 0
    assert attempt_env.matrix.rows() == [row]  # written last, exactly once


def test_attempt_artifacts_are_written(attempt_env: AttemptEnv) -> None:
    report = attempt_env.run()
    d = report.attempt_dir
    assert d is not None
    assert (d / "patch.diff").read_text() == report.outcome.src_patch  # type: ignore[union-attr]
    data = json.loads((d / "result.json").read_text())
    assert data["row"]["true_success"] is True and data["grade"]["reason"] == "passed"
    assert data["verifier"]["claimed"] is True and data["leak_violations"] == []
    assert [s["stage"] for s in data["stages"]][:2] == ["triager", "reproducer"]
    assert (d / "trace.jsonl").read_text().count('"kind": "llm"') == 10


def test_rerunning_is_idempotent_and_makes_no_new_calls(attempt_env: AttemptEnv) -> None:
    first = attempt_env.run()
    source = attempt_env.source()
    again = attempt_env.run(source=source)
    assert again.reused and again.row == first.row and again.outcome is None
    assert source.models == {} and len(attempt_env.ledger.rows()) == 10  # nothing new was called


def test_force_reruns_with_a_fresh_attempt_id_and_replaces_the_row(attempt_env: AttemptEnv) -> None:
    attempt_env.run()
    forced = attempt_env.run(force=True)
    assert not forced.reused and len(attempt_env.matrix.rows()) == 1
    ids = attempt_env.ledger.attempt_ids(task_id=attempt_env.task.task_id)
    assert len(ids) == 2 and ids[1].endswith("|try1")
    # the retry's cost counts only its own calls, while task and global totals include both
    assert forced.row.turns == 10 and attempt_env.ledger.spent_usd() == pytest.approx(
        sum(r.cost_usd for r in attempt_env.ledger.rows())
    )


def test_different_seeds_arms_and_configs_are_separate_rows(attempt_env: AttemptEnv) -> None:
    from tests.agent_helpers import pipeline_config

    attempt_env.run(seed=0)
    attempt_env.run(seed=1)
    attempt_env.run(config=pipeline_config(reviewer=False))
    attempt_env.run(arm=attempt_env.models.arms[1])
    assert len(attempt_env.matrix.rows()) == 4


def test_a_wrong_patch_is_neither_claimed_nor_true(attempt_env: AttemptEnv) -> None:
    plans = good_plans()
    plans["patcher"] = noop_patch_steps() * 5
    report = attempt_env.run(plans)
    assert not report.row.true_success and not report.row.verifier_claimed
    assert report.grade is not None and report.grade.reason == "empty_patch"
    assert report.row.patch_sha is None


def test_an_overfit_patch_is_a_verifier_false_positive(attempt_env: AttemptEnv) -> None:
    """Claimed by the agent's own evidence, but fails the hidden test: the case the matrix exists for."""
    plans = good_plans()
    plans["patcher"] = patch_steps(OVERFIT)
    report = attempt_env.run(plans)
    assert report.row.verifier_claimed and not report.row.true_success
    assert report.grade is not None and report.grade.reason == "tests_failed"
    assert report.grade.failed == ["tests/test_clamp.py::test_above_range"]


def test_a_correct_patch_behind_a_broken_reproducer_is_a_verifier_false_negative(
    attempt_env: AttemptEnv,
) -> None:
    from tests.agent_helpers import BROKEN_REPRO_CODE

    plans = good_plans()
    plans["reproducer"] = repro_steps(BROKEN_REPRO_CODE)
    plans["patcher"] = patch_steps() * 3
    report = attempt_env.run(plans)
    assert report.row.true_success and not report.row.verifier_claimed


def test_a_reviewer_less_reproducer_less_attempt_is_recorded_under_its_own_config(
    attempt_env: AttemptEnv,
) -> None:
    from tests.agent_helpers import pipeline_config

    report = attempt_env.run(config=pipeline_config(reproducer=False, reviewer=False))
    assert report.row.true_success and report.row.verifier_claimed and report.row.turns == 6
    assert json.loads(report.row.pipeline_config)["reproducer"] is False


def test_an_interrupted_attempt_leaves_no_row_and_can_be_rerun(attempt_env: AttemptEnv) -> None:
    class Real(MockModelSource):
        is_real = True  # trips the spend gate on the first call

    with pytest.raises(SpendNotAllowedError):
        attempt_env.run(source=Real(good_plans()))
    assert attempt_env.matrix.rows() == []
    ok = attempt_env.run()
    assert ok.row.true_success and len(attempt_env.matrix.rows()) == 1


def test_a_kill_switched_attempt_records_the_stop_reason(attempt_env: AttemptEnv) -> None:
    attempt_env.budget.kill_switch_path.touch()
    report = attempt_env.run()
    assert report.row.error is not None and "KillSwitchEngagedError" in report.row.error
    assert not report.row.true_success and not report.row.verifier_claimed and report.row.turns == 0
    assert report.row.cost_usd == 0.0


def test_the_leak_audit_flags_a_trace_that_contains_hidden_material(
    attempt_env: AttemptEnv,
) -> None:
    """Positive control: if an agent's output echoed hidden test content, the runner says so."""
    plans = good_plans()
    leaked = "see test_above_range in the hidden suite"
    plans["patcher"] = [
        *patch_steps()[:-1],
        final_step(PatchOutput(summary=leaked, files_changed=[CLAMP], confidence="high")),
    ]
    plans["reviewer"] = [approve_step()]
    report = attempt_env.run(plans)
    assert any("test_above_range" in v for v in report.leak_violations)
    data = json.loads((report.attempt_dir / "result.json").read_text())  # type: ignore[union-attr]
    assert data["leak_violations"] == report.leak_violations


# --- grade_patch ---------------------------------------------------------------------------


class Canned:
    def __init__(self, result: RunResult) -> None:
        self.result = result

    def run(self, *, workspace: Path, pytest_args: Sequence[str], timeout_s: float) -> RunResult:
        return self.result


def grade(env: AttemptEnv, patch: str, executor: TestExecutor, tmp: Path) -> GradeResult:
    return grade_patch(
        task=env.task, patch=patch, clone=env.fx.path, executor=executor, workdir=tmp, timeout_s=60
    )


def test_grading_with_the_real_gold_patch_passes_and_leaves_no_workspace(
    attempt_env: AttemptEnv, tmp_path: Path
) -> None:
    work = tmp_path / "grade-work"
    work.mkdir()
    result = grade(attempt_env, attempt_env.task.gold_patch, NativeExecutor(), work)
    assert result.true_success and result.reason == "passed"
    assert list(work.iterdir()) == []  # the grading workspace is cleaned up


def test_grading_rejects_empty_and_non_applying_and_conflicting_patches(
    attempt_env: AttemptEnv, tmp_path: Path
) -> None:
    ok = Canned(RunResult(0, 1.0, False, {}, ""))
    assert grade(attempt_env, "  \n", ok, tmp_path).reason == "empty_patch"
    bogus = "diff --git a/nope.py b/nope.py\n--- a/nope.py\n+++ b/nope.py\n@@ -1 +1 @@\n-x\n+y\n"
    assert grade(attempt_env, bogus, ok, tmp_path).reason == "patch_apply_failed"
    # A patch that edits a test file the hidden test patch also edits cannot be combined with it.
    clash = (
        "diff --git a/tests/test_clamp.py b/tests/test_clamp.py\n--- a/tests/test_clamp.py\n"
        "+++ b/tests/test_clamp.py\n@@ -7,2 +7,2 @@\n def test_below_range():\n-    assert clamp(-3, 0, 10) == 0\n+    assert True\n"
    )
    assert grade(attempt_env, clash, ok, tmp_path).reason == "test_patch_conflict"


def test_grading_requires_every_hidden_test_to_pass(
    attempt_env: AttemptEnv, tmp_path: Path
) -> None:
    t = attempt_env.task
    all_pass = {n: PASSED for n in [*t.fail_to_pass, *t.pass_to_pass]}
    patch = t.gold_patch
    assert grade(
        attempt_env, patch, Canned(RunResult(0, 1.0, False, all_pass, "")), tmp_path
    ).true_success

    one_pass_fails = {**all_pass, t.pass_to_pass[0]: "failed"}
    r = grade(attempt_env, patch, Canned(RunResult(1, 1.0, False, one_pass_fails, "")), tmp_path)
    assert not r.true_success and r.failed == [t.pass_to_pass[0]]

    missing = {k: v for k, v in all_pass.items() if k != t.fail_to_pass[0]}
    r = grade(attempt_env, patch, Canned(RunResult(1, 1.0, False, missing, "")), tmp_path)
    assert not r.true_success and r.failed == [t.fail_to_pass[0]]


def test_grading_distinguishes_timeouts_and_unusable_runs(
    attempt_env: AttemptEnv, tmp_path: Path
) -> None:
    patch = attempt_env.task.gold_patch
    assert (
        grade(attempt_env, patch, Canned(RunResult(None, 60.0, True, {}, "")), tmp_path).reason
        == "grading_timeout"
    )
    assert (
        grade(attempt_env, patch, Canned(RunResult(2, 1.0, False, {}, "")), tmp_path).reason
        == "grading_run_error"
    )


def test_attempt_ids_get_a_try_counter_only_when_needed(attempt_env: AttemptEnv) -> None:
    from cascade.harness.ledger import CallContext
    from cascade.harness.providers.base import Usage

    base = "arm|seed0|cfg"
    assert next_attempt_id(attempt_env.ledger, "t1", base) == base
    for i, attempt in enumerate([base, f"{base}|try1", "unrelated"]):
        attempt_env.ledger.record(
            CallContext(f"c{i}", "r", "t1", attempt, "arm", "triager"), provider="mock",
            model=attempt_env.models.models[0].name, usage=Usage(input_tokens=1, output_tokens=1),
            latency_s=0.0,
        )  # fmt: skip
    assert next_attempt_id(attempt_env.ledger, "t1", base) == f"{base}|try2"


def test_the_pipeline_class_is_what_the_runner_drives(attempt_env: AttemptEnv) -> None:
    assert AgentPipeline.__module__ == "cascade.agents.pipeline" and BUGGY != FIXED
