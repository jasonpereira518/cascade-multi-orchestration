"""The agent pipeline end to end on the synthetic fixture task (mock models, real pytest runs).

These tests use the fixture repo, not a real validated task: no live task pipeline run exists yet.
"""

from __future__ import annotations

import pytest
from agents.testing import ModelStep

from cascade.agents.pipeline import PipelineConfig
from cascade.agents.schemas import TriageOutput
from cascade.agents.tools import REPRO_PATH
from cascade.config import ResolvedBudget
from cascade.errors import SpendNotAllowedError
from cascade.harness.ledger import Ledger
from cascade.harness.providers.base import ModelSource
from cascade.harness.providers.mock import MockModelSource, final_step, sdk_usage
from cascade.taskgen.schema import Task
from tests.agent_helpers import (
    BROKEN_REPRO_CODE,
    FIXED,
    AttemptEnv,
    NativeExecutor,
    approve_step,
    good_plans,
    noop_patch_steps,
    patch_steps,
    pipeline_config,
    repro_steps,
    request_changes_step,
    request_text,
)


def run_pipeline(env: AttemptEnv, source: MockModelSource, config: PipelineConfig | None = None):
    pipeline, base, work = env.pipeline(source, config=config, executor=NativeExecutor())
    return pipeline.run(env.task.agent_view(), base, work), work


def stage_names(outcome) -> list[str]:
    return [s.stage for s in outcome.stages]


def test_the_full_pipeline_runs_every_role_and_claims_a_correct_patch(
    attempt_env: AttemptEnv,
) -> None:
    source = attempt_env.source()
    out, work = run_pipeline(attempt_env, source)
    assert out.error is None and out.claimed and out.rounds == 1
    assert out.repro_valid and out.review_verdicts == ["approve"]
    assert stage_names(out) == [
        "triager", "reproducer", "repro_check", "localizer", "patcher", "verifier", "reviewer",
    ]  # fmt: skip
    assert all(s.ok for s in out.stages)
    assert "return high" in out.src_patch and "tests/" not in out.src_patch  # source-only patch
    assert REPRO_PATH in out.full_patch  # the reproducer is in the full diff but not the graded one
    assert out.triage is not None and out.triage.difficulty == 2
    assert (work / REPRO_PATH).is_file()


def test_every_model_call_is_in_the_ledger_with_its_role(attempt_env: AttemptEnv) -> None:
    source = attempt_env.source()
    run_pipeline(attempt_env, source)
    rows = attempt_env.ledger.rows()
    by_role = {r: sum(1 for x in rows if x.role == r) for r in {x.role for x in rows}}
    for role in ("triager", "reproducer", "localizer", "patcher", "reviewer"):
        assert by_role[role] == len(source.calls(role)) > 0  # ledger rows == model calls
    assert all(r.status == "ok" and r.cost_usd > 0 for r in rows)


def test_the_reproducer_can_be_switched_off(attempt_env: AttemptEnv) -> None:
    source = attempt_env.source()
    out, work = run_pipeline(attempt_env, source, pipeline_config(reproducer=False))
    assert source.calls("reproducer") == ()
    assert "reproducer" not in stage_names(out) and "repro_check" not in stage_names(out)
    assert out.report is not None and not out.report.repro_enabled and out.claimed
    assert not (work / REPRO_PATH).exists()


def test_the_reviewer_can_be_switched_off(attempt_env: AttemptEnv) -> None:
    source = attempt_env.source()
    out, _ = run_pipeline(attempt_env, source, pipeline_config(reviewer=False))
    assert source.calls("reviewer") == () and out.review_verdicts == [] and out.claimed


def test_both_toggles_off_leaves_triage_localize_patch_verify(attempt_env: AttemptEnv) -> None:
    source = attempt_env.source()
    out, _ = run_pipeline(attempt_env, source, pipeline_config(reproducer=False, reviewer=False))
    assert stage_names(out) == ["triager", "localizer", "patcher", "verifier"] and out.claimed


def test_a_failed_check_sends_feedback_to_the_patcher_and_a_second_round_fixes_it(
    attempt_env: AttemptEnv,
) -> None:
    plans = good_plans()
    plans["patcher"] = noop_patch_steps() + patch_steps(call="q")
    source = attempt_env.source(plans)
    out, _ = run_pipeline(attempt_env, source)
    assert out.rounds == 2 and out.claimed
    second = request_text(source.calls("patcher")[-1] if len(source.calls("patcher")) else None)
    assert "Feedback on your previous attempt" in second
    assert "the patch is empty" in second  # the Verifier's reason reached the model
    verdicts = [s for s in out.stages if s.stage == "verifier"]
    assert [s.ok for s in verdicts] == [False, True]


def test_repair_is_bounded_by_max_repairs(attempt_env: AttemptEnv) -> None:
    plans = good_plans()
    plans["patcher"] = noop_patch_steps() * 10  # never fixes anything
    source = attempt_env.source(plans)
    out, _ = run_pipeline(attempt_env, source, pipeline_config(max_repairs=2))
    assert out.rounds == 3 and not out.claimed and out.error is None
    assert source.calls("reviewer") == ()  # nothing verified, so no review was ever requested
    assert len(source.calls("patcher")) == 3


def test_max_repairs_zero_means_a_single_patch_round(attempt_env: AttemptEnv) -> None:
    plans = good_plans()
    plans["patcher"] = noop_patch_steps() * 5
    out, _ = run_pipeline(attempt_env, attempt_env.source(plans), pipeline_config(max_repairs=0))
    assert out.rounds == 1 and not out.claimed


def test_reviewer_change_requests_trigger_a_repair_round_with_the_review_text(
    attempt_env: AttemptEnv,
) -> None:
    plans = good_plans()
    plans["patcher"] = patch_steps() + noop_patch_steps()  # round 2 changes nothing more
    plans["reviewer"] = [request_changes_step(), approve_step()]
    source = attempt_env.source(plans)
    out, _ = run_pipeline(attempt_env, source)
    assert out.rounds == 2 and out.review_verdicts == ["request_changes", "approve"] and out.claimed
    patcher_calls = source.calls("patcher")
    assert "A reviewer requested changes" in request_text(patcher_calls[-1])
    assert "handle values below low too" in request_text(patcher_calls[-1])


def test_change_requests_on_the_last_round_do_not_loop_forever(attempt_env: AttemptEnv) -> None:
    plans = good_plans()
    plans["patcher"] = patch_steps() + noop_patch_steps() * 5
    plans["reviewer"] = [request_changes_step()] * 10
    source = attempt_env.source(plans)
    out, _ = run_pipeline(attempt_env, source, pipeline_config(max_repairs=1))
    assert out.rounds == 2 and len(source.calls("reviewer")) == 2
    assert out.claimed  # the Verifier's evidence stands; the reviewer only drives repair


def test_a_reproducer_that_fails_for_the_wrong_reason_is_not_trusted(
    attempt_env: AttemptEnv,
) -> None:
    plans = good_plans()
    plans["reproducer"] = repro_steps(BROKEN_REPRO_CODE)
    out, _ = run_pipeline(attempt_env, attempt_env.source(plans))
    assert not out.repro_valid and not out.claimed
    assert out.report is not None
    assert any("did not fail on the unpatched code" in r for r in out.report.reasons)
    check = next(s for s in out.stages if s.stage == "repro_check")
    assert not check.ok


def test_no_reproducer_file_means_no_valid_reproduction(attempt_env: AttemptEnv) -> None:
    plans = good_plans()
    plans["reproducer"] = []  # falls through to the default: a final answer, no file written
    out, _ = run_pipeline(attempt_env, attempt_env.source(plans))
    assert not out.repro_valid and not out.claimed
    assert (
        next(s for s in out.stages if s.stage == "repro_check").detail
        == "no reproducer file was written"
    )


def test_a_role_that_returns_garbage_is_survivable(attempt_env: AttemptEnv) -> None:
    plans = good_plans()
    plans["triager"] = [final_step("this is not json at all")]
    out, _ = run_pipeline(attempt_env, attempt_env.source(plans))
    triage_stage = out.stages[0]
    assert triage_stage.stage == "triager" and not triage_stage.ok
    assert out.triage is not None and out.triage.issue_type == "other"  # fallback triage
    assert out.claimed  # the rest of the pipeline carried on


def test_difficulty_is_clamped_to_the_documented_range(attempt_env: AttemptEnv) -> None:
    plans = good_plans()
    plans["triager"] = [
        final_step(
            TriageOutput(
                summary="s",
                issue_type="bug",
                suspected_areas=[],
                search_terms=[],
                difficulty=99,
                reproducible_with_unit_test=True,
            )
        )
    ]
    out, _ = run_pipeline(attempt_env, attempt_env.source(plans))
    assert out.triage is not None and out.triage.difficulty == 5


def test_the_pipeline_only_accepts_an_agent_view(attempt_env: AttemptEnv) -> None:
    pipeline, base, work = attempt_env.pipeline(attempt_env.source())
    task: Task = attempt_env.task
    with pytest.raises(TypeError, match="AgentView"):
        pipeline.run(task, base, work)  # type: ignore[arg-type]


# --- caps, kill switch, spend gate ---------------------------------------------------------


def tight_env(attempt_env: AttemptEnv, **budget_changes: float | int) -> AttemptEnv:
    """The same environment, but with a ledger built on a modified budget."""
    attempt_env.ledger.close()
    budget: ResolvedBudget = attempt_env.budget.model_copy(update=budget_changes)
    attempt_env.ledger = Ledger(
        attempt_env.tmp / "tight.sqlite", budget=budget, models=attempt_env.models
    )
    attempt_env.budget = budget
    return attempt_env


def test_the_attempt_token_cap_stops_model_calls_but_the_attempt_is_still_finalized(
    attempt_env: AttemptEnv,
) -> None:
    env = tight_env(attempt_env, per_attempt_max_tokens=3000)
    source = env.source()
    out, _ = run_pipeline(env, source)
    assert out.error is not None and "BudgetExceededError" in out.error and "token cap" in out.error
    failed = [s for s in out.stages if not s.ok]
    assert failed and "BudgetExceededError" in failed[-1].detail
    calls_made = sum(
        len(source.calls(r)) for r in ("triager", "reproducer", "localizer", "patcher", "reviewer")
    )
    assert calls_made == len(env.ledger.rows())  # nothing was sent that the ledger did not log
    used = env.ledger.tokens_used(task_id=env.task.task_id, attempt_id="attempt-0")
    assert used <= 3000  # the cap held


def test_the_dollar_cap_stops_the_attempt_before_the_offending_call(
    attempt_env: AttemptEnv,
) -> None:
    # mock-small worst case per call is 512 output tokens * $4/Mtok ~ $0.002; allow only ~2 calls.
    env = tight_env(
        attempt_env, per_attempt_cap_usd=0.004, per_task_cap_usd=0.4, global_cap_usd=0.9
    )
    source = env.source()
    out, _ = run_pipeline(env, source)
    assert out.error is not None and "attempt cap" in out.error
    assert env.ledger.spent_usd() <= 0.004 + 1e-9


def test_an_attempt_stopped_mid_patch_still_records_its_patch_and_verifier_report(
    attempt_env: AttemptEnv,
) -> None:
    """The kill switch trips while the Patcher is producing its edit; the next call is refused.

    The edit was already applied, so the outcome must still carry that patch and a Verifier report
    (computed without any LLM), plus the reason the attempt stopped. Nothing is discarded.
    """
    steps = patch_steps()

    def patcher_plan(_call: object, index: int) -> ModelStep:
        if index == 1:  # the edit_file step: trip the switch, so call index 2 is refused
            attempt_env.budget.kill_switch_path.touch()
        return steps[index]

    plans: dict[str, object] = dict(good_plans())
    plans["patcher"] = patcher_plan
    source = MockModelSource(plans)  # type: ignore[arg-type]
    out, _ = run_pipeline(attempt_env, source)
    assert out.error is not None and "KillSwitchEngagedError" in out.error
    assert "return high" in out.src_patch  # the edit that landed before the stop
    assert out.report is not None and out.report.claimed  # verified without any further LLM call
    assert source.calls("reviewer") == ()  # and no later role was called
    assert len(source.calls("patcher")) == 2
    failed = [s for s in out.stages if not s.ok]
    assert [s.stage for s in failed] == ["patcher"]


def test_the_kill_switch_aborts_before_any_model_call(attempt_env: AttemptEnv) -> None:
    attempt_env.budget.kill_switch_path.touch()
    source = attempt_env.source()
    out, _ = run_pipeline(attempt_env, source)
    assert out.error is not None and "KillSwitchEngagedError" in out.error
    assert all(
        source.calls(r) == () for r in ("triager", "reproducer", "localizer", "patcher", "reviewer")
    )
    assert attempt_env.ledger.rows() == [] and out.report is None  # patcher never started


class _FakeRealSource(MockModelSource):
    is_real = True


def test_the_spend_gate_is_never_swallowed_by_the_pipeline(attempt_env: AttemptEnv) -> None:
    pipeline, base, work = attempt_env.pipeline(_FakeRealSource(good_plans()))
    with pytest.raises(SpendNotAllowedError):
        pipeline.run(
            attempt_env.task.agent_view(), base, work
        )  # a config error, not an attempt result
    assert attempt_env.ledger.rows() == []


def test_a_model_error_inside_a_role_is_survivable_and_logged(attempt_env: AttemptEnv) -> None:
    plans = good_plans()
    plans["reviewer"] = [ModelStep(error=RuntimeError("socket closed")), final_step("ignored")]
    source = attempt_env.source(plans)
    pipeline, base, work = attempt_env.pipeline(source)
    with pytest.raises(RuntimeError, match="socket closed"):
        pipeline.run(attempt_env.task.agent_view(), base, work)  # unexpected bugs surface loudly
    errors = [r for r in attempt_env.ledger.rows() if r.status == "error"]
    assert len(errors) == 1 and errors[0].role == "reviewer" and errors[0].cost_usd == 0.0


def test_pipeline_config_validation_and_id() -> None:
    assert PipelineConfig().config_id() == PipelineConfig().config_id()
    assert PipelineConfig().config_id() != PipelineConfig(reviewer=False).config_id()
    assert PipelineConfig().config_id() != PipelineConfig(max_repairs=3).config_id()
    for bad in (dict(max_repairs=-1), dict(max_output_tokens=0), dict(test_timeout_s=0)):
        with pytest.raises(ValueError, match="invalid pipeline config"):
            PipelineConfig(**bad)  # type: ignore[arg-type]


def test_source_protocol_is_the_modelsource_abstraction() -> None:
    assert issubclass(MockModelSource, ModelSource) and sdk_usage(1, 1).requests == 1
    assert FIXED.strip().startswith("if value > high")
