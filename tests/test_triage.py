"""Attempt triage: pipeline bug vs Verifier issue vs genuine model failure. Synthetic rows only.

The heuristics are tested against hand-built (matrix row, result.json) pairs that mimic the shapes
the runner writes; they show the classification rules, not how often each category occurs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cascade.agents import triage as tr
from cascade.agents.runner import _safe  # pyright: ignore[reportPrivateUsage]
from cascade.routing.matrix import MatrixRow, MatrixStore


def row(**over: Any) -> MatrixRow:
    base: dict[str, Any] = {
        "task_id": "o__a-1", "arm_id": "cheap", "seed": 0, "true_success": False,
        "verifier_claimed": False, "cost_usd": 0.01, "input_tokens": 100, "output_tokens": 20,
        "reasoning_tokens": 0, "wall_s": 3.0, "turns": 4, "patch_sha": None,
        "pipeline_config": "{}", "error": None,
    }  # fmt: skip
    return MatrixRow(**{**base, **over})


def cat(r: MatrixRow, result: dict[str, Any] | None = None) -> str:
    return tr.diagnose(r, result).category


def test_clean_success_is_success() -> None:
    d = tr.diagnose(row(true_success=True, verifier_claimed=True), {"grade": {"reason": "passed"}})
    assert d.category == tr.SUCCESS and d.flags == [] and d.action == "nothing"


@pytest.mark.parametrize(
    "reason", ["patch_apply_failed", "test_patch_conflict", "grading_run_error", "grading_timeout"]
)
def test_grading_infrastructure_failures_are_pipeline_bugs_not_model_failures(reason: str) -> None:
    d = tr.diagnose(row(), {"grade": {"reason": reason}})
    assert d.category == tr.PIPELINE_BUG and reason in d.evidence[0]


def test_pipeline_bug_wins_even_if_the_row_claims_success() -> None:
    assert (
        cat(row(true_success=True, verifier_claimed=True), {"grade": {"reason": "grading_timeout"}})
        == tr.PIPELINE_BUG
    )


def test_a_patch_that_could_not_be_built_is_a_pipeline_bug() -> None:
    d = tr.diagnose(row(error="patch_error: edit does not apply"), None)
    assert d.category == tr.PIPELINE_BUG and "patch_error" in d.evidence[-1]


def test_visible_suite_failing_on_the_base_commit_is_a_pipeline_bug() -> None:
    assert cat(row(), {"verifier": {"baseline_ok": False}}) == tr.PIPELINE_BUG
    assert cat(row(), {"verifier": {"baseline_ok": True}}) == tr.MODEL_FAILURE
    assert cat(row(), {"verifier": {}}) == tr.MODEL_FAILURE  # no verifier report: no inference


@pytest.mark.parametrize("err", ["BudgetExceededError: cap", "KillSwitchEngagedError: stop"])
def test_cap_and_kill_switch_are_cost_data_points_not_defects(err: str) -> None:
    d = tr.diagnose(row(error=err), None)
    assert d.category == tr.CAP and "cost data point" in d.action


def test_failed_stage_with_provider_error_is_a_config_problem() -> None:
    stages = [{"ok": True, "detail": "fine"}, {"ok": False, "detail": "NotFoundError: no model"}]
    d = tr.diagnose(row(), {"stages": stages})
    assert d.category == tr.PROVIDER and "NotFoundError" in d.evidence[0]


@pytest.mark.parametrize("marker", ["ModelBehaviorError", "MaxTurnsExceeded", "Invalid JSON"])
def test_failed_stage_with_malformed_output_is_a_model_output_failure(marker: str) -> None:
    assert (
        cat(row(), {"stages": [{"ok": False, "detail": f"{marker}: x"}]}) == tr.MODEL_OUTPUT_FAILURE
    )


def test_only_failed_stages_are_considered_for_error_markers() -> None:
    stages = [{"ok": True, "detail": "NotFoundError mentioned in passing"}]
    assert cat(row(), {"stages": stages}) == tr.MODEL_FAILURE


def test_verifier_claims_success_but_hidden_tests_fail_is_a_false_positive() -> None:
    grade = {"reason": "tests_failed", "failed": ["t::a", "t::b", "t::c", "t::d"]}
    d = tr.diagnose(row(verifier_claimed=True), {"grade": grade})
    assert d.category == tr.VERIFIER_FALSE_POSITIVE
    assert "t::c" in d.evidence[1] and "t::d" not in d.evidence[1]  # first three only
    assert "reproducer" in d.action


def test_hidden_tests_pass_but_verifier_rejected_is_a_false_negative_with_its_reasons() -> None:
    d = tr.diagnose(
        row(true_success=True), {"verifier": {"reasons": ["reproducer still fails", "x"]}}
    )
    assert d.category == tr.VERIFIER_FALSE_NEGATIVE
    assert d.evidence == ["Verifier said: reproducer still fails", "Verifier said: x"]


def test_ordinary_wrong_patch_is_a_genuine_model_failure_with_evidence() -> None:
    result = {
        "grade": {"reason": "tests_failed"},
        "verifier": {"reasons": ["a", "b", "c", "d"]},
    }
    d = tr.diagnose(row(), result)
    assert d.category == tr.MODEL_FAILURE
    assert d.evidence == ["grading: tests_failed", "Verifier: a", "Verifier: b", "Verifier: c"]


def test_empty_patch_is_described_as_such() -> None:
    d = tr.diagnose(row(), {"grade": {"reason": "empty_patch"}})
    assert d.category == tr.MODEL_FAILURE and "no source change" in d.evidence[0]


def test_missing_result_json_falls_back_to_the_row_and_says_so() -> None:
    d = tr.diagnose(row(), None)
    assert d.category == tr.MODEL_FAILURE and "no result.json" in d.evidence[0]


def test_leak_hit_is_flagged_on_top_of_whatever_the_category_is() -> None:
    d = tr.diagnose(
        row(true_success=True, verifier_claimed=True), {"leak_violations": ["gold line in trace"]}
    )
    assert d.category == tr.SUCCESS and d.flags == [tr.LEAK]
    assert "gold line in trace" in d.evidence[0]


def test_malformed_result_shapes_do_not_crash_diagnosis() -> None:
    junk = {"grade": "oops", "verifier": [1], "stages": "x", "leak_violations": {"a": 1}}
    assert cat(row(), junk) == tr.MODEL_FAILURE
    assert cat(row(), {"stages": [1, "x", None, {"ok": False, "detail": None}]}) == tr.MODEL_FAILURE


def test_every_category_has_an_action() -> None:
    for c in (
        tr.SUCCESS, tr.PIPELINE_BUG, tr.VERIFIER_FALSE_POSITIVE, tr.VERIFIER_FALSE_NEGATIVE,
        tr.VERIFIER_ISSUE, tr.MODEL_FAILURE, tr.MODEL_OUTPUT_FAILURE, tr.CAP, tr.PROVIDER, tr.LEAK,
    ):  # fmt: skip
        assert tr.ACTIONS[c]


# --- reading a runs directory ------------------------------------------------------------


def _put(runs: Path, r: MatrixRow, result: dict[str, Any] | None) -> None:
    store = MatrixStore(runs / "matrix.sqlite")
    store.put(r)
    store.close()
    if result is not None:
        d = (
            runs / "attempts" / _safe(r.task_id) / _safe(r.arm_id)
            / f"seed{r.seed}-{r.config_id}"
        )  # fmt: skip
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(result))


def test_triage_runs_reads_the_matrix_and_each_attempts_result(tmp_path: Path) -> None:
    _put(tmp_path, row(task_id="o__a-1"), {"grade": {"reason": "grading_timeout"}})
    _put(tmp_path, row(task_id="o__a-2", true_success=True, verifier_claimed=True), {})
    _put(tmp_path, row(task_id="o__a-3"), None)
    got = {d.task_id: d.category for d in tr.triage_runs(tmp_path)}
    assert got == {
        "o__a-1": tr.PIPELINE_BUG, "o__a-2": tr.SUCCESS, "o__a-3": tr.MODEL_FAILURE,
    }  # fmt: skip


def test_triage_of_a_missing_runs_dir_is_empty_and_says_so(tmp_path: Path) -> None:
    assert tr.triage_runs(tmp_path / "none") == []
    assert tr.format_triage([]) == "no attempts found"


def test_format_summarises_counts_hides_clean_successes_and_surfaces_leaks() -> None:
    ds = [
        tr.diagnose(row(task_id="ok", true_success=True, verifier_claimed=True), {}),
        tr.diagnose(row(task_id="bug"), {"grade": {"reason": "grading_timeout"}}),
        tr.diagnose(
            row(task_id="leak", true_success=True, verifier_claimed=True),
            {"leak_violations": ["x"]},
        ),
    ]
    text = tr.format_triage(ds)
    assert "3 attempt(s)" in text and "success 2" in text and "pipeline_bug_suspected 1" in text
    assert "LEAK AUDIT HITS: 1" in text
    assert "- bug | cheap | seed 0: pipeline_bug_suspected" in text
    assert "- leak | cheap | seed 0: success (+leak_audit_hit)" in text
    assert "- ok |" not in text  # a clean success is not listed
    assert text.count("next:") == 2  # the bug and the leak; the clean success is omitted
