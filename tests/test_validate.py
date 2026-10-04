from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from cascade.taskgen import validate as v
from cascade.taskgen.results import ERROR, FAILED, MISSING, PASSED, SKIPPED, XFAILED, RunResult
from cascade.taskgen.validate import classify, validate_task


def run(outcomes: dict[str, str], *, exit_code: int | None = 0, duration: float = 1.0,
        timed_out: bool = False) -> RunResult:  # fmt: skip
    return RunResult(exit_code, duration, timed_out, outcomes, "")


def three(outcomes: dict[str, str], **kw: float | int | bool | None) -> list[RunResult]:
    return [run(dict(outcomes), **kw) for _ in range(3)]  # type: ignore[arg-type]


# --- classification --------------------------------------------------------------------------


def test_fail_to_pass_and_pass_to_pass_basic() -> None:
    pre = three({"a": FAILED, "b": PASSED, "c": PASSED}, exit_code=1)
    post = three({"a": PASSED, "b": PASSED, "c": PASSED})
    f2p, p2p, flaky, reason = classify(pre, post)
    assert (f2p, p2p, flaky, reason) == (["a"], ["b", "c"], [], None)


def test_test_missing_before_but_passing_after_is_fail_to_pass() -> None:
    # The test patch adds a module that cannot even be imported before the fix.
    pre = three({"old": PASSED, "tests/test_new.py": ERROR}, exit_code=1)
    post = three({"old": PASSED, "tests/test_new.py::test_new": PASSED})
    f2p, p2p, _, reason = classify(pre, post)
    assert f2p == ["tests/test_new.py::test_new"] and p2p == ["old"] and reason is None


def test_flaky_tests_are_dropped_from_both_lists() -> None:
    pre = [run({"a": FAILED, "wobble": FAILED}, exit_code=1),
           run({"a": FAILED, "wobble": PASSED}, exit_code=1),
           run({"a": FAILED, "wobble": FAILED}, exit_code=1)]  # fmt: skip
    post = three({"a": PASSED, "wobble": PASSED})
    f2p, p2p, flaky, reason = classify(pre, post)
    assert f2p == ["a"] and p2p == [] and flaky == ["wobble"] and reason is None


def test_flaky_in_post_only_is_also_dropped() -> None:
    pre = three({"a": FAILED, "b": PASSED}, exit_code=1)
    post = [run({"a": PASSED, "b": PASSED}), run({"a": PASSED, "b": FAILED}, exit_code=1),
            run({"a": PASSED, "b": PASSED})]  # fmt: skip
    f2p, p2p, flaky, reason = classify(pre, post)
    assert f2p == ["a"] and p2p == [] and flaky == ["b"] and reason is None


def test_a_test_present_in_only_some_runs_counts_as_flaky() -> None:
    pre = [run({"a": FAILED}, exit_code=1), run({"a": FAILED, "ghost": PASSED}, exit_code=1),
           run({"a": FAILED}, exit_code=1)]  # fmt: skip
    post = three({"a": PASSED, "ghost": PASSED})
    _, _, flaky, _ = classify(pre, post)
    assert flaky == ["ghost"]


def test_gold_patch_breaking_a_passing_test_drops_the_task() -> None:
    pre = three({"a": FAILED, "b": PASSED}, exit_code=1)
    post = three({"a": PASSED, "b": FAILED}, exit_code=1)
    assert classify(pre, post)[3] == v.GOLD_REGRESSES


def test_no_fail_to_pass_drops_the_task() -> None:
    pre = three({"a": PASSED, "b": PASSED})
    post = three({"a": PASSED, "b": PASSED})
    f2p, p2p, _, reason = classify(pre, post)
    assert reason == v.NO_FAIL_TO_PASS and f2p == [] and p2p == ["a", "b"]


def test_tests_failing_in_both_and_skipped_ones_belong_to_neither_list() -> None:
    pre = three({"a": FAILED, "broken": FAILED, "skip": SKIPPED, "xf": XFAILED}, exit_code=1)
    post = three({"a": PASSED, "broken": FAILED, "skip": PASSED, "xf": PASSED}, exit_code=1)
    f2p, p2p, _, reason = classify(pre, post)
    assert f2p == ["a"] and p2p == [] and reason is None


def test_results_are_sorted_for_determinism() -> None:
    pre = three({"z": FAILED, "m": FAILED, "p2": PASSED, "p1": PASSED}, exit_code=1)
    post = three({"z": PASSED, "m": PASSED, "p2": PASSED, "p1": PASSED})
    f2p, p2p, _, _ = classify(pre, post)
    assert f2p == ["m", "z"] and p2p == ["p1", "p2"]


# --- orchestration ---------------------------------------------------------------------------


class ScriptedExecutor:
    """Returns the next scripted result for each call and records what it was asked to run."""

    def __init__(self, results: list[RunResult]) -> None:
        self._results = list(results)
        self.calls: list[tuple[Path, float]] = []

    def run(self, *, workspace: Path, pytest_args: Sequence[str], timeout_s: float) -> RunResult:
        self.calls.append((workspace, timeout_s))
        return self._results.pop(0)


def harness(results: list[RunResult], tmp_path: Path):
    made: list[tuple[bool, Path]] = []
    cleaned: list[Path] = []

    def factory(with_gold: bool) -> Path:
        ws = tmp_path / f"ws{len(made)}"
        ws.mkdir()
        made.append((with_gold, ws))
        return ws

    return ScriptedExecutor(results), factory, made, cleaned


def test_runs_three_pre_then_three_post_in_fresh_workspaces(tmp_path: Path) -> None:
    pre = [run({"a": FAILED}, exit_code=1, duration=d) for d in (1.0, 1.0, 1.0)]
    post = [run({"a": PASSED}, duration=d) for d in (2.0, 4.0, 3.0)]
    ex, factory, made, cleaned = harness(pre + post, tmp_path)
    out = validate_task(ex, factory, runs=3, runtime_cap_s=60, cleanup=cleaned.append)
    assert out.validated and out.fail_to_pass == ["a"]
    assert [g for g, _ in made] == [False, False, False, True, True, True]
    assert len({ws for _, ws in made}) == 6  # six distinct workspaces
    assert cleaned == [ws for _, ws in made]  # every workspace cleaned, in order
    assert all(t == 60 for _, t in ex.calls)  # the cap is also the container timeout
    assert out.test_runtime_s == 3.0  # median of the post-gold runs


def test_timeout_drops_the_task_as_over_runtime_cap(tmp_path: Path) -> None:
    results = (
        [run({"a": FAILED}, exit_code=1)] * 3
        + [run({}, exit_code=None, timed_out=True)]
        + [run({"a": PASSED})] * 2
    )
    ex, factory, _, _ = harness(results, tmp_path)
    out = validate_task(ex, factory, runs=3, runtime_cap_s=30)
    assert out.reason == v.OVER_RUNTIME_CAP and not out.validated


def test_slow_run_without_timeout_is_still_over_the_cap(tmp_path: Path) -> None:
    results = [run({"a": FAILED}, exit_code=1)] * 3 + [run({"a": PASSED}, duration=31.0)] * 3
    ex, factory, _, _ = harness(results, tmp_path)
    assert validate_task(ex, factory, runs=3, runtime_cap_s=30).reason == v.OVER_RUNTIME_CAP


@pytest.mark.parametrize("exit_code", [2, 3, 4, 5, None])
def test_unusable_pre_run_is_a_run_error(tmp_path: Path, exit_code: int | None) -> None:
    results = [run({"a": FAILED}, exit_code=exit_code)] + [run({"a": FAILED}, exit_code=1)] * 2
    results += [run({"a": PASSED})] * 3
    ex, factory, _, _ = harness(results, tmp_path)
    assert validate_task(ex, factory, runs=3, runtime_cap_s=30).reason == v.RUN_ERROR_PRE


def test_unusable_post_run_is_a_run_error(tmp_path: Path) -> None:
    results = (
        [run({"a": FAILED}, exit_code=1)] * 3 + [run({}, exit_code=0)] + [run({"a": PASSED})] * 2
    )
    ex, factory, _, _ = harness(results, tmp_path)
    assert validate_task(ex, factory, runs=3, runtime_cap_s=30).reason == v.RUN_ERROR_POST


def test_workspace_errors_propagate_and_still_clean_up(tmp_path: Path) -> None:
    ex = ScriptedExecutor([])
    cleaned: list[Path] = []

    def factory(with_gold: bool) -> Path:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        validate_task(ex, factory, runs=3, runtime_cap_s=30, cleanup=cleaned.append)
    assert cleaned == []  # nothing was created, so nothing to clean


def test_missing_constant_is_not_an_executor_outcome() -> None:
    # The executor never emits "missing"; only classification synthesizes it for absent tests.
    assert MISSING not in {PASSED, FAILED, ERROR, SKIPPED, XFAILED}
