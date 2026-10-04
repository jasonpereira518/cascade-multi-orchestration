"""Task validation: run the repo's tests with and without the gold patch, three times each.

Two configurations are executed, each ``runs`` times in fresh workspaces:
  pre  = base commit + test patch
  post = base commit + test patch + gold patch

Per test, a configuration is *stable* if all runs agree (a test absent from a run counts as
"missing"). Unstable tests are flaky and are dropped from every list. Then, over stable tests:
  fail_to_pass = not passing in pre (failed / error / missing) and passing in post
  pass_to_pass = passing in pre and passing in post
A task is dropped if the gold patch breaks a test that passed before ("gold_regresses"), if any
run times out or exceeds the runtime cap, if a run is unusable, or if fail_to_pass ends up empty.
Tests that fail in both configurations, or are skipped/xfailed, belong to neither list.

The independence assumption is the obvious one: three runs of an unchanged workspace are treated
as repeated samples of the same test, so disagreement means nondeterminism.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from cascade.taskgen.results import ERROR, FAILED, MISSING, NOT_PASSING, PASSED, RunResult

# Drop reasons
RUN_ERROR_PRE = "run_error_pre"
RUN_ERROR_POST = "run_error_post"
OVER_RUNTIME_CAP = "over_runtime_cap"
GOLD_REGRESSES = "gold_regresses"
NO_FAIL_TO_PASS = "no_fail_to_pass"  # noqa: S105  (a reason code, not a password)
PATCH_APPLY_FAILED = "patch_apply_failed"

# pytest exit codes that still produce a usable per-test report: 0 = all passed, 1 = some failed.
_USABLE_EXIT_CODES = frozenset({0, 1})


class TestExecutor(Protocol):
    __test__ = False  # not a pytest class, despite the name

    def run(
        self, *, workspace: Path, pytest_args: Sequence[str], timeout_s: float
    ) -> RunResult: ...


# Builds a fresh workspace each call: with_gold=False -> base + test patch, True -> + gold patch.
WorkspaceFactory = Callable[[bool], Path]


@dataclass(frozen=True)
class ValidationOutcome:
    reason: str | None = None
    fail_to_pass: list[str] = field(default_factory=lambda: [])
    pass_to_pass: list[str] = field(default_factory=lambda: [])
    flaky: list[str] = field(default_factory=lambda: [])
    test_runtime_s: float = 0.0
    detail: str = ""

    @property
    def validated(self) -> bool:
        return self.reason is None


def _usable(run: RunResult) -> bool:
    return not run.timed_out and run.exit_code in _USABLE_EXIT_CODES and bool(run.outcomes)


def _stable_outcomes(runs: Sequence[RunResult]) -> tuple[dict[str, str], set[str]]:
    """Return ({test: outcome} for stable tests, {flaky tests})."""
    names: set[str] = set()
    for r in runs:
        names.update(r.outcomes)
    stable: dict[str, str] = {}
    flaky: set[str] = set()
    for name in names:
        seen = {r.outcomes.get(name, MISSING) for r in runs}
        if len(seen) == 1:
            stable[name] = seen.pop()
        else:
            flaky.add(name)
    return stable, flaky


def classify(
    pre_runs: Sequence[RunResult], post_runs: Sequence[RunResult]
) -> tuple[list[str], list[str], list[str], str | None]:
    """Return (fail_to_pass, pass_to_pass, flaky, drop_reason) from the two sets of runs."""
    pre, pre_flaky = _stable_outcomes(pre_runs)
    post, post_flaky = _stable_outcomes(post_runs)
    flaky = pre_flaky | post_flaky

    fail_to_pass: list[str] = []
    pass_to_pass: list[str] = []
    for name in sorted(set(pre) | set(post)):
        if name in flaky:
            continue
        before = pre.get(name, MISSING)
        after = post.get(name, MISSING)
        if before == PASSED and after in (FAILED, ERROR):
            return [], [], sorted(flaky), GOLD_REGRESSES
        if after != PASSED:
            continue
        if before in NOT_PASSING:
            fail_to_pass.append(name)
        elif before == PASSED:
            pass_to_pass.append(name)
    if not fail_to_pass:
        return [], pass_to_pass, sorted(flaky), NO_FAIL_TO_PASS
    return fail_to_pass, pass_to_pass, sorted(flaky), None


def validate_task(
    executor: TestExecutor,
    make_workspace: WorkspaceFactory,
    *,
    runs: int = 3,
    runtime_cap_s: float,
    cleanup: Callable[[Path], None] = lambda _p: None,
) -> ValidationOutcome:
    def execute(with_gold: bool) -> list[RunResult]:
        results: list[RunResult] = []
        for _ in range(runs):
            ws = make_workspace(with_gold)
            try:
                results.append(executor.run(workspace=ws, pytest_args=[], timeout_s=runtime_cap_s))
            finally:
                cleanup(ws)
        return results

    pre_runs = execute(False)
    post_runs = execute(True)

    for run in [*pre_runs, *post_runs]:
        if run.timed_out or run.duration_s > runtime_cap_s:
            return ValidationOutcome(
                reason=OVER_RUNTIME_CAP,
                detail=f"a run exceeded the {runtime_cap_s:.0f}s cap",
            )
    for reason, group in ((RUN_ERROR_PRE, pre_runs), (RUN_ERROR_POST, post_runs)):
        bad = next((r for r in group if not _usable(r)), None)
        if bad is not None:
            return ValidationOutcome(
                reason=reason,
                detail=f"exit_code={bad.exit_code} tests_reported={len(bad.outcomes)} "
                f"tail={bad.stdout_tail[-300:]!r}",
            )

    f2p, p2p, flaky, reason = classify(pre_runs, post_runs)
    runtime = statistics.median(r.duration_s for r in post_runs)
    return ValidationOutcome(
        reason=reason,
        fail_to_pass=f2p,
        pass_to_pass=p2p,
        flaky=flaky,
        test_runtime_s=runtime,
    )
