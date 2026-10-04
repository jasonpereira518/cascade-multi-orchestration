"""The Verifier: the sixth role. No LLM; it executes tests and reports what the evidence shows.

verifier_claimed_success is the *agent's own* evidence that its patch works:
  - the reproducer test FAILS before the patch (checked by the pipeline on the base state) and
    PASSES after it, and
  - no test of the repo's existing suite regresses (passes on the base commit, does not pass after).
With the Reproducer disabled (an ablation) the first condition is dropped.

It never sees hidden tests; true_success (hidden fail_to_pass / pass_to_pass) is graded separately
in ``runner.grade_patch``. Comparing the two is what lets us measure Verifier false positives.

Each suite run happens in a throwaway copy, so the pristine base export and the agent's workspace
are never polluted by test runs. A regression must reproduce on a second run before it counts,
which keeps one flaky test from turning a good patch into a "claimed failure".
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from cascade.agents.failures import failing_ids, passed_ids, summarize_run
from cascade.agents.patching import IGNORED_DIRS, IGNORED_FILES
from cascade.agents.schemas import VerifierReport
from cascade.taskgen.results import RunResult
from cascade.taskgen.validate import TestExecutor

_USABLE_EXIT_CODES = frozenset({0, 1})
_MAX_LISTED = 15


@dataclass(frozen=True)
class VerifyResult:
    report: VerifierReport
    feedback: str  # failure details to hand back to the Patcher


def usable(run: RunResult) -> bool:
    return not run.timed_out and run.exit_code in _USABLE_EXIT_CODES and bool(run.outcomes)


def _ignore(_dir: str, names: list[str]) -> list[str]:
    return [n for n in names if n in IGNORED_DIRS or n in IGNORED_FILES or n.endswith(".pyc")]


class Verifier:
    def __init__(
        self,
        *,
        executor: TestExecutor,
        base_dir: Path,
        scratch: Path,
        repro_path: str,
        timeout_s: float,
    ) -> None:
        self._executor = executor
        self._base = base_dir
        self._scratch = scratch
        self._repro_path = repro_path
        self._timeout = timeout_s
        self._baseline: RunResult | None = None
        self._counter = 0

    def _run_copy(self, source: Path, args: list[str], *, drop_repro: bool = False) -> RunResult:
        """Run pytest in a throwaway copy of ``source`` under the scratch directory."""
        self._counter += 1
        copy = self._scratch / f"verify-{self._counter}"
        shutil.rmtree(copy, ignore_errors=True)
        shutil.copytree(source, copy, ignore=_ignore, symlinks=False)
        try:
            if drop_repro:
                (copy / self._repro_path).unlink(missing_ok=True)
            return self._executor.run(
                workspace=copy, pytest_args=["--tb=short", *args], timeout_s=self._timeout
            )
        finally:
            shutil.rmtree(copy, ignore_errors=True)

    def baseline(self) -> RunResult:
        """The visible suite on the untouched base commit. Computed once per attempt."""
        if self._baseline is None:
            self._baseline = self._run_copy(self._base, [])
        return self._baseline

    def verify(
        self,
        *,
        work_dir: Path,
        patch_nonempty: bool,
        repro_enabled: bool,
        repro_failed_before: bool,
    ) -> VerifyResult:
        reasons: list[str] = []
        detail: list[str] = []
        if not patch_nonempty:
            reasons.append("the patch is empty: no source file was changed")
            return VerifyResult(
                VerifierReport(
                    patch_nonempty=False, baseline_ok=True, repro_enabled=repro_enabled,
                    repro_failed_before=repro_failed_before, repro_passes_after=False,
                    regressions=[], suite_passed_before=0, suite_passed_after=0,
                    claimed=False, reasons=reasons,
                ),
                "",
            )  # fmt: skip

        before = self.baseline()
        baseline_ok = usable(before)
        if not baseline_ok:
            reasons.append("the existing test suite does not run on the base commit")

        after = self._run_copy(work_dir, [], drop_repro=True)
        after_ok = usable(after)
        if not after_ok:
            reasons.append("the existing test suite did not run after the patch")
            detail.append(summarize_run(after))

        regressions: list[str] = []
        if baseline_ok and after_ok:
            suspects = sorted(passed_ids(before) - passed_ids(after))
            if suspects:
                again = self._run_copy(work_dir, suspects, drop_repro=True)
                regressions = [t for t in suspects if t not in passed_ids(again)]
            if regressions:
                reasons.append(
                    f"{len(regressions)} previously passing test(s) now fail: "
                    + ", ".join(regressions[:_MAX_LISTED])
                )
                detail.append(summarize_run(after))

        repro_passes = False
        if repro_enabled:
            if not repro_failed_before:
                reasons.append(
                    "the reproducer did not fail on the unpatched code (it proves nothing)"
                )
            repro_file = work_dir / self._repro_path
            if repro_file.exists():
                run = self._run_copy(work_dir, [self._repro_path])
                prefix = self._repro_path + "::"
                outcomes = {n: o for n, o in run.outcomes.items() if n.startswith(prefix)}
                repro_passes = bool(outcomes) and all(o == "passed" for o in outcomes.values())
                if not repro_passes:
                    reasons.append("the reproducer still fails after the patch")
                    detail.append(summarize_run(run))
            else:
                reasons.append("the reproducer file is missing")

        claimed = (
            baseline_ok
            and after_ok
            and not regressions
            and (not repro_enabled or (repro_failed_before and repro_passes))
        )
        report = VerifierReport(
            patch_nonempty=True,
            baseline_ok=baseline_ok,
            repro_enabled=repro_enabled,
            repro_failed_before=repro_failed_before,
            repro_passes_after=repro_passes,
            regressions=regressions[:_MAX_LISTED],
            suite_passed_before=len(passed_ids(before)),
            suite_passed_after=len(passed_ids(after)),
            claimed=claimed,
            reasons=reasons,
        )
        return VerifyResult(report, "\n\n".join(detail))


__all__ = ["Verifier", "VerifyResult", "failing_ids", "usable"]
