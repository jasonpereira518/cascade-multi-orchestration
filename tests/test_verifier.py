from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from cascade.agents.verifier import Verifier, usable
from cascade.harness.sandbox import prepare_workspace
from cascade.taskgen.results import RunResult
from tests.agent_helpers import (
    BUGGY,
    CLAMP,
    FIXED,
    OVERFIT,
    PASSING_REPRO_CODE,
    REPRO_CODE,
    REPRO_PATH,
    NativeExecutor,
)
from tests.conftest import FixtureRepo


def make_work(
    fx: FixtureRepo, tmp: Path, name: str, *, branch: str | None, repro: str | None
) -> Path:
    work = prepare_workspace(fx.path, fx.task.base_commit, tmp / name)
    if branch is not None:
        clamp = work / CLAMP
        clamp.write_text(clamp.read_text().replace(BUGGY, branch))
    if repro is not None:
        (work / REPRO_PATH).write_text(repro)
    return work


@pytest.fixture
def base(fixture_repo: FixtureRepo, tmp_path: Path) -> Path:
    return prepare_workspace(fixture_repo.path, fixture_repo.task.base_commit, tmp_path / "base")


def verifier(base: Path, tmp_path: Path, executor: NativeExecutor | None = None) -> Verifier:
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    return Verifier(
        executor=executor or NativeExecutor(), base_dir=base, scratch=scratch,
        repro_path=REPRO_PATH, timeout_s=60.0,
    )  # fmt: skip


def test_a_correct_patch_with_a_valid_reproducer_is_claimed(
    fixture_repo: FixtureRepo, base: Path, tmp_path: Path
) -> None:
    work = make_work(fixture_repo, tmp_path, "work", branch=FIXED, repro=REPRO_CODE)
    result = verifier(base, tmp_path).verify(
        work_dir=work, patch_nonempty=True, repro_enabled=True, repro_failed_before=True
    )
    r = result.report
    assert r.claimed and r.reasons == [] and r.regressions == []
    assert r.repro_failed_before and r.repro_passes_after and r.baseline_ok
    assert r.suite_passed_before == 2 and r.suite_passed_after == 2  # the two visible tests


def test_an_empty_patch_is_never_claimed_and_runs_nothing(
    fixture_repo: FixtureRepo, base: Path, tmp_path: Path
) -> None:
    executor = NativeExecutor()
    work = make_work(fixture_repo, tmp_path, "work", branch=None, repro=REPRO_CODE)
    r = verifier(base, tmp_path, executor).verify(
        work_dir=work, patch_nonempty=False, repro_enabled=True, repro_failed_before=True
    ).report  # fmt: skip
    assert not r.claimed and not r.patch_nonempty and "empty" in r.reasons[0]
    assert executor.calls == []  # no suite was executed for an empty patch


def test_a_patch_that_breaks_a_visible_test_is_a_regression(
    fixture_repo: FixtureRepo, base: Path, tmp_path: Path
) -> None:
    work = make_work(fixture_repo, tmp_path, "work", branch=FIXED, repro=REPRO_CODE)
    (work / CLAMP).write_text(  # fixes the reported case but returns `high` for values below `low`
        "def clamp(value, low, high):\n"
        "    if value < low:\n        return high\n"
        "    if value > high:\n        return high\n"
        "    return value\n"
    )
    result = verifier(base, tmp_path).verify(
        work_dir=work, patch_nonempty=True, repro_enabled=True, repro_failed_before=True
    )
    r = result.report
    assert not r.claimed and r.regressions == ["tests/test_clamp.py::test_below_range"]
    assert any("previously passing" in reason for reason in r.reasons)
    assert "test_below_range" in result.feedback


def test_a_patch_that_does_not_fix_the_reproducer_is_not_claimed(
    fixture_repo: FixtureRepo, base: Path, tmp_path: Path
) -> None:
    work = make_work(
        fixture_repo,
        tmp_path,
        "work",
        branch=BUGGY.replace("low", "low  # unchanged"),
        repro=REPRO_CODE,
    )
    result = verifier(base, tmp_path).verify(
        work_dir=work, patch_nonempty=True, repro_enabled=True, repro_failed_before=True
    )
    assert not result.report.claimed and not result.report.repro_passes_after
    assert "still fails after the patch" in "; ".join(result.report.reasons)
    assert "test_repro_value_above_range" in result.feedback


def test_a_reproducer_that_never_failed_proves_nothing(
    fixture_repo: FixtureRepo, base: Path, tmp_path: Path
) -> None:
    work = make_work(fixture_repo, tmp_path, "work", branch=FIXED, repro=PASSING_REPRO_CODE)
    r = verifier(base, tmp_path).verify(
        work_dir=work, patch_nonempty=True, repro_enabled=True, repro_failed_before=False
    ).report  # fmt: skip
    assert not r.claimed and any("did not fail on the unpatched code" in x for x in r.reasons)


def test_a_missing_reproducer_file_blocks_the_claim_when_enabled(
    fixture_repo: FixtureRepo, base: Path, tmp_path: Path
) -> None:
    work = make_work(fixture_repo, tmp_path, "work", branch=FIXED, repro=None)
    r = verifier(base, tmp_path).verify(
        work_dir=work, patch_nonempty=True, repro_enabled=True, repro_failed_before=True
    ).report  # fmt: skip
    assert not r.claimed and "the reproducer file is missing" in r.reasons


def test_with_the_reproducer_disabled_the_claim_rests_on_regressions_alone(
    fixture_repo: FixtureRepo, base: Path, tmp_path: Path
) -> None:
    work = make_work(fixture_repo, tmp_path, "work", branch=FIXED, repro=None)
    r = verifier(base, tmp_path).verify(
        work_dir=work, patch_nonempty=True, repro_enabled=False, repro_failed_before=False
    ).report  # fmt: skip
    assert r.claimed and not r.repro_enabled


def test_an_overfit_patch_is_claimed_which_is_a_verifier_false_positive(
    fixture_repo: FixtureRepo, base: Path, tmp_path: Path
) -> None:
    """The Verifier only sees its own evidence; hidden tests are what expose this patch."""
    work = make_work(fixture_repo, tmp_path, "work", branch=OVERFIT, repro=REPRO_CODE)
    r = verifier(base, tmp_path).verify(
        work_dir=work, patch_nonempty=True, repro_enabled=True, repro_failed_before=True
    ).report  # fmt: skip
    assert r.claimed


def test_the_baseline_suite_runs_once_per_attempt(
    fixture_repo: FixtureRepo, base: Path, tmp_path: Path
) -> None:
    executor = NativeExecutor()
    v = verifier(base, tmp_path, executor)
    work = make_work(fixture_repo, tmp_path, "work", branch=FIXED, repro=REPRO_CODE)
    v.verify(work_dir=work, patch_nonempty=True, repro_enabled=True, repro_failed_before=True)
    first = len(executor.calls)  # baseline + after + reproducer
    v.verify(work_dir=work, patch_nonempty=True, repro_enabled=True, repro_failed_before=True)
    assert first == 3 and len(executor.calls) == 5  # the second verify skips the baseline


def test_verification_never_pollutes_the_pristine_base_or_the_agent_workspace(
    fixture_repo: FixtureRepo, base: Path, tmp_path: Path
) -> None:
    work = make_work(fixture_repo, tmp_path, "work", branch=FIXED, repro=REPRO_CODE)

    def listing(root: Path) -> list[str]:
        return sorted(str(p.relative_to(root)) for p in root.rglob("*"))

    before_base, before_work = listing(base), listing(work)
    verifier(base, tmp_path).verify(
        work_dir=work, patch_nonempty=True, repro_enabled=True, repro_failed_before=True
    )
    assert listing(base) == before_base and listing(work) == before_work
    assert not list((tmp_path / "scratch").iterdir())  # throwaway copies are removed


class Scripted:
    """Returns canned results in order, recording the arguments of every call."""

    def __init__(self, results: list[RunResult]) -> None:
        self._results = list(results)
        self.args: list[list[str]] = []

    def run(self, *, workspace: Path, pytest_args: Sequence[str], timeout_s: float) -> RunResult:
        self.args.append(list(pytest_args))
        return self._results.pop(0)


def suite(**outcomes: str) -> RunResult:
    return RunResult(1 if "failed" in outcomes.values() else 0, 1.0, False, dict(outcomes), "")


def test_a_regression_must_reproduce_on_a_second_run_to_count(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    work_dir = tmp_path / "work"
    for d in (base_dir, work_dir):
        d.mkdir()
    executor = Scripted([
        suite(a="passed", b="passed"),  # baseline
        suite(a="passed", b="failed"),  # after the patch: b looks like a regression
        suite(b="passed"),  # re-run of just b: it was a flake
    ])  # fmt: skip
    v = Verifier(
        executor=executor, base_dir=base_dir, scratch=tmp_path, repro_path=REPRO_PATH, timeout_s=5
    )  # type: ignore[arg-type]
    r = v.verify(
        work_dir=work_dir, patch_nonempty=True, repro_enabled=False, repro_failed_before=False
    ).report
    assert r.claimed and r.regressions == []
    assert executor.args[2][-1] == "b"  # only the suspect test was re-run


def test_a_regression_that_persists_is_reported(tmp_path: Path) -> None:
    for d in ("base", "work"):
        (tmp_path / d).mkdir()
    executor = Scripted(
        [suite(a="passed", b="passed"), suite(a="passed", b="failed"), suite(b="failed")]
    )
    v = Verifier(
        executor=executor,
        base_dir=tmp_path / "base",
        scratch=tmp_path,
        repro_path=REPRO_PATH,
        timeout_s=5,
    )  # type: ignore[arg-type]
    r = v.verify(
        work_dir=tmp_path / "work",
        patch_nonempty=True,
        repro_enabled=False,
        repro_failed_before=False,
    ).report
    assert not r.claimed and r.regressions == ["b"]


def test_a_test_that_disappears_counts_as_a_regression(tmp_path: Path) -> None:
    for d in ("base", "work"):
        (tmp_path / d).mkdir()
    executor = Scripted([suite(a="passed", b="passed"), suite(a="passed"), suite(a="passed")])
    v = Verifier(
        executor=executor,
        base_dir=tmp_path / "base",
        scratch=tmp_path,
        repro_path=REPRO_PATH,
        timeout_s=5,
    )  # type: ignore[arg-type]
    r = v.verify(
        work_dir=tmp_path / "work",
        patch_nonempty=True,
        repro_enabled=False,
        repro_failed_before=False,
    ).report
    assert r.regressions == ["b"] and not r.claimed


def test_a_suite_that_cannot_run_is_never_a_claim(tmp_path: Path) -> None:
    for d in ("base", "work"):
        (tmp_path / d).mkdir()
    broken = RunResult(2, 1.0, False, {}, "usage error")
    executor = Scripted([suite(a="passed"), broken])
    v = Verifier(
        executor=executor,
        base_dir=tmp_path / "base",
        scratch=tmp_path,
        repro_path=REPRO_PATH,
        timeout_s=5,
    )  # type: ignore[arg-type]
    r = v.verify(
        work_dir=tmp_path / "work",
        patch_nonempty=True,
        repro_enabled=False,
        repro_failed_before=False,
    ).report
    assert not r.claimed and any("did not run after the patch" in x for x in r.reasons)

    executor2 = Scripted([broken])
    v2 = Verifier(
        executor=executor2,
        base_dir=tmp_path / "base",
        scratch=tmp_path,
        repro_path=REPRO_PATH,
        timeout_s=5,
    )  # type: ignore[arg-type]
    executor2._results.append(suite(a="passed"))
    r2 = v2.verify(
        work_dir=tmp_path / "work",
        patch_nonempty=True,
        repro_enabled=False,
        repro_failed_before=False,
    ).report
    assert not r2.claimed and not r2.baseline_ok


@pytest.mark.parametrize(
    ("run", "ok"),
    [
        (RunResult(0, 1.0, False, {"a": "passed"}, ""), True),
        (RunResult(1, 1.0, False, {"a": "failed"}, ""), True),
        (RunResult(2, 1.0, False, {"a": "passed"}, ""), False),
        (RunResult(0, 1.0, False, {}, ""), False),
        (RunResult(None, 1.0, True, {"a": "passed"}, ""), False),
    ],
)
def test_usable(run: RunResult, ok: bool) -> None:
    assert usable(run) is ok
