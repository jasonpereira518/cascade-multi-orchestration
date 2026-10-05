"""The single-shot final evaluation and its guard.

Every test runs the guard against a throwaway git repository under tmp_path and a labeled-SYNTHETIC
matrix (``allow_synthetic=True`` exists only so the machinery can be exercised). The real final run
is never executed here: `cascade eval --final` itself refuses synthetic data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from cascade.evals import bootstrap as bs
from cascade.evals import final as fin
from cascade.evals.final import FinalGuardError, check_guard, run_final
from tests.eval_helpers import PLAN, git, init_repo, rewrite_matrix, write_synthetic
from tests.synthetic import gen

B = 300


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo")


@pytest.fixture
def data(tmp_path: Path) -> Path:
    return write_synthetic(tmp_path / "data", 0)


def go(repo: Path, data: Path, *, allow: bool = True, out: Path | None = None) -> fin.FinalOutcome:
    return run_final(
        repo_root=repo, data_dir=data, manifest=data / "manifest.jsonl",
        audit_file=data / "audit.jsonl", arms=gen.ARM_IDS, out_dir=out or repo.parent / "final-out",
        allow_synthetic=allow, resamples=B,
    )  # fmt: skip


def lock_of(repo: Path) -> dict[str, Any]:
    return json.loads((repo / fin.LOCK_NAME).read_text())


# --- the guard ----------------------------------------------------------------------------------


def test_the_guard_passes_for_a_committed_plan_and_a_clean_tree(repo: Path) -> None:
    info = check_guard(repo)
    assert info.commit == git(repo, "rev-parse", "HEAD")
    assert info.plan_blob == git(repo, "rev-parse", f"HEAD:{PLAN}")


def test_a_plan_that_was_never_committed_blocks_the_run(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "r", commit_plan=False)
    with pytest.raises(FinalGuardError, match="is not committed"):
        check_guard(repo)
    (repo / PLAN).parent.mkdir(parents=True)
    (repo / PLAN).write_text("# plan, but untracked\n")  # present on disk is not committed
    with pytest.raises(FinalGuardError, match="is not committed"):
        check_guard(repo)


def test_an_uncommitted_edit_to_the_plan_blocks_the_run(repo: Path) -> None:
    (repo / PLAN).write_text("# plan, edited after commit\n")
    with pytest.raises(FinalGuardError, match="working tree is not clean"):
        check_guard(repo)


def test_any_dirty_file_or_untracked_file_blocks_the_run(repo: Path) -> None:
    (repo / "README.md").write_text("changed\n")
    with pytest.raises(FinalGuardError, match="not clean"):
        check_guard(repo)
    git(repo, "checkout", "--", "README.md")
    (repo / "stray.txt").write_text("x\n")
    with pytest.raises(FinalGuardError, match="not clean") as e:
        check_guard(repo)
    assert "stray.txt" in str(e.value)  # the message names the offending file


def test_a_directory_that_is_not_a_git_repository_blocks_the_run(tmp_path: Path) -> None:
    (tmp_path / "plain").mkdir()
    with pytest.raises(FinalGuardError, match="not a git repository"):
        check_guard(tmp_path / "plain")


def test_an_existing_lock_blocks_the_run(repo: Path) -> None:
    (repo / fin.LOCK_NAME).write_text("{}\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "lock")
    with pytest.raises(FinalGuardError, match="has already been run"):
        check_guard(repo)


def test_a_lock_that_was_deleted_is_still_found_in_git_history(repo: Path) -> None:
    (repo / fin.LOCK_NAME).write_text("{}\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "lock")
    git(repo, "rm", "-q", fin.LOCK_NAME)
    git(repo, "commit", "-q", "-m", "remove the lock to run again")
    assert not (repo / fin.LOCK_NAME).exists()
    with pytest.raises(FinalGuardError, match="appears in git history"):
        check_guard(repo)


def test_a_lock_in_another_branchs_history_also_counts(repo: Path) -> None:
    git(repo, "checkout", "-q", "-b", "elsewhere")
    (repo / fin.LOCK_NAME).write_text("{}\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "lock on another branch")
    git(repo, "checkout", "-q", "main")
    with pytest.raises(FinalGuardError, match="appears in git history"):
        check_guard(repo)


# --- a successful run, once ---------------------------------------------------------------------


def test_the_run_writes_the_lock_the_outputs_and_the_stamp(repo: Path, data: Path) -> None:
    head = git(repo, "rev-parse", "HEAD")
    outcome = go(repo, data)
    lock = lock_of(repo)
    assert lock["status"] == "completed" and lock["commit"] == head
    assert lock["plan_blob"] == git(repo, "rev-parse", f"HEAD:{PLAN}")
    assert lock["matrix_hash"] == hashlib.sha256((data / "matrix.parquet").read_bytes()).hexdigest()
    assert lock["data_source"] == "synthetic" and lock["n_test_tasks"] == 17
    assert lock["margin"] == bs.margin_for(17) and lock["bootstrap_seed"] == bs.SEED_DEFAULT
    assert lock["n_train_tasks"] == 36
    for name in ("report.md", "results.json", "pareto.svg"):
        text = (outcome.out_dir / name).read_text()
        assert lock["outputs_sha256"][name] == hashlib.sha256(text.encode()).hexdigest()
    md = (outcome.out_dir / "report.md").read_text()
    assert "TEST (final)" in md and lock["matrix_hash"][:16] in md and head[:12] in md


def test_the_run_evaluates_only_test_tasks_after_fitting_on_train_only(
    repo: Path, data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cascade.evals import run as run_mod

    fitted_on: list[set[str]] = []
    original = run_mod.pol.fit_final_router

    def spy(records: Any, feature_set: str, margin: float) -> Any:
        fitted_on.append({r.repo for r in records})
        return original(records, feature_set, margin)

    monkeypatch.setattr(run_mod.pol, "fit_final_router", spy)
    outcome = go(repo, data)
    assert fitted_on and all(repos == {"syn/a", "syn/b", "syn/c"} for repos in fitted_on)  # train
    assert {x.repo for x in outcome.result.per_repo} == {"syn/d", "syn/e"}  # evaluated on test
    assert outcome.result.n_tasks == 17 and outcome.result.mode == "final"
    assert outcome.result.ablation == [] and outcome.result.noise == []  # dev-only analyses


def test_train_and_test_are_loaded_in_two_separate_calls_train_first(
    repo: Path, data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []
    original = fin.load_dataset

    def spy(**kw: Any) -> Any:
        calls.append(tuple(kw["splits"]))
        return original(**kw)

    monkeypatch.setattr(fin, "load_dataset", spy)
    go(repo, data)
    assert calls == [("train",), ("test",)]


def test_a_second_run_is_refused_even_after_the_lock_is_committed_or_removed(
    repo: Path, data: Path
) -> None:
    go(repo, data)
    with pytest.raises(FinalGuardError, match="not clean"):  # the lock file itself is untracked
        go(repo, data)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "record the final run")
    with pytest.raises(FinalGuardError, match="has already been run"):
        go(repo, data)
    git(repo, "rm", "-q", fin.LOCK_NAME)
    git(repo, "commit", "-q", "-m", "try to erase it")
    with pytest.raises(FinalGuardError, match="appears in git history"):
        go(repo, data)


def test_the_lock_is_created_exclusively_so_a_concurrent_run_cannot_start(
    repo: Path, data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = check_guard(repo)
    monkeypatch.setattr(fin, "check_guard", lambda _root: info)  # pretend the guard passed
    (repo / fin.LOCK_NAME).write_text("{}\n")  # but another run got there first
    with pytest.raises(FinalGuardError, match="already running"):
        go(repo, data)
    assert (repo / fin.LOCK_NAME).read_text() == "{}\n"  # the other run's lock is untouched


# --- refusals that must NOT use up the single run -----------------------------------------------


def test_synthetic_data_is_refused_without_the_test_only_flag_and_no_lock_is_written(
    repo: Path, data: Path
) -> None:
    with pytest.raises(FinalGuardError, match="needs real data"):
        go(repo, data, allow=False)
    assert not (repo / fin.LOCK_NAME).exists()


def test_a_mixed_tier_matrix_is_refused_before_the_lock_is_written(repo: Path, data: Path) -> None:
    def flip(rows: list[dict[str, Any]]) -> None:
        rows[0]["processing_tier"] = "flex"

    rewrite_matrix(data, flip)
    with pytest.raises(FinalGuardError, match="mixes processing tiers"):
        go(repo, data)
    assert not (repo / fin.LOCK_NAME).exists()  # the shot was not spent


def test_missing_data_refuses_before_the_lock_is_written(repo: Path, tmp_path: Path) -> None:
    with pytest.raises(FinalGuardError, match="NOT started"):
        go(repo, tmp_path / "no-such-dir")
    assert not (repo / fin.LOCK_NAME).exists()


# --- a failure after the lock is written uses up the run ----------------------------------------


def test_a_failure_during_evaluation_burns_the_single_run_and_is_not_retried(
    repo: Path, data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(**_kw: Any) -> Any:
        raise RuntimeError("numerical failure")

    monkeypatch.setattr(fin, "evaluate_dataset", boom)
    with pytest.raises(FinalGuardError, match="single run is used up"):
        go(repo, data)
    lock = lock_of(repo)
    assert lock["status"] == "failed" and "numerical failure" in lock["error"]
    assert lock["commit"] and lock["matrix_hash"]  # the lock was written BEFORE evaluating
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "record")
    monkeypatch.undo()
    with pytest.raises(FinalGuardError, match="has already been run"):
        go(repo, data)


def test_the_lock_exists_while_the_evaluation_is_running(
    repo: Path, data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    original = fin.evaluate_dataset

    def peek(**kw: Any) -> Any:
        seen.append(lock_of(repo)["status"])
        return original(**kw)

    monkeypatch.setattr(fin, "evaluate_dataset", peek)
    go(repo, data)
    assert seen == ["started"]


def test_the_margin_in_the_lock_follows_the_rule_for_the_number_of_test_tasks(
    repo: Path, data: Path
) -> None:
    go(repo, data)
    assert lock_of(repo)["margin"] == bs.margin_for(17) == 0.15
