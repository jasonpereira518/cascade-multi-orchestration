"""The single test-split evaluation, and the guard that makes it single.

CLAUDE.md hard rule 4: the test split is never used for fitting, tuning, or selection. This
module is the only code that evaluates on it, and it can do so once. In order, ``run_final``:

1. checks the guard: ``docs/EVAL_PLAN.md`` is committed (tracked and present in HEAD), the working
   tree is clean (so the code that runs is the code that is committed), and no lock exists, neither
   in the tree nor anywhere in git history (so deleting the lock file and rerunning is refused too);
2. loads the data and refuses real-looking mistakes BEFORE spending the shot: synthetic data
   (unless a test says otherwise), mixed processing tiers, no train or no test tasks;
3. writes the lock (exclusive create) with the commit, the plan's blob hash, and the matrix hash,
   BEFORE evaluating, so a crash or a bad result cannot be retried;
4. fits the router on TRAIN only, evaluates the test tasks once, writes the report, and records the
   output hashes in the lock.

If step 4 fails the lock stays, marked ``failed``: the shot is used up on purpose. Recovering means
a deliberate human act (a recorded deviation from the plan), never an automatic retry.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cascade.errors import CascadeError
from cascade.evals import bootstrap as bs
from cascade.evals import report
from cascade.evals.data import DataError, load_dataset
from cascade.evals.run import EvalResult, evaluate_dataset

PLAN_PATH = "docs/EVAL_PLAN.md"
LOCK_NAME = "FINAL.lock.json"
_GIT = "git"


class FinalGuardError(CascadeError):
    """The final evaluation is not allowed to run (yet, or ever again)."""


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603  (fixed argv, no shell)
        [_GIT, "-C", str(repo_root), *args], capture_output=True, text=True, check=False
    )


@dataclass(frozen=True)
class GuardInfo:
    commit: str
    plan_blob: str


def check_guard(repo_root: Path, *, plan: str = PLAN_PATH, lock: str = LOCK_NAME) -> GuardInfo:
    head = _git(repo_root, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise FinalGuardError(f"{repo_root} is not a git repository with a commit")
    in_head = _git(repo_root, "cat-file", "-e", f"HEAD:{plan}")
    if in_head.returncode != 0:
        raise FinalGuardError(
            f"{plan} is not committed. Commit the evaluation plan before the final run, so what "
            "was decided is fixed before the test split is touched."
        )
    status = _git(repo_root, "status", "--porcelain")
    if status.stdout.strip():
        shown = "\n  ".join(status.stdout.strip().splitlines()[:8])
        raise FinalGuardError(
            "the working tree is not clean; commit or stash first so the code that runs is the "
            f"code that is committed:\n  {shown}"
        )
    if (repo_root / lock).exists():
        raise FinalGuardError(f"{lock} exists: the final evaluation has already been run")
    history = _git(repo_root, "log", "--all", "--oneline", "--", lock)
    if history.stdout.strip():
        raise FinalGuardError(
            f"{lock} appears in git history: the final evaluation was already run (and the lock "
            "removed). It cannot be run again."
        )
    blob = _git(repo_root, "rev-parse", f"HEAD:{plan}").stdout.strip()
    return GuardInfo(head.stdout.strip(), blob)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _write_lock(path: Path, lock: dict[str, Any]) -> None:
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class FinalOutcome:
    result: EvalResult
    lock_path: Path
    out_dir: Path


def run_final(
    *,
    repo_root: Path,
    data_dir: Path,
    manifest: Path,
    audit_file: Path,
    arms: Mapping[str, str],
    out_dir: Path,
    allow_synthetic: bool = False,
    resamples: int = bs.B_DEFAULT,
    seed: int = bs.SEED_DEFAULT,
) -> FinalOutcome:
    info = check_guard(repo_root)
    try:
        train = load_dataset(
            data_dir=data_dir, manifest=manifest, audit_file=audit_file, arms=arms,
            splits=("train",), repo_root=repo_root,
        )  # fmt: skip
        test = load_dataset(
            data_dir=data_dir, manifest=manifest, audit_file=audit_file, arms=arms,
            splits=("test",), repo_root=repo_root,
        )  # fmt: skip
    except DataError as e:
        raise FinalGuardError(f"the final run was NOT started and no lock was written: {e}") from e
    if test.provenance.data_source != "real" and not allow_synthetic:
        raise FinalGuardError(
            "the final run needs real data; this matrix is "
            f"{test.provenance.data_source}. Nothing was started and no lock was written."
        )
    n_test = test.audited_by_split.get("test", 0)
    margin = bs.margin_for(n_test)
    basis = f"rule of docs/EVAL_PLAN.md for {n_test} audited test tasks"

    lock_path = repo_root / LOCK_NAME
    lock: dict[str, Any] = {
        "commit": info.commit, "plan_blob": info.plan_blob,
        "matrix_hash": test.provenance.matrix_hash, "data_source": test.provenance.data_source,
        "n_test_tasks": n_test, "n_train_tasks": len(train.records), "margin": margin,
        "bootstrap_resamples": resamples, "bootstrap_seed": seed, "status": "started",
        "started_at": dt.datetime.now(dt.UTC).isoformat(),
    }  # fmt: skip
    try:
        with lock_path.open("x", encoding="utf-8") as f:  # exclusive: a second run cannot start
            f.write(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    except FileExistsError as e:
        raise FinalGuardError(
            f"{LOCK_NAME} appeared: the final evaluation is already running"
        ) from e

    try:
        result = evaluate_dataset(
            mode="final", train=train, target=test, margin=margin, margin_basis=basis,
            b=resamples, seed=seed,
        )  # fmt: skip
        out_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "report.md": report.to_markdown(result),
            "results.json": report.to_json(result),
            "pareto.svg": report.to_svg(result),
        }
        for name, text in files.items():
            (out_dir / name).write_text(text, encoding="utf-8")
        lock.update(
            status="completed", completed_at=dt.datetime.now(dt.UTC).isoformat(),
            outputs_sha256={n: _sha256(t) for n, t in files.items()},
        )  # fmt: skip
    except Exception as e:
        lock.update(status="failed", error=f"{type(e).__name__}: {e}")
        _write_lock(lock_path, lock)
        raise FinalGuardError(
            f"the final evaluation failed after the lock was written ({type(e).__name__}: {e}); "
            "the single run is used up and will not be retried automatically"
        ) from e
    _write_lock(lock_path, lock)
    return FinalOutcome(result, lock_path, out_dir)
