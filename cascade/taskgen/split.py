"""The train/test split: whole repos held out, assigned by hash, frozen in a committed file.

CLAUDE.md rule 4: the split is frozen by hash and by repo, and the test split is never used for
fitting, tuning, or selection. How that is enforced here:

- **By repo.** A repo is entirely train or entirely test, so no repo's idiosyncrasies leak across.
- **By hash.** Repos are ordered by SHA-256 of ``salt:repo``, never by anything about outcomes. The
  held-out set is the prefix of that order whose task count is closest to ``target_test_fraction``.
  (Python's ``hash()`` is not used: it is randomized per process.)
- **Frozen.** The assignment is written once to ``tasks/split.json`` and never recomputed.
  ``freeze`` refuses to overwrite it (``force`` exists, and invalidates any result that used the
  old split).
  Repos that appear later get a hash-bucket assignment appended; existing entries never change.
- **Guarded.** ``fitting_tasks`` returns train tasks only and refuses an unsplit manifest.

With a handful of repos the split is coarse by nature (one repo can be most of the tasks); that is
the price of repo-level holdout and is why the result reports per-repo counts.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from cascade.errors import CascadeError
from cascade.taskgen.schema import Task

SPLIT_VERSION = 1
DEFAULT_SALT = "cascade-split-v1"
DEFAULT_TARGET = 0.4
Side = Literal["train", "test"]


class SplitError(CascadeError):
    """The split cannot be made, read, or safely used."""


class SplitFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    salt: str
    target_test_fraction: float
    frozen_at: str
    counts_at_freeze: dict[str, int]
    assignments: dict[str, Side]
    # repos that appeared after the freeze, with the time each was assigned
    late_assignments: dict[str, str] = {}


def _digest(repo: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{repo}".encode()).hexdigest()


def bucket(repo: str, salt: str) -> float:
    """A repo's position in [0, 1), a pure function of (salt, repo)."""
    return int(_digest(repo, salt)[:12], 16) / 16**12


def hash_order(repos: Sequence[str], salt: str) -> list[str]:
    return sorted(repos, key=lambda r: (_digest(r, salt), r))


def choose_test_repos(
    counts: Mapping[str, int], *, salt: str, target_test_fraction: float
) -> list[str]:
    """Hash-order prefix of repos whose share of tasks is closest to the target (ties: shorter)."""
    present = {r: n for r, n in counts.items() if n > 0}
    if len(present) < 2:
        raise SplitError("a repo-level split needs tasks from at least two repos")
    if not 0 < target_test_fraction < 1:
        raise SplitError("target_test_fraction must be strictly between 0 and 1")
    order = hash_order(list(present), salt)
    total = sum(present.values())
    best_k, best_gap, running = 1, float("inf"), 0
    for k in range(1, len(order)):  # at least one repo stays in train
        running += present[order[k - 1]]
        gap = abs(running / total - target_test_fraction)
        if gap < best_gap - 1e-12:
            best_k, best_gap = k, gap
    return order[:best_k]


def freeze(
    counts: Mapping[str, int],
    *,
    path: Path,
    salt: str = DEFAULT_SALT,
    target_test_fraction: float = DEFAULT_TARGET,
    now: dt.datetime | None = None,
    force: bool = False,
) -> SplitFile:
    if path.exists() and not force:
        raise SplitError(
            f"{path} already exists: the split is frozen. Changing it invalidates every result "
            "computed with the old split; pass --force only if that is intended."
        )
    test = set(choose_test_repos(counts, salt=salt, target_test_fraction=target_test_fraction))
    split = SplitFile(
        version=SPLIT_VERSION,
        salt=salt,
        target_test_fraction=target_test_fraction,
        frozen_at=(now or dt.datetime.now(dt.UTC)).isoformat(),
        counts_at_freeze={r: n for r, n in sorted(counts.items())},
        assignments={
            r: ("test" if r in test else "train") for r in sorted(counts) if counts[r] > 0
        },
    )
    write_split(path, split)
    return split


def read_split(path: Path) -> SplitFile:
    if not path.exists():
        raise SplitError(f"no split at {path}; run `cascade split freeze` first")
    try:
        split = SplitFile.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise SplitError(f"unreadable split file {path}: {e}") from e
    if split.version != SPLIT_VERSION:
        raise SplitError(f"split file version {split.version} is not supported")
    return split


def write_split(path: Path, split: SplitFile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(split.model_dump(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def assign_new_repo(split: SplitFile, repo: str, *, now: dt.datetime | None = None) -> SplitFile:
    """Append an assignment for a repo added after the freeze. Existing entries never change."""
    if repo in split.assignments:
        return split
    side: Side = "test" if bucket(repo, split.salt) < split.target_test_fraction else "train"
    stamp = (now or dt.datetime.now(dt.UTC)).isoformat()
    return split.model_copy(
        update={
            "assignments": {**split.assignments, repo: side},
            "late_assignments": {**split.late_assignments, repo: stamp},
        }
    )


def apply_split(tasks: Sequence[Task], split: SplitFile) -> list[Task]:
    """Tasks with ``split`` set from the frozen assignments. Every repo must already be assigned."""
    missing = sorted({t.repo for t in tasks if t.repo not in split.assignments})
    if missing:
        raise SplitError(f"repos without a frozen assignment: {', '.join(missing)}")
    return [t.model_copy(update={"split": split.assignments[t.repo]}) for t in tasks]


def fitting_tasks(tasks: Sequence[Task]) -> list[Task]:
    """The only tasks anything may be fitted, tuned, or selected on: the train split.

    Refuses a manifest where any task has no split yet, so an unsplit manifest can never be
    mistaken for training data.
    """
    unsplit = [t.task_id for t in tasks if t.split is None]
    if unsplit:
        raise SplitError(
            f"{len(unsplit)} task(s) have no split assigned (e.g. {unsplit[0]}); "
            "apply the split first"
        )
    return [t for t in tasks if t.split == "train"]
