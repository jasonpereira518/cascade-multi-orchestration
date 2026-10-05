"""Which tasks an experiment may touch, and in what order.

Two rules live here so no command can break them:

- Only audited-ok tasks are eligible at all (docs/SPEC.md).
- The pilot, the ablations, and the repeat subset draw from the TRAIN split only. ``draw_train``
  refuses a pool containing an unsplit task and re-checks its own output, so the test split cannot
  be reached through it even by a bug upstream (CLAUDE.md hard rule 4).

Draws are deterministic functions of (task ids, seed, purpose), stratified so a small sample covers
every repo and every size band instead of whatever a plain shuffle happens to give.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from cascade.errors import CascadeError
from cascade.taskgen.audit import AuditVerdict, experiment_tasks
from cascade.taskgen.schema import Task

SIZE_BANDS = ("small", "medium", "large")


class SelectionError(CascadeError):
    """A task set cannot be drawn safely or at the requested size."""


def _h(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def size_bands(pool: Sequence[Task]) -> dict[str, str]:
    """Terciles of the model-free size feature (changed source lines) within the pool."""
    ranked = sorted(pool, key=lambda t: (t.features.src_lines, t.task_id))
    n = len(ranked)
    return {t.task_id: SIZE_BANDS[min(i * 3 // n, 2)] for i, t in enumerate(ranked)} if n else {}


def stratified_sample(pool: Sequence[Task], n: int, *, seed: int, purpose: str) -> list[Task]:
    """Up to ``n`` tasks, round-robin over (repo, size band) strata, deterministic."""
    if n < 0:
        raise SelectionError("sample size must not be negative")
    bands = size_bands(pool)
    strata: dict[tuple[str, str], list[Task]] = {}
    for t in pool:
        strata.setdefault((t.repo, bands[t.task_id]), []).append(t)
    ordered = sorted(strata.items(), key=lambda kv: _h(seed, purpose, "stratum", *kv[0]))
    queues = [
        sorted(members, key=lambda t: _h(seed, purpose, "task", t.task_id))
        for _key, members in ordered
    ]
    chosen: list[Task] = []
    while len(chosen) < n and any(queues):
        for q in queues:
            if q and len(chosen) < n:
                chosen.append(q.pop(0))
    return chosen


def audited_ok(tasks: Sequence[Task], verdicts: Mapping[str, AuditVerdict]) -> list[Task]:
    """Tasks whose latest audit verdict is ok. Nothing else may enter an experiment."""
    return experiment_tasks(tasks, verdicts)


def train_pool(tasks: Sequence[Task], verdicts: Mapping[str, AuditVerdict]) -> list[Task]:
    ok = audited_ok(tasks, verdicts)
    unsplit = [t.task_id for t in ok if t.split is None]
    if unsplit:
        raise SelectionError(
            f"{len(unsplit)} audited-ok task(s) have no split (e.g. {unsplit[0]}); run "
            "`cascade tasks finalize` after freezing the split"
        )
    return [t for t in ok if t.split == "train"]


def draw_train(
    tasks: Sequence[Task],
    verdicts: Mapping[str, AuditVerdict],
    n: int,
    *,
    seed: int,
    purpose: str,
) -> list[Task]:
    """Up to ``n`` audited-ok TRAIN tasks, stratified by repo and size. Never touches test."""
    pool = train_pool(tasks, verdicts)
    if not pool:
        raise SelectionError("no audited-ok train tasks to draw from")
    drawn = stratified_sample(pool, n, seed=seed, purpose=purpose)
    if any(t.split != "train" for t in drawn):
        raise SelectionError("internal error: a non-train task was drawn")  # belt and braces
    return drawn


def task_major_order(tasks: Sequence[Task], seed: int) -> list[Task]:
    """A fixed shuffle: any prefix of the order is a random sample of the tasks."""
    return sorted(tasks, key=lambda t: _h(seed, "matrix-order", t.task_id))
