"""The outcome matrix: one row per (task, arm, seed, pipeline config), both outcomes recorded.

Row fields are those in docs/SPEC.md. Everything numeric is computed from the logged ledger calls of
the attempt (rule 1: no hand-entered or estimated metrics). ``true_success`` is graded against the
hidden tests; ``verifier_claimed`` is the agent's own evidence. Keeping both per attempt is what
allows Verifier false-positive analysis and the cascade policy later.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from cascade.config import ResolvedBudget
from cascade.errors import CascadeError
from cascade.harness.ledger import Ledger
from cascade.routing.selection import task_major_order
from cascade.taskgen.schema import Task


class MatrixRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    arm_id: str
    seed: int
    true_success: bool
    verifier_claimed: bool
    cost_usd: Annotated[float, Field(ge=0)]
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    reasoning_tokens: Annotated[int, Field(ge=0)]
    wall_s: Annotated[float, Field(ge=0)]
    turns: Annotated[int, Field(ge=0)]  # LLM calls made
    patch_sha: str | None
    pipeline_config: str  # canonical JSON of the PipelineConfig
    error: str | None
    # Added in Phase 4. Defaults let rows written before then still load.
    model: str = ""
    reasoning_effort: str | None = None
    processing_tier: str = "standard"
    cached_tokens: Annotated[int, Field(ge=0)] = 0
    cache_write_tokens: Annotated[int, Field(ge=0)] = 0
    started_at: str = ""  # ISO-8601 UTC, when the attempt began

    @property
    def config_id(self) -> str:
        return config_id_of(self.pipeline_config)


def config_id_of(pipeline_config: str) -> str:
    return hashlib.sha256(pipeline_config.encode()).hexdigest()[:10]


class MatrixStore:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS matrix (task_id TEXT NOT NULL, arm_id TEXT NOT NULL,"
            " seed INTEGER NOT NULL, config_id TEXT NOT NULL, row TEXT NOT NULL,"
            " PRIMARY KEY (task_id, arm_id, seed, config_id))"
        )

    def close(self) -> None:
        self._conn.close()

    def put(self, row: MatrixRow) -> None:
        self._conn.execute(
            "INSERT INTO matrix(task_id, arm_id, seed, config_id, row) VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(task_id, arm_id, seed, config_id) DO UPDATE SET row = excluded.row",
            (row.task_id, row.arm_id, row.seed, row.config_id, row.model_dump_json()),
        )

    def delete(self, task_id: str, arm_id: str, seed: int, config_id: str) -> None:
        """Remove one cell's row. Used only for an attempt cut short by a global stop (runner)."""
        self._conn.execute(
            "DELETE FROM matrix WHERE task_id = ? AND arm_id = ? AND seed = ? AND config_id = ?",
            (task_id, arm_id, seed, config_id),
        )

    def get(self, task_id: str, arm_id: str, seed: int, config_id: str) -> MatrixRow | None:
        found = self._conn.execute(
            "SELECT row FROM matrix"
            " WHERE task_id = ? AND arm_id = ? AND seed = ? AND config_id = ?",
            (task_id, arm_id, seed, config_id),
        ).fetchone()
        return None if found is None else MatrixRow.model_validate_json(found["row"])

    def rows(self) -> list[MatrixRow]:
        cur = self._conn.execute("SELECT row FROM matrix ORDER BY task_id, arm_id, seed, config_id")
        return [MatrixRow.model_validate_json(r["row"]) for r in cur.fetchall()]


# --- the resumable runner ------------------------------------------------------------------------
#
# The runner decides WHAT to run and WHEN TO STOP; the ``execute`` callback does the running (the
# CLI passes agents.runner.run_attempt), which keeps this logic testable with no Docker or model.
#
# Order: tasks are shuffled with a fixed seed and processed task-major, every arm of a task (cheap
# to max, the order of ``arm_ids``) before the next task, so any prefix of a run is a random sample
# of tasks with complete arm coverage. Before a task's first pending cell the runner checks that the
# global, bucket, and per-task budgets can cover ALL its pending cells at their per-attempt caps; if
# not it stops there, so a budget stop can never leave a task with only some of its arms.
#
# Caps: the per-attempt cap (per arm) is enforced by the ledger on every call. An attempt that hits
# it is a normal failed cell, recorded with the cost incurred, never dropped. An attempt cut short
# by something else (the kill switch, or the global or bucket cap tripping mid-attempt) says nothing
# about the model, so its row is removed and the cell reruns on resume; its spend stays in the
# ledger and counts against every cap.

STOP_COMPLETE = "complete"
STOP_GLOBAL = "global_cap"
STOP_BUCKET = "bucket_cap"
STOP_TASK = "task_cap_too_low"
STOP_KILL = "kill_switch"
STOP_LIMIT = "cell_limit"


class BudgetShapeError(CascadeError):
    """The budget cannot cover a task's planned attempts, so cells would be cut by a cap."""


class TierMixError(CascadeError):
    """A matrix must use one processing tier; existing rows use another."""


@dataclass(frozen=True)
class Cell:
    task_id: str
    arm_id: str
    seed: int

    @property
    def label(self) -> str:
        return f"{self.task_id}|{self.arm_id}|seed{self.seed}"


def plan_cells(
    tasks: Sequence[Task], arm_ids: Sequence[str], seeds: Sequence[int], *, shuffle_seed: int
) -> list[Cell]:
    """Task-major cells: shuffled tasks; per task every seed; per seed every arm, in order."""
    return [
        Cell(t.task_id, arm, seed)
        for t in task_major_order(tasks, shuffle_seed)
        for seed in seeds
        for arm in arm_ids
    ]


def interruption_reason(error: str | None) -> str | None:
    """Why an attempt ended for a reason unrelated to the model, or None for an ordinary outcome."""
    if not error:
        return None
    if "KillSwitchEngagedError" in error:
        return STOP_KILL
    if "BudgetExceededError: global cap" in error:
        return STOP_GLOBAL
    if "BudgetExceededError: bucket cap" in error:
        return STOP_BUCKET
    return None


@dataclass
class MatrixStatus:
    state: str  # "complete" or "stopped"
    reason: str
    tier: str
    bucket: str
    planned: int
    done: int  # cells with a row, including ones finished in earlier runs
    executed: int  # cells finished by this invocation
    remaining: int
    done_by_arm: dict[str, int]
    spent_usd: float
    global_cap_usd: float
    bucket_spent_usd: float | None
    bucket_cap_usd: float | None
    next_cell: str | None
    updated_at: str


def write_status(path: Path, status: MatrixStatus) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(status), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_matrix(
    *,
    cells: Sequence[Cell],
    config_id: str,
    store: MatrixStore,
    ledger: Ledger,
    budget: ResolvedBudget,
    execute: Callable[[Cell], MatrixRow],
    bucket: str,
    tier: str,
    status_path: Path,
    max_cells: int | None = None,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
) -> MatrixStatus:
    existing = {c: store.get(c.task_id, c.arm_id, c.seed, config_id) for c in cells}
    mixed = sorted(
        {r.processing_tier for r in existing.values() if r and r.processing_tier != tier}
    )
    if mixed:
        raise TierMixError(
            f"existing rows use tier {mixed} but this run uses {tier!r}; one matrix, one tier"
        )
    done = {c for c, r in existing.items() if r is not None}
    executed = 0
    stop = STOP_COMPLETE
    bucket_cap = budget.buckets.get(bucket)

    def snapshot() -> MatrixStatus:
        by_arm: dict[str, int] = {}
        for c in done:
            by_arm[c.arm_id] = by_arm.get(c.arm_id, 0) + 1
        pending = [c for c in cells if c not in done]
        status = MatrixStatus(
            state="complete" if not pending else "stopped",
            reason=STOP_COMPLETE if not pending else stop,
            tier=tier,
            bucket=bucket,
            planned=len(cells),
            done=len(done),
            executed=executed,
            remaining=len(pending),
            done_by_arm=dict(sorted(by_arm.items())),
            spent_usd=ledger.spent_usd(),
            global_cap_usd=budget.global_cap_usd,
            bucket_spent_usd=ledger.bucket_spent_usd(bucket) if bucket_cap is not None else None,
            bucket_cap_usd=bucket_cap,
            next_cell=pending[0].label if pending else None,
            updated_at=now().isoformat(),
        )
        write_status(status_path, status)
        return status

    group_checked: tuple[str, int] | None = None
    for cell in cells:
        if cell in done:
            continue
        if budget.kill_switch_path.exists():
            stop = STOP_KILL
            break
        if max_cells is not None and executed >= max_cells:
            stop = STOP_LIMIT
            break
        group = (cell.task_id, cell.seed)
        if group != group_checked:
            pending_group = [c for c in cells if (c.task_id, c.seed) == group and c not in done]
            need = sum(ledger.attempt_cap(c.arm_id) for c in pending_group)
            problem = _group_budget_problem(ledger, budget, bucket, bucket_cap, cell.task_id, need)
            if problem is not None:
                stop = problem
                break
            group_checked = group
        row = execute(cell)
        reason = interruption_reason(row.error)
        if reason is not None:
            store.delete(cell.task_id, cell.arm_id, cell.seed, config_id)
            stop = reason
            break
        done.add(cell)
        executed += 1
        snapshot()
    return snapshot()


def _group_budget_problem(
    ledger: Ledger,
    budget: ResolvedBudget,
    bucket: str,
    bucket_cap: float | None,
    task_id: str,
    need: float,
) -> str | None:
    if ledger.spent_usd() + need > budget.global_cap_usd + 1e-12:
        return STOP_GLOBAL
    if bucket_cap is not None and ledger.bucket_spent_usd(bucket) + need > bucket_cap + 1e-12:
        return STOP_BUCKET
    if ledger.spent_usd(task_id=task_id) + need > budget.per_task_cap_usd + 1e-12:
        return STOP_TASK
    return None
