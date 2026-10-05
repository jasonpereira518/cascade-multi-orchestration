"""The resumable matrix runner: ordering, resume, caps, stops, and status. Synthetic, no Docker.

``execute`` is a fake that makes real ledger entries (so spend and caps are real) and writes a real
matrix row, exactly as agents.runner.run_attempt does. Costs are deterministic: every cell costs
$0.10 (100k luna tokens at $1/M, or 10k sol tokens at $10/M). Per-arm attempt caps are 0.2 / 0.5 / 1.0,
so one task's three cells need 1.7 of headroom.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from cascade.agents.pipeline import PipelineConfig
from cascade.harness.ledger import Ledger
from cascade.routing.matrix import (
    Cell,
    MatrixRow,
    MatrixStatus,
    MatrixStore,
    TierMixError,
    config_id_of,
    interruption_reason,
    plan_cells,
    run_matrix,
)
from cascade.taskgen.schema import Task
from tests.matrix_helpers import (
    ARMS,
    CHEAP,
    MAX,
    MID,
    matrix_budget,
    new_ledger,
    row,
    spend,
    tasks_with_splits,
)

CFG = config_id_of(PipelineConfig().canonical())
TOKENS = {CHEAP: 100_000, MID: 100_000, MAX: 10_000}  # $0.10 each


def tasks(n: int = 6) -> list[Task]:
    return tasks_with_splits({f"t{i}": ("train", 10 + i) for i in range(n)})


class Harness:
    def __init__(self, tmp_path: Path, **budget_over: Any) -> None:
        self.tmp = tmp_path
        self.ledger: Ledger = new_ledger(tmp_path, **budget_over)
        self.budget = matrix_budget(tmp_path, **budget_over)
        self.store = MatrixStore(":memory:")
        self.calls: list[Cell] = []
        self.hook: Callable[[Cell, int], MatrixRow | None] = lambda _c, _n: None
        self.status_path = tmp_path / "matrix_status.json"

    def execute(
        self, bucket: str = "matrix", tier: str = "standard"
    ) -> Callable[[Cell], MatrixRow]:
        def run(cell: Cell) -> MatrixRow:
            self.calls.append(cell)
            override = self.hook(cell, len(self.calls))
            attempt = f"{cell.arm_id}|seed{cell.seed}|try{len(self.calls)}"
            cost = spend(
                self.ledger, cell.arm_id, TOKENS[cell.arm_id], task_id=cell.task_id,
                attempt_id=attempt, call_id=f"{cell.label}-{len(self.calls)}", bucket=bucket,
                tier=tier,
            )  # fmt: skip
            r = override or row(
                cell.task_id, cell.arm_id, seed=cell.seed, cost_usd=cost, processing_tier=tier
            )
            self.store.put(r)
            return r

        return run

    def run(
        self,
        cells: list[Cell],
        *,
        max_cells: int | None = None,
        tier: str = "standard",
        execute: Callable[[Cell], MatrixRow] | None = None,
    ) -> MatrixStatus:
        return run_matrix(
            cells=cells, config_id=CFG, store=self.store, ledger=self.ledger, budget=self.budget,
            execute=execute or self.execute(), bucket="matrix", tier=tier,
            status_path=self.status_path, max_cells=max_cells,
        )  # fmt: skip


@pytest.fixture
def h(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


# --- ordering -----------------------------------------------------------------------------------


def test_cells_are_task_major_with_arms_cheap_to_max() -> None:
    cells = plan_cells(tasks(4), ARMS, [0], shuffle_seed=0)
    assert len(cells) == 12
    for i in range(0, 12, 3):
        group = cells[i : i + 3]
        assert len({c.task_id for c in group}) == 1  # one task's arms are contiguous
        assert [c.arm_id for c in group] == list(ARMS)  # cheapest first


def test_the_task_order_is_a_seeded_shuffle_and_every_task_appears_once() -> None:
    ts = tasks(8)
    a = [c.task_id for c in plan_cells(ts, ARMS, [0], shuffle_seed=0)][::3]
    assert a == [c.task_id for c in plan_cells(list(reversed(ts)), ARMS, [0], shuffle_seed=0)][::3]
    assert sorted(a) == sorted(t.task_id for t in ts) and a != sorted(a)
    assert a != [c.task_id for c in plan_cells(ts, ARMS, [0], shuffle_seed=1)][::3]


def test_any_prefix_that_ends_on_a_task_boundary_has_complete_arm_coverage() -> None:
    cells = plan_cells(tasks(5), ARMS, [0], shuffle_seed=0)
    for k in range(1, 6):
        by_task: dict[str, set[str]] = {}
        for c in cells[: 3 * k]:
            by_task.setdefault(c.task_id, set()).add(c.arm_id)
        assert len(by_task) == k and all(arms == set(ARMS) for arms in by_task.values())


def test_with_several_seeds_each_task_runs_all_seeds_before_the_next_task() -> None:
    cells = plan_cells(tasks(3), (CHEAP, MID), [0, 1], shuffle_seed=0)
    assert [(c.seed, c.arm_id) for c in cells[:4]] == [(0, CHEAP), (0, MID), (1, CHEAP), (1, MID)]
    assert len({c.task_id for c in cells[:4]}) == 1


# --- running and resuming -----------------------------------------------------------------------


def test_a_full_run_executes_every_cell_once_and_writes_the_status(h: Harness) -> None:
    cells = plan_cells(tasks(4), ARMS, [0], shuffle_seed=0)
    status = h.run(cells)
    assert status.state == "complete" and status.reason == "complete"
    assert (status.planned, status.done, status.executed, status.remaining) == (12, 12, 12, 0)
    assert status.done_by_arm == {CHEAP: 4, MID: 4, MAX: 4}
    assert h.calls == cells and len(h.store.rows()) == 12
    on_disk = json.loads(h.status_path.read_text())
    assert on_disk["state"] == "complete" and on_disk["tier"] == "standard"
    assert on_disk["spent_usd"] == pytest.approx(1.2) and on_disk["next_cell"] is None
    assert on_disk["bucket"] == "matrix" and on_disk["bucket_cap_usd"] == 20.0
    assert on_disk["bucket_spent_usd"] == pytest.approx(1.2)
    assert not list(h.tmp.glob("*.tmp"))


def test_rerunning_a_finished_matrix_does_nothing(h: Harness) -> None:
    cells = plan_cells(tasks(3), ARMS, [0], shuffle_seed=0)
    h.run(cells)
    spent = h.ledger.spent_usd()
    again = h.run(cells)
    assert again.executed == 0 and again.done == 9 and len(h.calls) == 9
    assert h.ledger.spent_usd() == spent


def test_resume_after_a_simulated_crash_skips_finished_cells_and_loses_none(h: Harness) -> None:
    cells = plan_cells(tasks(4), ARMS, [0], shuffle_seed=0)

    def crash(_cell: Cell, n: int) -> MatrixRow | None:
        if n == 5:
            raise RuntimeError("power cut")
        return None

    h.hook = crash
    with pytest.raises(RuntimeError, match="power cut"):
        h.run(cells)
    finished = {(r.task_id, r.arm_id) for r in h.store.rows()}
    assert finished == {(c.task_id, c.arm_id) for c in cells[:4]}  # the crashed attempt left no row

    h.hook = lambda _c, _n: None
    h.calls.clear()
    status = h.run(cells)
    assert status.state == "complete" and status.done == 12
    assert h.calls == cells[4:]  # exactly the unfinished cells, in order
    rows = h.store.rows()
    assert len(rows) == 12 and len({(r.task_id, r.arm_id) for r in rows}) == 12


def test_max_cells_stops_cleanly_and_the_next_run_continues(h: Harness) -> None:
    cells = plan_cells(tasks(4), ARMS, [0], shuffle_seed=0)
    first = h.run(cells, max_cells=5)
    assert first.state == "stopped" and first.reason == "cell_limit"
    assert (first.done, first.executed, first.remaining) == (5, 5, 7)
    assert first.next_cell == cells[5].label
    second = h.run(cells)
    assert second.state == "complete" and second.executed == 7 and h.calls == cells


# --- caps and stops -----------------------------------------------------------------------------


def test_a_global_budget_stop_happens_at_a_task_boundary_never_mid_task(tmp_path: Path) -> None:
    h = Harness(tmp_path, global_cap_usd=2.2)  # a task needs 1.7 of headroom: 2 fit, a 3rd does not
    status = h.run(plan_cells(tasks(5), ARMS, [0], shuffle_seed=0))
    assert status.state == "stopped" and status.reason == "global_cap"
    assert status.executed == 6 and status.remaining == 9
    per_task: dict[str, int] = {}
    for c in h.calls:
        per_task[c.task_id] = per_task.get(c.task_id, 0) + 1
    assert sorted(per_task.values()) == [3, 3]  # complete arm coverage for what did run
    assert h.ledger.spent_usd() <= 2.2


def test_a_bucket_cap_stops_the_run_and_other_buckets_are_untouched(tmp_path: Path) -> None:
    h = Harness(tmp_path, buckets={"matrix": 2.2, "pilot": 10.0})
    status = h.run(plan_cells(tasks(5), ARMS, [0], shuffle_seed=0))
    assert status.reason == "bucket_cap" and status.executed == 6
    assert status.bucket_cap_usd == 2.2 and status.bucket_spent_usd == pytest.approx(0.6)
    assert h.ledger.bucket_spent_usd("pilot") == 0.0


def test_a_per_task_cap_below_a_tasks_planned_attempts_stops_before_any_biased_cell(
    tmp_path: Path,
) -> None:
    h = Harness(tmp_path, per_task_cap_usd=1.0)  # 0.2 + 0.5 + 1.0 > 1.0: the max arm would starve
    status = h.run(plan_cells(tasks(3), ARMS, [0], shuffle_seed=0))
    assert status.reason == "task_cap_too_low" and status.executed == 0 and h.calls == []


def test_the_kill_switch_stops_the_run_between_cells(h: Harness) -> None:
    def touch(_cell: Cell, n: int) -> MatrixRow | None:
        if n == 4:
            h.budget.kill_switch_path.touch()
        return None

    h.hook = touch
    cells = plan_cells(tasks(4), ARMS, [0], shuffle_seed=0)
    status = h.run(cells)
    assert status.reason == "kill_switch" and status.executed == 4 and status.remaining == 8
    h.budget.kill_switch_path.unlink()
    resumed = h.run(cells)
    assert resumed.state == "complete" and resumed.done == 12


def test_a_kill_switch_present_at_the_start_runs_nothing(h: Harness) -> None:
    h.budget.kill_switch_path.touch()
    status = h.run(plan_cells(tasks(2), ARMS, [0], shuffle_seed=0))
    assert status.reason == "kill_switch" and h.calls == [] and status.remaining == 6


def test_an_attempt_that_hit_its_own_cap_is_a_failed_cell_that_is_kept(h: Harness) -> None:
    cells = plan_cells(tasks(2), ARMS, [0], shuffle_seed=0)

    def capped(cell: Cell, n: int) -> MatrixRow | None:
        if n == 2:
            return row(
                cell.task_id, cell.arm_id, cost_usd=0.19,
                error="BudgetExceededError: attempt cap: spent $0.19 + worst-case $0.05 > cap $0.2",
            )  # fmt: skip
        return None

    h.hook = capped
    status = h.run(cells)
    assert status.state == "complete" and status.done == 6  # nothing dropped, the run went on
    kept = h.store.get(cells[1].task_id, cells[1].arm_id, 0, CFG)
    assert kept is not None and kept.true_success is False and kept.cost_usd == 0.19
    assert "attempt cap" in (kept.error or "")


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (None, None),
        ("", None),
        ("BudgetExceededError: attempt cap: spent", None),
        ("BudgetExceededError: task cap: spent", None),
        ("BudgetExceededError: attempt token cap: used", None),
        ("patch_error: x", None),
        ("BudgetExceededError: global cap: spent $1", "global_cap"),
        ("BudgetExceededError: bucket cap: spent $1", "bucket_cap"),
        ("KillSwitchEngagedError: kill switch present at KILL_SWITCH", "kill_switch"),
    ],
)
def test_only_global_bucket_and_kill_switch_endings_count_as_interruptions(
    error: str | None, reason: str | None
) -> None:
    assert interruption_reason(error) == reason


def test_an_attempt_cut_short_by_a_global_stop_is_not_recorded_and_reruns_on_resume(
    h: Harness,
) -> None:
    cells = plan_cells(tasks(2), ARMS, [0], shuffle_seed=0)

    def interrupted(cell: Cell, n: int) -> MatrixRow | None:
        if n == 2:
            return row(cell.task_id, cell.arm_id, error="BudgetExceededError: global cap: spent")
        return None

    h.hook = interrupted
    status = h.run(cells)
    assert status.reason == "global_cap" and status.done == 1 and status.remaining == 5
    assert (
        h.store.get(cells[1].task_id, cells[1].arm_id, 0, CFG) is None
    )  # not baked in as a failure
    assert h.ledger.spent_usd() > 0  # the attempt's spend stays in the ledger

    h.hook = lambda _c, _n: None
    h.calls.clear()
    resumed = h.run(cells)
    assert resumed.state == "complete" and h.calls == cells[1:]


# --- one tier per matrix ------------------------------------------------------------------------


def test_existing_rows_on_another_tier_block_the_run_before_anything_executes(h: Harness) -> None:
    cells = plan_cells(tasks(2), ARMS, [0], shuffle_seed=0)
    h.store.put(row(cells[0].task_id, cells[0].arm_id, processing_tier="flex"))
    with pytest.raises(TierMixError, match="one matrix, one tier"):
        h.run(cells)
    assert h.calls == []


def test_the_status_records_the_tier(h: Harness) -> None:
    cells = plan_cells(tasks(1), ARMS, [0], shuffle_seed=0)
    status = h.run(cells, tier="flex", execute=h.execute(tier="flex"))
    assert status.tier == "flex"
    assert {r.processing_tier for r in h.store.rows()} == {"flex"}
