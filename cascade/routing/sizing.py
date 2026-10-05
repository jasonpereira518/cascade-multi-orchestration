"""Sizing: what the pilot's measured costs say the full matrix will cost, and what is affordable.

Everything is computed from the pilot's logged ledger rows (bucket ``pilot``). Nothing here is an
estimate typed by hand, and nothing picks an option: ``cascade size`` prints a table and the human
decides. Method, so the numbers can be audited:

- cost per attempt: the sum of ``cost_usd`` over an attempt's logged calls (so an attempt cut short
  by a cap counts at the cost it incurred, and a retried attempt's earlier spend counts too);
- per arm: mean, median, and p90 over those attempts, p90 by linear interpolation between order
  statistics. With only ~10 attempts per arm the p90 is essentially the second-largest attempt, so
  it is a rough upper figure and not a stable estimate; the table says so;
- cost per task: the sum of the per-arm figures over the arms in the option (every arm runs on every
  task), at the mean and at the p90 of each arm;
- affordable N: floor(remaining matrix budget / cost per task) of NEW tasks. Pilot tasks already
  have every arm at standard tier, so under a standard-tier option they count as covered at no
  further cost; under flex they do not (a matrix uses one tier), so flex gets no pilot credit.
"""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cascade.config import ModelsConfig, ResolvedBudget

MIN_TASKS = 40


@dataclass(frozen=True)
class ArmCost:
    arm_id: str
    label: str | None
    attempts: int
    mean: float
    median: float
    p90: float


@dataclass(frozen=True)
class SizeOption:
    name: str
    arms: list[str]
    available: bool
    note: str
    per_task_mean: float
    per_task_p90: float
    new_tasks_mean: int
    new_tasks_p90: int
    covered_mean: int  # new tasks plus pilot tasks that need no further spend
    covered_p90: int


@dataclass(frozen=True)
class SizingReport:
    arms: list[ArmCost]
    budget_usd: float
    budget_basis: str
    audited_tasks: int
    pilot_tasks: int
    options: list[SizeOption]
    decision_required: bool


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 1]), the numpy default method."""
    if not values:
        raise ValueError("percentile of no values")
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def median(values: Sequence[float]) -> float:
    return percentile(values, 0.5)


def attempt_costs(ledger_db: Path, bucket: str) -> dict[str, list[float]]:
    """arm_id -> cost of each attempt in the bucket, from the ledger (read-only)."""
    conn = sqlite3.connect(f"file:{ledger_db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT arm_id, SUM(cost_usd) FROM calls WHERE bucket = ?"
            " GROUP BY task_id, attempt_id, arm_id",
            (bucket,),
        ).fetchall()
    finally:
        conn.close()
    out: dict[str, list[float]] = defaultdict(list)
    for arm_id, cost in rows:
        out[str(arm_id)].append(float(cost))
    return dict(out)


def bucket_task_ids(ledger_db: Path, bucket: str) -> set[str]:
    conn = sqlite3.connect(f"file:{ledger_db}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT DISTINCT task_id FROM calls WHERE bucket = ?", (bucket,))
        return {str(r[0]) for r in rows.fetchall()}
    finally:
        conn.close()


def arm_cost_stats(costs: dict[str, list[float]], models: ModelsConfig) -> list[ArmCost]:
    out: list[ArmCost] = []
    for arm in models.matrix_arms():
        xs = costs.get(arm.arm_id, [])
        if not xs:
            raise ValueError(f"no pilot attempts logged for arm {arm.arm_id}; run `cascade pilot`")
        mean = sum(xs) / len(xs)
        out.append(ArmCost(arm.arm_id, arm.label, len(xs), mean, median(xs), percentile(xs, 0.9)))
    return out


def _option(
    name: str,
    arms: list[ArmCost],
    models: ModelsConfig,
    *,
    budget_usd: float,
    pilot_tasks: int,
    flex: bool = False,
    note: str = "",
) -> SizeOption:
    mult: dict[str, float] = {}
    available = True
    if flex:
        for a in arms:
            model = next(m for m in models.arms if m.arm_id == a.arm_id).model
            factor = models.model_spec(model).flex_multiplier
            if factor is None:
                available = False
                note = f"not available: {model} has no flex_multiplier in models.yaml"
            else:
                mult[a.arm_id] = factor
    scale = {a.arm_id: mult.get(a.arm_id, 1.0) for a in arms}
    mean = sum(a.mean * scale[a.arm_id] for a in arms)
    p90 = sum(a.p90 * scale[a.arm_id] for a in arms)
    credit = 0 if flex else pilot_tasks
    new_mean = math.floor(budget_usd / mean) if mean > 0 and available else 0
    new_p90 = math.floor(budget_usd / p90) if p90 > 0 and available else 0
    return SizeOption(
        name=name, arms=[a.arm_id for a in arms], available=available, note=note,
        per_task_mean=mean, per_task_p90=p90, new_tasks_mean=new_mean, new_tasks_p90=new_p90,
        covered_mean=new_mean + credit if available else 0,
        covered_p90=new_p90 + credit if available else 0,
    )  # fmt: skip


def build_report(
    stats: list[ArmCost],
    models: ModelsConfig,
    budget: ResolvedBudget,
    *,
    matrix_spent_usd: float,
    global_spent_usd: float,
    audited_tasks: int,
    pilot_tasks: int,
) -> SizingReport:
    if "matrix" in budget.buckets:
        remaining = budget.buckets["matrix"] - matrix_spent_usd
        cap = budget.buckets["matrix"]
        basis = f"bucket 'matrix' cap ${cap:.2f} - spent ${matrix_spent_usd:.4f}"
    else:
        remaining = budget.global_cap_usd - global_spent_usd
        basis = f"global cap ${budget.global_cap_usd:.2f} - spent ${global_spent_usd:.4f}"
    remaining = max(remaining, 0.0)
    mid = next((a for a in stats if a.label == "mid"), None)
    if mid is None and len(stats) >= 3:
        mid = stats[len(stats) // 2]
    options = [
        _option(
            f"full {len(stats)}-arm matrix (standard tier)",
            stats,
            models,
            budget_usd=remaining,
            pilot_tasks=pilot_tasks,
        ),
        _option(
            "same, flex tier (cost x flex_multiplier)",
            stats,
            models,
            flex=True,
            budget_usd=remaining,
            pilot_tasks=pilot_tasks,
        ),
    ]
    if mid is not None:
        rest = [a for a in stats if a.arm_id != mid.arm_id]
        note = "a 2-arm matrix cannot answer questions about the mid arm"
        options.append(
            _option(
                f"drop the mid arm ({mid.arm_id})",
                rest,
                models,
                note=note,
                budget_usd=remaining,
                pilot_tasks=pilot_tasks,
            )
        )
    full = options[0]
    options.append(
        SizeOption(
            name=f"subset of tasks, full {len(stats)} arms (random prefix, standard tier)",
            arms=full.arms,
            available=True,
            note="the task-major order makes any prefix a random sample",
            per_task_mean=full.per_task_mean,
            per_task_p90=full.per_task_p90,
            new_tasks_mean=full.new_tasks_mean,
            new_tasks_p90=full.new_tasks_p90,
            covered_mean=min(full.covered_mean, audited_tasks),
            covered_p90=min(full.covered_p90, audited_tasks),
        )
    )
    return SizingReport(
        arms=stats, budget_usd=remaining, budget_basis=basis, audited_tasks=audited_tasks,
        pilot_tasks=pilot_tasks, options=options, decision_required=full.covered_p90 < MIN_TASKS,
    )  # fmt: skip


def format_report(r: SizingReport) -> str:
    lines = ["SIZING (from the pilot ledger; every figure is computed from logged calls)"]
    lines.append("cost per attempt, by arm:")
    lines.append(f"  {'arm':32s} {'n':>3s} {'mean':>9s} {'median':>9s} {'p90':>9s}")
    for a in r.arms:
        name = f"{a.label + ' = ' if a.label else ''}{a.arm_id}"
        lines.append(f"  {name:32s} {a.attempts:3d} ${a.mean:8.4f} ${a.median:8.4f} ${a.p90:8.4f}")
    lines.append("  (p90 over ~10 attempts is roughly the 2nd-largest attempt: a rough ceiling)")
    lines.append(f"remaining matrix budget: ${r.budget_usd:.2f}  [{r.budget_basis}]")
    lines.append(
        f"audited-ok tasks: {r.audited_tasks}   pilot tasks already complete: {r.pilot_tasks}"
        f"   minimum for the study: {MIN_TASKS}"
    )
    lines.append("")
    lines.append("OPTIONS (tasks that can be fully covered; 'covered' includes pilot tasks)")
    lines.append(
        f"  {'option':58s} {'$/task mean':>11s} {'$/task p90':>10s} "
        f"{'covered@mean':>12s} {'covered@p90':>11s}"
    )
    for o in r.options:
        if not o.available:
            lines.append(f"  {o.name:58s} {o.note}")
            continue
        lines.append(
            f"  {o.name:58s} ${o.per_task_mean:10.4f} ${o.per_task_p90:9.4f} "
            f"{o.covered_mean:12d} {o.covered_p90:11d}"
        )
        if o.note:
            lines.append(f"      note: {o.note}")
    lines.append("")
    full = r.options[0]
    if r.decision_required:
        lines.append(
            f"DECISION REQUIRED: the full {len(r.arms)}-arm matrix covers only {full.covered_p90} "
            f"tasks at p90 (and {full.covered_mean} at the mean), below the minimum of {MIN_TASKS}."
        )
        lines.append("No option has been chosen. Pick one (or change the budget) and tell me.")
    else:
        lines.append(
            f"The full {len(r.arms)}-arm matrix covers {full.covered_p90} tasks at p90 "
            f"(minimum {MIN_TASKS}). No option is chosen automatically."
        )
    return "\n".join(lines)
