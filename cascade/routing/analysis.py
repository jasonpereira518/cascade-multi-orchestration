"""Paired comparisons over matrix rows: ablations (same seed, different pipeline) and the repeat
subset (same pipeline, different seed).

Both ask "how much does the outcome move when one thing changes on the same tasks", so they share
one function. Only tasks that have BOTH rows are compared, and the count of such tasks is always
reported, because an ablation that ran on 20 tasks but has a baseline for 12 is a 12-task
comparison.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from cascade.agents.pipeline import PipelineConfig
from cascade.routing.matrix import MatrixRow


def ablation_configs(base: PipelineConfig) -> dict[str, PipelineConfig]:
    """The two ablations of the task: the full pipeline minus one optional role."""
    return {
        "no_reproducer": PipelineConfig(
            reproducer=False, reviewer=base.reviewer, max_repairs=base.max_repairs,
            max_output_tokens=base.max_output_tokens, test_timeout_s=base.test_timeout_s,
            max_test_runs_per_role=base.max_test_runs_per_role,
        ),
        "no_reviewer": PipelineConfig(
            reproducer=base.reproducer, reviewer=False, max_repairs=base.max_repairs,
            max_output_tokens=base.max_output_tokens, test_timeout_s=base.test_timeout_s,
            max_test_runs_per_role=base.max_test_runs_per_role,
        ),
    }  # fmt: skip


@dataclass(frozen=True)
class Paired:
    arm_id: str
    label: str
    tasks_with_both: int
    base_pass: int
    other_pass: int
    both_pass: int
    only_base: int  # passed in the baseline, failed in the other
    only_other: int
    base_claimed: int
    other_claimed: int
    base_cost: float  # mean cost per attempt over the paired tasks
    other_cost: float

    @property
    def agreement(self) -> float | None:
        if not self.tasks_with_both:
            return None
        disagree = self.only_base + self.only_other
        return 1 - disagree / self.tasks_with_both


def pair(
    rows: Sequence[MatrixRow],
    *,
    arm_id: str,
    label: str,
    task_ids: Collection[str],
    base: tuple[int, str],  # (seed, pipeline_config JSON)
    other: tuple[int, str],
) -> Paired:
    def index(seed: int, config: str) -> dict[str, MatrixRow]:
        return {
            r.task_id: r
            for r in rows
            if r.arm_id == arm_id and r.seed == seed and r.pipeline_config == config
            and r.task_id in task_ids
        }  # fmt: skip

    a, b = index(*base), index(*other)
    both = sorted(set(a) & set(b))
    n = len(both)
    return Paired(
        arm_id=arm_id,
        label=label,
        tasks_with_both=n,
        base_pass=sum(a[t].true_success for t in both),
        other_pass=sum(b[t].true_success for t in both),
        both_pass=sum(a[t].true_success and b[t].true_success for t in both),
        only_base=sum(a[t].true_success and not b[t].true_success for t in both),
        only_other=sum(b[t].true_success and not a[t].true_success for t in both),
        base_claimed=sum(a[t].verifier_claimed for t in both),
        other_claimed=sum(b[t].verifier_claimed for t in both),
        base_cost=sum(a[t].cost_usd for t in both) / n if n else 0.0,
        other_cost=sum(b[t].cost_usd for t in both) / n if n else 0.0,
    )


def format_paired(title: str, base_name: str, results: Sequence[Paired]) -> str:
    lines = [title]
    lines.append(
        f"  {'comparison':34s} {'paired':>6s} {base_name + ' pass':>14s} {'other pass':>11s} "
        f"{'base-only':>9s} {'other-only':>10s} {'cost base':>10s} {'cost other':>10s}"
    )
    for p in results:
        lines.append(
            f"  {p.label:34s} {p.tasks_with_both:6d} {p.base_pass:14d} {p.other_pass:11d} "
            f"{p.only_base:9d} {p.only_other:10d} ${p.base_cost:9.4f} ${p.other_cost:9.4f}"
        )
    if any(p.tasks_with_both == 0 for p in results):
        lines.append("  (0 paired tasks: the baseline rows for these tasks do not exist yet)")
    lines.append("  'pass' counts are over the paired tasks only; small n, so read as counts.")
    return "\n".join(lines)
