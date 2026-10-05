"""Calibration: do the arms discriminate on this task set, and is the Verifier trustworthy?

Computed from the pilot's matrix rows only (default pipeline config, the base seed, pilot tasks).
With ~10 tasks every rate is noisy, so counts and a Wilson 95% interval are always shown; the flags
are prompts for a human decision, never actions. In particular this module recommends swapping the
cheap arm when it solves almost nothing, but it never switches anything.

Definitions (stated so the numbers cannot be misread):
- pass rate: true_success / n, graded against the hidden tests;
- claimed rate: verifier_claimed / n, the Verifier's own verdict;
- Verifier false-positive rate: P(claimed | the patch actually failed) = FP / (n - successes);
- Verifier false-negative rate: P(not claimed | the patch actually passed) = FN / successes.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from cascade.config import ModelsConfig
from cascade.routing.matrix import MatrixRow

MAX_ARM_LOW = 0.40
MAX_ARM_HIGH = 0.70
CHEAP_CEILING = 0.90
CHEAP_NEAR_ZERO = 0.10
SWAP_LABEL = "swap_in_cheap"


@dataclass(frozen=True)
class ArmCalibration:
    arm_id: str
    label: str | None
    n: int
    successes: int
    claimed: int
    false_pos: int  # claimed but failed
    false_neg: int  # passed but not claimed

    @property
    def rate(self) -> float:
        return self.successes / self.n if self.n else 0.0

    @property
    def claimed_rate(self) -> float:
        return self.claimed / self.n if self.n else 0.0

    @property
    def fp_rate(self) -> float | None:
        failed = self.n - self.successes
        return self.false_pos / failed if failed else None

    @property
    def fn_rate(self) -> float | None:
        return self.false_neg / self.successes if self.successes else None


@dataclass(frozen=True)
class CalibrationReport:
    arms: list[ArmCalibration]
    flags: list[str]
    tasks: int


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def calibrate(
    rows: Sequence[MatrixRow],
    models: ModelsConfig,
    *,
    task_ids: Collection[str],
    pipeline_config: str,
    seed: int = 0,
) -> CalibrationReport:
    arms: list[ArmCalibration] = []
    for arm in models.matrix_arms():
        mine = [
            r
            for r in rows
            if r.arm_id == arm.arm_id
            and r.task_id in task_ids
            and r.seed == seed
            and r.pipeline_config == pipeline_config
        ]
        arms.append(
            ArmCalibration(
                arm_id=arm.arm_id,
                label=arm.label,
                n=len(mine),
                successes=sum(r.true_success for r in mine),
                claimed=sum(r.verifier_claimed for r in mine),
                false_pos=sum(r.verifier_claimed and not r.true_success for r in mine),
                false_neg=sum(r.true_success and not r.verifier_claimed for r in mine),
            )
        )
    return CalibrationReport(arms, _flags(arms, models), len(set(task_ids)))


def _flags(arms: list[ArmCalibration], models: ModelsConfig) -> list[str]:
    flags: list[str] = []
    if len(arms) < 2:
        return flags
    cheap = next((a for a in arms if a.label == "cheap"), arms[0])
    top = next((a for a in arms if a.label == "max"), arms[-1])
    if top.n and not MAX_ARM_LOW <= top.rate <= MAX_ARM_HIGH:
        if top.rate > MAX_ARM_HIGH:
            where = "too easy: little room to show savings"
        else:
            where = "too hard: too few resolved tasks to compare policies"
        flags.append(
            f"MAX ARM OUTSIDE 40-70%: {top.arm_id} solved {top.successes}/{top.n} "
            f"({top.rate:.0%}); the task set may be {where}"
        )
    if cheap.n and cheap.rate >= CHEAP_CEILING:
        flags.append(
            f"CHEAP ARM AT CEILING: {cheap.arm_id} solved {cheap.successes}/{cheap.n} "
            f"({cheap.rate:.0%}); there is nothing for a router to escalate"
        )
    if cheap.n and cheap.rate <= CHEAP_NEAR_ZERO:
        has_swap = any(a.label == SWAP_LABEL for a in models.arms)
        advice = (
            f"consider enabling the optional '{SWAP_LABEL}' arm in models.yaml"
            if has_swap
            else f"models.yaml defines no '{SWAP_LABEL}' arm; you would need to add one"
        )
        flags.append(
            f"CHEAP ARM NEAR ZERO: {cheap.arm_id} solved {cheap.successes}/{cheap.n} "
            f"({cheap.rate:.0%}); {advice}. Nothing is switched automatically"
        )
    return flags


def format_calibration(r: CalibrationReport) -> str:
    lines = [f"CALIBRATION on {r.tasks} pilot task(s) (n is small: read the counts, not the %)"]
    lines.append(
        f"  {'arm':30s} {'passed':>8s} {'95% CI':>11s} {'claimed':>8s} {'FP':>9s} {'FN':>9s}"
    )
    for a in r.arms:
        lo, hi = wilson(a.successes, a.n)
        fp = "n/a" if a.fp_rate is None else f"{a.false_pos}/{a.n - a.successes}"
        fn = "n/a" if a.fn_rate is None else f"{a.false_neg}/{a.successes}"
        name = f"{a.label + ' = ' if a.label else ''}{a.arm_id}"
        lines.append(
            f"  {name:30s} {a.successes:>3d}/{a.n:<4d} {lo:4.0%}-{hi:<5.0%} "
            f"{a.claimed:>3d}/{a.n:<4d} {fp:>9s} {fn:>9s}"
        )
    lines.append("  FP = Verifier claimed success on a patch that failed (of all failures);")
    lines.append("  FN = Verifier rejected a patch that passed (of all passes).")
    if r.flags:
        lines.append("FLAGS (decisions for you, none acted on):")
        lines += [f"  - {f}" for f in r.flags]
    else:
        lines.append("no flags raised")
    return "\n".join(lines)
