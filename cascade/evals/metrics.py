"""Policy metrics with bootstrap CIs, the paired headline comparison, and the secondary analyses.

Definitions (fixed in docs/EVAL_PLAN.md):
- resolve rate: share of tasks whose policy outcome is a true success;
- cost per task: mean total cost the policy incurred per task (a cascade pays for every arm it
  tried); total cost: that times the number of tasks;
- cost per resolved issue: total cost / number resolved (undefined if nothing was resolved);
- median latency: median over tasks of the policy's wall time (a cascade's is the sum over the arms
  it tried, since they run in sequence).

Intervals are 95% percentile bootstrap intervals, 10,000 resamples, fixed seed. The PRIMARY scheme
resamples tasks; the repo-clustered scheme is a sensitivity check. A comparison between two policies
uses the SAME resamples for both, so it is paired.

HEADLINE: "cost reduction versus always-max at equal-or-better resolve rate", made precise as a
non-inferiority claim. Let delta be the pre-specified margin. With the task-bootstrap CI:
  non-inferior   <=> the lower bound of (resolve rate of policy - of always-max) > -delta
  saves          <=> the lower bound of cost reduction 1 - cost(policy)/cost(always-max) > 0
  verdict MET        if both hold;
  verdict INFERIOR   if the whole resolve-rate-difference interval is below -delta;
  verdict INCONCLUSIVE otherwise (a CI too wide to show either).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from cascade.evals import bootstrap as bs
from cascade.evals.bootstrap import Arrays, Interval, Weights
from cascade.routing.calibrate import wilson
from cascade.routing.matrix import MatrixRow
from cascade.routing.policies import ARMS, Outcome, TaskRecord

BASELINE = "always_max"
ORACLE = "oracle"


def to_arrays(outcomes: Sequence[Outcome]) -> Arrays:
    return Arrays(
        success=np.array([float(o.success) for o in outcomes]),
        cost=np.array([o.cost for o in outcomes]),
        latency=np.array([o.latency for o in outcomes]),
    )


@dataclass(frozen=True)
class PolicyMetrics:
    name: str
    n: int
    resolve_rate: Interval
    cost_per_task: Interval
    total_cost: Interval
    cost_per_resolved: Interval
    median_latency: Interval
    resolve_rate_clustered: Interval
    cost_per_task_clustered: Interval
    mean_arms_tried: float


def _iv(stat: bs.StatFn, a: Arrays, w: Weights) -> Interval:
    return bs.interval(bs.point_of(stat, a), stat(w, a))


def policy_metrics(
    name: str, outcomes: Sequence[Outcome], w_task: Weights, w_repo: Weights
) -> PolicyMetrics:
    a = to_arrays(outcomes)
    return PolicyMetrics(
        name=name,
        n=len(outcomes),
        resolve_rate=_iv(bs.resolve_rate, a, w_task),
        cost_per_task=_iv(bs.cost_per_task, a, w_task),
        total_cost=_iv(bs.total_cost, a, w_task),
        cost_per_resolved=_iv(bs.cost_per_resolved, a, w_task),
        median_latency=_iv(bs.median_latency, a, w_task),
        resolve_rate_clustered=_iv(bs.resolve_rate, a, w_repo),
        cost_per_task_clustered=_iv(bs.cost_per_task, a, w_repo),
        mean_arms_tried=float(np.mean([o.arms_tried for o in outcomes])),
    )


@dataclass(frozen=True)
class Comparison:
    policy: str
    baseline: str
    margin: float
    resolve_rate_diff: Interval  # policy - baseline
    cost_reduction: Interval  # 1 - cost(policy) / cost(baseline)
    cost_per_resolved_diff: Interval  # policy - baseline (undefined where either is)
    oracle_savings_share: Interval
    resolve_rate_diff_clustered: Interval
    cost_reduction_clustered: Interval
    non_inferior: bool
    saves: bool
    verdict: str
    verdict_clustered: str


def verdict_of(rr: Interval, saving: Interval, margin: float) -> tuple[bool, bool, str]:
    non_inferior = bool(rr.lo > -margin)
    saves = bool(saving.lo > 0)
    if non_inferior and saves:
        return non_inferior, saves, "MET"
    if rr.hi < -margin:
        return non_inferior, saves, "INFERIOR"
    return non_inferior, saves, "INCONCLUSIVE"


def _diff(stat: bs.StatFn, a: Arrays, b: Arrays, w: Weights) -> Interval:
    point = bs.point_of(stat, a) - bs.point_of(stat, b)
    return bs.interval(point, stat(w, a) - stat(w, b))


def _reduction(a: Arrays, base: Arrays, w: Weights) -> Interval:
    point = float(bs.cost_reduction(np.ones((1, a.cost.shape[0])), a, base)[0])
    return bs.interval(point, bs.cost_reduction(w, a, base))


def compare(
    name: str,
    outcomes: Sequence[Outcome],
    baseline: Sequence[Outcome],
    oracle: Sequence[Outcome],
    w_task: Weights,
    w_repo: Weights,
    margin: float,
) -> Comparison:
    a, base, orc = to_arrays(outcomes), to_arrays(baseline), to_arrays(oracle)
    rr = _diff(bs.resolve_rate, a, base, w_task)
    saving = _reduction(a, base, w_task)
    ones = np.ones((1, a.cost.shape[0]))
    share = bs.interval(
        float(bs.oracle_savings_share(ones, a, base, orc)[0]),
        bs.oracle_savings_share(w_task, a, base, orc),
    )
    rr_c = _diff(bs.resolve_rate, a, base, w_repo)
    saving_c = _reduction(a, base, w_repo)
    ni, sv, verdict = verdict_of(rr, saving, margin)
    return Comparison(
        policy=name, baseline=BASELINE, margin=margin, resolve_rate_diff=rr, cost_reduction=saving,
        cost_per_resolved_diff=_diff(bs.cost_per_resolved, a, base, w_task),
        oracle_savings_share=share, resolve_rate_diff_clustered=rr_c,
        cost_reduction_clustered=saving_c, non_inferior=ni, saves=sv, verdict=verdict,
        verdict_clustered=verdict_of(rr_c, saving_c, margin)[2],
    )  # fmt: skip


# --- per-repo breakdown --------------------------------------------------------------------------


@dataclass(frozen=True)
class RepoRow:
    repo: str
    policy: str
    n: int
    resolve_rate: float
    cost_per_task: float


def per_repo(
    policies: Mapping[str, Sequence[Outcome]], records: Sequence[TaskRecord]
) -> list[RepoRow]:
    """Point estimates only: a repo has too few tasks for an interval to mean much."""
    rows: list[RepoRow] = []
    for repo in sorted({r.repo for r in records}):
        idx = [i for i, r in enumerate(records) if r.repo == repo]
        for name, outs in policies.items():
            sel = [outs[i] for i in idx]
            rows.append(
                RepoRow(
                    repo, name, len(idx),
                    float(np.mean([o.success for o in sel])), float(np.mean([o.cost for o in sel])),
                )
            )  # fmt: skip
    return rows


# --- Verifier false positives / negatives per arm ------------------------------------------------


@dataclass(frozen=True)
class VerifierRow:
    arm: str
    n: int
    passed: int
    claimed: int
    false_pos: int  # claimed success, patch failed
    false_neg: int  # rejected, patch passed
    fp_rate: Interval | None  # P(claimed | failed), Wilson 95%
    fn_rate: Interval | None  # P(not claimed | passed), Wilson 95%


def _wilson_iv(k: int, n: int) -> Interval | None:
    if n == 0:
        return None
    lo, hi = wilson(k, n)
    return Interval(k / n, lo, hi, 0)


def verifier_errors(records: Sequence[TaskRecord]) -> list[VerifierRow]:
    out: list[VerifierRow] = []
    for a, arm in enumerate(ARMS):
        cells = [r.cells[a] for r in records]
        passed = sum(c.success for c in cells)
        failed = len(cells) - passed
        fp = sum(c.claimed and not c.success for c in cells)
        fn = sum(c.success and not c.claimed for c in cells)
        out.append(
            VerifierRow(
                arm, len(cells), passed, sum(c.claimed for c in cells), fp, fn,
                _wilson_iv(fp, failed), _wilson_iv(fn, passed),
            )
        )  # fmt: skip
    return out


# --- ablations (mid arm, paired on the tasks that have every variant) ---------------------------


@dataclass(frozen=True)
class AblationRow:
    variant: str
    n_shared: int
    resolve_rate: float
    cost_per_task: float
    resolve_rate_diff: Interval  # variant - default, paired over the shared tasks
    cost_per_task_diff: Interval


def ablation_analysis(
    default: Mapping[str, MatrixRow],
    variants: Mapping[str, Mapping[str, MatrixRow]],
    *,
    b: int = bs.B_DEFAULT,
    seed: int = bs.SEED_DEFAULT,
) -> list[AblationRow]:
    """Default vs each ablation, on the tasks present in the default AND every variant."""
    shared = sorted(set(default).intersection(*[set(v) for v in variants.values()]))
    if not shared:
        return []
    w = bs.task_weights(len(shared), b, seed)

    def arr(rows: Mapping[str, MatrixRow]) -> Arrays:
        sel = [rows[t] for t in shared]
        return Arrays(
            np.array([float(r.true_success) for r in sel]),
            np.array([r.cost_usd for r in sel]),
            np.array([r.wall_s for r in sel]),
        )

    base = arr(default)
    zero = Interval(0.0, 0.0, 0.0, 0)
    out = [
        AblationRow(
            "default", len(shared), bs.point_of(bs.resolve_rate, base),
            bs.point_of(bs.cost_per_task, base), zero, zero,
        )
    ]  # fmt: skip
    for name, rows in variants.items():
        a = arr(rows)
        out.append(
            AblationRow(
                name, len(shared), bs.point_of(bs.resolve_rate, a),
                bs.point_of(bs.cost_per_task, a),
                _diff(bs.resolve_rate, a, base, w), _diff(bs.cost_per_task, a, base, w),
            )
        )  # fmt: skip
    return out


# --- noise from repeat-subset rows ---------------------------------------------------------------


@dataclass(frozen=True)
class NoiseRow:
    arm: str
    n_pairs: int
    flips: int
    flip_rate: Interval  # cell outcome (true_success) differs between seed 0 and the repeat seed
    claimed_flip_rate: float


def noise_estimate(
    first: Mapping[tuple[str, str], MatrixRow],
    repeat: Mapping[tuple[str, str], MatrixRow],
    arms: Sequence[str],
    *,
    b: int = bs.B_DEFAULT,
    seed: int = bs.SEED_DEFAULT,
) -> list[NoiseRow]:
    """Keys are (task_id, arm_id). Flip rate = share of repeated cells whose outcome changed."""
    out: list[NoiseRow] = []
    for arm in arms:
        keys = sorted(k for k in repeat if k[1] == arm and k in first)
        if not keys:
            continue
        flips = np.array([float(first[k].true_success != repeat[k].true_success) for k in keys])
        cflips = np.mean([first[k].verifier_claimed != repeat[k].verifier_claimed for k in keys])
        w = bs.task_weights(len(keys), b, seed)
        a = Arrays(flips, flips, flips)
        out.append(
            NoiseRow(
                arm, len(keys), int(flips.sum()),
                bs.interval(float(flips.mean()), bs.resolve_rate(w, a)), float(cflips),
            )
        )  # fmt: skip
    return out
