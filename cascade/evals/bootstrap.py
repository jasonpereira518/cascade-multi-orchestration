"""Bootstrap machinery: task and repo-clustered resampling, weighted statistics, and intervals.

Every bootstrap here is expressed as a WEIGHT MATRIX ``W`` of shape (resamples, tasks):
``W[b, i]`` is how many times task i appears in resample b. A statistic is then a function of
``W`` and the per-task arrays, so two policies evaluated with the same ``W`` are evaluated on
identical resamples. That is what makes comparisons PAIRED: the difference between two policies is
computed inside each resample, so the interval reflects how the policies differ on the same tasks,
not how each varies alone.

- task bootstrap (primary): resample tasks with replacement.
- repo-clustered bootstrap (sensitivity): resample whole repos with replacement and carry all of a
  repo's tasks, which respects that tasks within a repo are not independent.

The fixed seed and 10,000 resamples are part of docs/EVAL_PLAN.md. Intervals are 95% percentile
intervals. A statistic can be undefined in a resample (cost per resolved issue when nothing in the
resample was resolved); those resamples are excluded and COUNTED, never silently dropped.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

B_DEFAULT = 10_000
SEED_DEFAULT = 4242
ALPHA = 0.05

Vec = NDArray[np.float64]
Weights = NDArray[np.float64]


def task_weights(n: int, b: int = B_DEFAULT, seed: int = SEED_DEFAULT) -> Weights:
    """(b, n) counts of each task in b resamples of n tasks drawn with replacement."""
    if n < 1:
        raise ValueError("need at least one task to bootstrap")
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=b).astype(np.float64)


def cluster_weights(
    clusters: Sequence[str], b: int = B_DEFAULT, seed: int = SEED_DEFAULT
) -> Weights:
    """(b, n) weights from resampling whole clusters (repos) with replacement."""
    names = sorted(set(clusters))
    k = len(names)
    if k < 1:
        raise ValueError("need at least one cluster to bootstrap")
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(k, np.full(k, 1.0 / k), size=b).astype(np.float64)
    col = np.array([names.index(c) for c in clusters])
    return counts[:, col]


@dataclass(frozen=True)
class Arrays:
    """One policy's per-task results, aligned with the task order of the weight matrix."""

    success: Vec  # 0.0 / 1.0
    cost: Vec
    latency: Vec


StatFn = Callable[[Weights, Arrays], Vec]


def resolve_rate(w: Weights, a: Arrays) -> Vec:
    return (w @ a.success) / w.sum(axis=1)


def cost_per_task(w: Weights, a: Arrays) -> Vec:
    return (w @ a.cost) / w.sum(axis=1)


def total_cost(w: Weights, a: Arrays) -> Vec:
    """Mean cost per task times n, where n is the real (observed) task count."""
    return cost_per_task(w, a) * a.cost.shape[0]


def cost_per_resolved(w: Weights, a: Arrays) -> Vec:
    num, den = w @ a.cost, w @ a.success
    out = np.full(num.shape, np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def median_latency(w: Weights, a: Arrays) -> Vec:
    """Weighted median per resample, equal to ``np.median`` of the resampled values.

    The lower median is the first value whose cumulative weight reaches half the total; the upper
    median is the first whose cumulative weight exceeds half. They coincide for an odd total and
    are the two middle values for an even one, whose mean is the conventional median.
    """
    order = np.argsort(a.latency, kind="stable")
    values = a.latency[order]
    cum = np.cumsum(w[:, order], axis=1)
    half = cum[:, -1:] / 2.0
    lower = values[(cum >= half).argmax(axis=1)]
    upper = values[(cum > half).argmax(axis=1)]
    return (lower + upper) / 2.0


def cost_reduction(w: Weights, a: Arrays, baseline: Arrays) -> Vec:
    """1 - cost(policy) / cost(baseline), per resample (NaN where the baseline cost is zero)."""
    base = cost_per_task(w, baseline)
    out = np.full(base.shape, np.nan)
    ok = base > 0
    out[ok] = 1.0 - cost_per_task(w, a)[ok] / base[ok]
    return out


def oracle_savings_share(w: Weights, a: Arrays, baseline: Arrays, oracle: Arrays) -> Vec:
    """(C_baseline - C_policy) / (C_baseline - C_oracle), per resample; NaN if the gap is ~0."""
    cb, cp, co = cost_per_task(w, baseline), cost_per_task(w, a), cost_per_task(w, oracle)
    gap = cb - co
    out = np.full(gap.shape, np.nan)
    ok = gap > 1e-12
    out[ok] = (cb[ok] - cp[ok]) / gap[ok]
    return out


def point_of(stat: StatFn, a: Arrays) -> float:
    """The statistic on the observed tasks (every weight 1)."""
    return float(stat(np.ones((1, a.cost.shape[0])), a)[0])


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    n_undefined: int  # resamples where the statistic was undefined (excluded from lo/hi)


def interval(point: float, samples: Vec, alpha: float = ALPHA) -> Interval:
    finite = samples[np.isfinite(samples)]
    undefined = int(samples.shape[0] - finite.shape[0])
    if finite.shape[0] == 0:
        return Interval(point, math.nan, math.nan, undefined)
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(point, float(lo), float(hi), undefined)


# --- non-inferiority power and the margin rule (docs/EVAL_PLAN.md) ------------------------------

MARGIN_CANDIDATES = (0.05, 0.10, 0.15)
MARGIN_CAP = 0.15
ASSUMED_DISCORDANCE = 0.20  # share of tasks on which the policy and always-max outcomes differ
POWER_TARGET = 0.50
_Z975 = 1.959964


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def noninferiority_power(n: int, margin: float, discordance: float = ASSUMED_DISCORDANCE) -> float:
    """P(95% CI lower bound of the paired resolve-rate difference > -margin) if the true difference
    is zero, by the normal approximation: SE = sqrt(discordance / n); power = Phi(margin/SE - 1.96).

    It is a planning calculation under an ASSUMED discordance, not a statement about the data.
    """
    if n < 1:
        return 0.0
    se = math.sqrt(discordance / n)
    return _phi(margin / se - _Z975)


def margin_for(n_tasks: int) -> float:
    """The smallest candidate margin with power >= POWER_TARGET; capped at MARGIN_CAP.

    When even the cap has too little power the headline is expected to be inconclusive, and that is
    reported as such; the margin is never loosened beyond the cap to manufacture a pass.
    """
    for m in MARGIN_CANDIDATES:
        if noninferiority_power(n_tasks, m) >= POWER_TARGET:
            return m
    return MARGIN_CAP
