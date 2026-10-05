"""Bootstrap weights, weighted statistics, intervals, and the margin rule (plus the plan's table)."""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from cascade.evals import bootstrap as bs
from cascade.evals.bootstrap import Arrays
from tests.conftest import ROOT


def arr(success: list[int], cost: list[float], latency: list[float]) -> Arrays:
    return Arrays(np.array(success, float), np.array(cost, float), np.array(latency, float))


# --- weights ------------------------------------------------------------------------------------


def test_task_weights_are_resample_counts_that_sum_to_n_and_are_seeded() -> None:
    w = bs.task_weights(7, b=500, seed=1)
    assert w.shape == (500, 7) and np.all(w.sum(axis=1) == 7) and np.all(w >= 0)
    assert np.array_equal(w, bs.task_weights(7, b=500, seed=1))
    assert not np.array_equal(w, bs.task_weights(7, b=500, seed=2))
    with pytest.raises(ValueError):
        bs.task_weights(0)


def test_defaults_are_the_plans_10000_resamples_and_seed_4242() -> None:
    assert bs.B_DEFAULT == 10_000 and bs.SEED_DEFAULT == 4242
    assert bs.task_weights(5).shape == (10_000, 5)


def test_cluster_weights_keep_every_task_of_a_repo_together() -> None:
    repos = ["a", "a", "a", "b", "b", "c"]
    w = bs.cluster_weights(repos, b=400, seed=3)
    assert w.shape == (400, 6)
    for row in w:
        assert row[0] == row[1] == row[2] and row[3] == row[4]  # whole repos, same multiplicity
    assert np.all((w[:, 0] + w[:, 3] + w[:, 5]) == 3)  # as many repo draws as repos
    assert np.array_equal(w, bs.cluster_weights(repos, b=400, seed=3))
    with pytest.raises(ValueError):
        bs.cluster_weights([])


# --- statistics against brute force -------------------------------------------------------------


def brute(a: Arrays, counts: np.ndarray) -> dict[str, float]:
    idx = np.repeat(np.arange(len(counts)), counts.astype(int))
    s, c, lat = a.success[idx], a.cost[idx], a.latency[idx]
    return {
        "rr": float(s.mean()), "cost": float(c.mean()),
        "cpr": float(c.sum() / s.sum()) if s.sum() else math.nan, "med": float(np.median(lat)),
    }  # fmt: skip


def test_weighted_statistics_equal_the_brute_force_resample_statistics() -> None:
    a = arr([1, 0, 1, 1, 0, 0, 1], [1, 2, 3, 4, 5, 6, 7], [10, 40, 20, 5, 30, 25, 15])
    w = bs.task_weights(7, b=60, seed=9)
    rr, cost = bs.resolve_rate(w, a), bs.cost_per_task(w, a)
    cpr, med = bs.cost_per_resolved(w, a), bs.median_latency(w, a)
    for i in range(60):
        want = brute(a, w[i])
        assert rr[i] == pytest.approx(want["rr"]) and cost[i] == pytest.approx(want["cost"])
        assert med[i] == pytest.approx(want["med"])
        if math.isnan(want["cpr"]):
            assert math.isnan(cpr[i])
        else:
            assert cpr[i] == pytest.approx(want["cpr"])


@pytest.mark.parametrize("n", [4, 5, 8, 9])
def test_the_weighted_median_equals_numpys_for_even_and_odd_counts(n: int) -> None:
    rng = np.random.default_rng(n)
    a = arr([1] * n, [1.0] * n, list(rng.permutation(n * 3)[:n].astype(float)))
    w = bs.task_weights(n, b=200, seed=n)
    got = bs.median_latency(w, a)
    for i in range(200):
        assert got[i] == pytest.approx(brute(a, w[i])["med"])


def test_point_estimates_use_unit_weights() -> None:
    a = arr([1, 0, 1, 1], [1, 2, 3, 4], [10, 20, 30, 40])
    assert bs.point_of(bs.resolve_rate, a) == 0.75
    assert bs.point_of(bs.cost_per_task, a) == 2.5
    assert bs.point_of(bs.total_cost, a) == 10.0
    assert bs.point_of(bs.cost_per_resolved, a) == pytest.approx(10 / 3)
    assert bs.point_of(bs.median_latency, a) == 25.0


def test_cost_per_resolved_is_undefined_when_nothing_is_resolved_and_the_count_is_reported() -> (
    None
):
    a = arr([0, 0, 0], [1, 2, 3], [1, 1, 1])
    assert math.isnan(bs.point_of(bs.cost_per_resolved, a))
    mixed = arr([1, 0, 0, 0, 0, 0, 0, 0], [1] * 8, [1] * 8)
    w = bs.task_weights(8, b=2000, seed=0)
    iv = bs.interval(bs.point_of(bs.cost_per_resolved, mixed), bs.cost_per_resolved(w, mixed))
    assert iv.n_undefined > 0 and not math.isnan(
        iv.lo
    )  # excluded and counted, not silently dropped
    allnan = bs.interval(math.nan, np.full(10, math.nan))
    assert math.isnan(allnan.lo) and allnan.n_undefined == 10


def test_cost_reduction_and_oracle_share_formulas() -> None:
    mx = arr([1, 1], [10.0, 10.0], [1, 1])
    pol_ = arr([1, 1], [4.0, 6.0], [1, 1])
    orc = arr([1, 1], [2.0, 2.0], [1, 1])
    ones = np.ones((1, 2))
    assert bs.cost_reduction(ones, pol_, mx)[0] == pytest.approx(0.5)  # 1 - 5/10
    # (C_max - C_policy) / (C_max - C_oracle) = (10-5)/(10-2)
    assert bs.oracle_savings_share(ones, pol_, mx, orc)[0] == pytest.approx(5 / 8)
    assert bs.oracle_savings_share(ones, orc, mx, orc)[0] == pytest.approx(1.0)
    assert bs.oracle_savings_share(ones, mx, mx, orc)[0] == pytest.approx(0.0)
    free = arr([1, 1], [0.0, 0.0], [1, 1])
    assert math.isnan(bs.cost_reduction(ones, pol_, free)[0])  # a zero-cost baseline has no ratio
    same = arr([1, 1], [3.0, 3.0], [1, 1])
    assert math.isnan(bs.oracle_savings_share(ones, pol_, same, same)[0])  # no gap to capture


def test_a_paired_difference_is_tighter_than_an_unpaired_one_for_correlated_policies() -> None:
    rng = np.random.default_rng(0)
    base = (rng.random(60) < 0.5).astype(float)
    other = base.copy()
    other[:3] = 1 - other[:3]  # nearly identical outcomes
    a, b = arr(list(base), [1] * 60, [1] * 60), arr(list(other), [1] * 60, [1] * 60)
    w = bs.task_weights(60, b=3000, seed=0)
    paired = bs.interval(0.0, bs.resolve_rate(w, a) - bs.resolve_rate(w, b))
    w2 = bs.task_weights(60, b=3000, seed=5)  # independent resamples for the second policy
    unpaired = bs.interval(0.0, bs.resolve_rate(w, a) - bs.resolve_rate(w2, b))
    assert (paired.hi - paired.lo) < 0.5 * (unpaired.hi - unpaired.lo)


def test_interval_is_the_95_percent_percentile_interval() -> None:
    samples = np.arange(1001, dtype=float)
    iv = bs.interval(500.0, samples)
    assert (iv.lo, iv.hi) == (pytest.approx(25.0), pytest.approx(975.0)) and iv.n_undefined == 0


def test_the_same_seed_gives_identical_intervals() -> None:
    a = arr([1, 0, 1, 0, 1, 1], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
    runs = [bs.interval(0.0, bs.resolve_rate(bs.task_weights(6, 1000, 11), a)) for _ in range(2)]
    assert runs[0] == runs[1]


# --- power and the margin rule ------------------------------------------------------------------


def test_power_matches_the_normal_approximation() -> None:
    se = math.sqrt(0.2 / 25)
    expected = 0.5 * (1 + math.erf((0.15 / se - 1.959964) / math.sqrt(2)))
    assert bs.noninferiority_power(25, 0.15) == pytest.approx(expected)
    assert bs.noninferiority_power(0, 0.15) == 0.0
    assert bs.noninferiority_power(40, 0.10) > bs.noninferiority_power(20, 0.10)  # more tasks
    assert bs.noninferiority_power(40, 0.15) > bs.noninferiority_power(40, 0.10)  # wider margin


@pytest.mark.parametrize(
    ("n", "margin"),
    [(5, 0.15), (13, 0.15), (25, 0.15), (60, 0.15), (99, 0.10), (100, 0.10), (400, 0.05)],
)
def test_margin_is_the_smallest_candidate_with_enough_power_capped_at_015(
    n: int, margin: float
) -> None:
    assert bs.margin_for(n) == margin
    assert bs.margin_for(n) <= bs.MARGIN_CAP


def test_the_plans_power_table_agrees_with_the_code() -> None:
    plan = (ROOT / "docs" / "EVAL_PLAN.md").read_text(encoding="utf-8")
    pattern = r"^\| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|$"
    rows = re.findall(pattern, plan, re.MULTILINE)
    assert len(rows) >= 7
    for n_s, *powers, rule in rows:
        n = int(n_s)
        for margin, shown in zip((0.05, 0.10, 0.15, 0.20), powers, strict=True):
            assert float(shown) == pytest.approx(bs.noninferiority_power(n, margin), abs=0.006)
        assert float(rule) == bs.margin_for(n), n


def test_the_plan_fixes_everything_the_brief_requires() -> None:
    plan = (ROOT / "docs" / "EVAL_PLAN.md").read_text(encoding="utf-8").lower()
    for needle in (
        "cost reduction versus always-max at equal-or-better resolve rate",
        "non-inferior",
        "(c_always_max - c_policy) / (c_always_max - c_oracle)",
        "leave-one-repo-out",
        "selection rule",
        "10,000 resamples",
        "seed 4242",
        "confidence intervals will be wide",
        "for the human to confirm",
        "independence assumption",
    ):
        assert needle in plan, needle
