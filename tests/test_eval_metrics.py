"""Policy metrics, the headline verdict, and the secondary analyses. Synthetic outcomes."""

from __future__ import annotations

import math

import pytest

from cascade.evals import bootstrap as bs
from cascade.evals import metrics as mt
from cascade.evals.bootstrap import Interval
from cascade.routing.matrix import MatrixRow
from cascade.routing.policies import Cell, Outcome, TaskRecord
from tests.matrix_helpers import row
from tests.phase3_helpers import make_task

B = 2000


def outs(success: list[bool], cost: list[float], lat: float = 10.0) -> list[Outcome]:
    return [Outcome(c, s, lat, 1, 0) for s, c in zip(success, cost, strict=True)]


def weights(n: int, repos: list[str] | None = None) -> tuple[bs.Weights, bs.Weights]:
    groups = repos or [f"r{i % 3}" for i in range(n)]
    return bs.task_weights(n, B, 1), bs.cluster_weights(groups, B, 1)


# --- policy metrics -----------------------------------------------------------------------------


def test_policy_metrics_points_and_intervals_bracket_the_observed_values() -> None:
    o = outs([True, True, False, True] * 5, [1.0, 2.0, 3.0, 4.0] * 5)
    wt, wr = weights(20)
    m = mt.policy_metrics("p", o, wt, wr)
    assert m.n == 20 and m.resolve_rate.point == 0.75 and m.cost_per_task.point == 2.5
    assert m.total_cost.point == pytest.approx(50.0) and m.mean_arms_tried == 1.0
    assert m.cost_per_resolved.point == pytest.approx(50 / 15)
    assert m.median_latency.point == 10.0
    for iv in (m.resolve_rate, m.cost_per_task, m.total_cost, m.cost_per_resolved):
        assert iv.lo <= iv.point <= iv.hi
    assert m.resolve_rate_clustered.lo <= 0.75 <= m.resolve_rate_clustered.hi


def test_total_cost_interval_is_n_times_the_cost_per_task_interval() -> None:
    o = outs([True] * 10, [float(i) for i in range(10)])
    wt, wr = weights(10)
    m = mt.policy_metrics("p", o, wt, wr)
    assert m.total_cost.lo == pytest.approx(10 * m.cost_per_task.lo)
    assert m.total_cost.hi == pytest.approx(10 * m.cost_per_task.hi)


# --- the headline verdict -----------------------------------------------------------------------


def iv(point: float, lo: float, hi: float) -> Interval:
    return Interval(point, lo, hi, 0)


@pytest.mark.parametrize(
    ("rr", "saving", "ni", "sv", "verdict"),
    [
        (iv(0.0, -0.05, 0.05), iv(0.5, 0.2, 0.8), True, True, "MET"),
        (iv(0.0, -0.15, 0.05), iv(0.5, 0.2, 0.8), False, True, "INCONCLUSIVE"),  # lo == -margin
        (iv(0.0, -0.10, 0.05), iv(0.5, -0.1, 0.8), True, False, "INCONCLUSIVE"),  # saving not shown
        (iv(0.0, 0.0, 0.0), iv(0.0, 0.0, 0.0), True, False, "INCONCLUSIVE"),  # identical policy
        (iv(-0.4, -0.6, -0.2), iv(0.5, 0.2, 0.8), False, True, "INFERIOR"),  # all below -margin
        (iv(-0.3, -0.5, -0.1), iv(0.5, 0.2, 0.8), False, True, "INCONCLUSIVE"),  # hi above -margin
    ],
)
def test_verdict_rule(rr: Interval, saving: Interval, ni: bool, sv: bool, verdict: str) -> None:
    assert mt.verdict_of(rr, saving, 0.15) == (ni, sv, verdict)


def test_a_much_cheaper_policy_with_equal_outcomes_meets_the_headline() -> None:
    n = 30
    base = outs([True] * 20 + [False] * 10, [10.0] * n)
    cheap = outs([True] * 20 + [False] * 10, [1.0] * n)
    orc = outs([True] * 20 + [False] * 10, [0.5] * n)
    wt, wr = weights(n)
    c = mt.compare("cheap_same", cheap, base, orc, wt, wr, 0.15)
    assert c.verdict == "MET" and c.non_inferior and c.saves
    assert c.resolve_rate_diff.point == 0.0 and c.cost_reduction.point == pytest.approx(0.9)
    assert c.oracle_savings_share.point == pytest.approx(9 / 9.5)
    assert c.verdict_clustered == "MET" and c.baseline == mt.BASELINE and c.margin == 0.15


def test_a_policy_that_fails_where_the_baseline_succeeds_is_inferior() -> None:
    n = 30
    base = outs([True] * n, [10.0] * n)
    bad = outs([False] * n, [1.0] * n)
    wt, wr = weights(n)
    c = mt.compare("bad", bad, base, base, wt, wr, 0.15)
    assert c.verdict == "INFERIOR" and c.resolve_rate_diff.point == -1.0


def test_a_noisy_small_comparison_is_inconclusive_not_a_win() -> None:
    n = 12
    base = outs([True, False] * 6, [10.0] * n)
    other = outs([True, True, False, False, True, False] * 2, [4.0] * n)
    wt, wr = weights(n)
    c = mt.compare("noisy", other, base, base, wt, wr, 0.05)
    assert c.verdict == "INCONCLUSIVE"


def test_a_cascade_style_policy_pays_for_every_arm_it_tried() -> None:
    n = 20
    base = outs([True] * n, [10.0] * n)
    casc = [Outcome(3.0, True, 30.0, 2, 1) for _ in range(n)]
    wt, wr = weights(n)
    m = mt.policy_metrics("cascade", casc, wt, wr)
    assert m.mean_arms_tried == 2.0 and m.cost_per_task.point == 3.0
    assert m.median_latency.point == 30.0
    c = mt.compare("cascade", casc, base, base, wt, wr, 0.15)
    assert c.cost_reduction.point == pytest.approx(0.7)


def test_cost_per_resolved_difference_is_undefined_when_a_policy_resolves_nothing() -> None:
    n = 10
    base = outs([True] * n, [10.0] * n)
    none = outs([False] * n, [1.0] * n)
    wt, wr = weights(n)
    c = mt.compare("none", none, base, base, wt, wr, 0.15)
    assert math.isnan(c.cost_per_resolved_diff.point) and math.isnan(c.cost_per_resolved_diff.lo)


# --- per repo and Verifier errors ---------------------------------------------------------------


def rec(repo: str, i: int, cells: list[tuple[bool, bool]]) -> TaskRecord:
    c = [Cell(s, cl, 1.0 + k, 10.0) for k, (s, cl) in enumerate(cells)]
    return TaskRecord(make_task(repo, i), (c[0], c[1], c[2]))


def test_per_repo_counts_and_point_estimates() -> None:
    recs = [
        rec("o/a", 1, [(True, True)] * 3),
        rec("o/a", 2, [(False, False)] * 3),
        rec("o/b", 3, [(True, True)] * 3),
    ]
    pols = {"always_cheap": outs([True, False, True], [1.0, 1.0, 1.0])}
    by = {(r.repo, r.policy): r for r in mt.per_repo(pols, recs)}
    assert by[("o/a", "always_cheap")].n == 2 and by[("o/a", "always_cheap")].resolve_rate == 0.5
    assert by[("o/b", "always_cheap")].n == 1 and by[("o/b", "always_cheap")].resolve_rate == 1.0


def test_verifier_error_counts_and_rates_per_arm() -> None:
    recs = [
        rec("o/a", 1, [(True, True), (True, False), (False, True)]),  # cheap ok; mid FN; max FP
        rec("o/a", 2, [(False, True), (False, False), (True, True)]),  # cheap FP
        rec("o/a", 3, [(False, False), (True, True), (True, True)]),
        rec("o/a", 4, [(True, True), (True, True), (False, False)]),
    ]
    cheap, mid, mx = mt.verifier_errors(recs)
    assert (cheap.arm, cheap.n, cheap.passed, cheap.claimed) == ("cheap", 4, 2, 3)
    assert (cheap.false_pos, cheap.false_neg) == (1, 0)
    assert cheap.fp_rate is not None and cheap.fp_rate.point == pytest.approx(1 / 2)  # of failures
    assert cheap.fn_rate is not None and cheap.fn_rate.point == 0.0
    assert (mid.false_neg, mid.false_pos) == (1, 0) and mid.fn_rate is not None
    assert mid.fn_rate.point == pytest.approx(1 / 3)  # of 3 passes
    assert (mx.false_pos, mx.false_neg) == (1, 0)
    assert 0.0 <= cheap.fp_rate.lo <= cheap.fp_rate.point <= cheap.fp_rate.hi <= 1.0


def test_verifier_rates_are_absent_when_the_denominator_is_empty() -> None:
    recs = [rec("o/a", 1, [(True, True)] * 3), rec("o/a", 2, [(True, True)] * 3)]
    cheap = mt.verifier_errors(recs)[0]
    assert cheap.fp_rate is None  # no failures at all
    assert cheap.fn_rate is not None


# --- ablations and noise ------------------------------------------------------------------------


def mrow(task: str, success: bool, cost: float = 1.0, claimed: bool = False) -> MatrixRow:
    return row(task, "m@high", true_success=success, cost_usd=cost, verifier_claimed=claimed)


def test_ablation_pairs_only_tasks_present_in_the_default_and_every_variant() -> None:
    tasks = [f"t{i}" for i in range(10)]
    default = {t: mrow(t, i < 6, 1.0) for i, t in enumerate(tasks)}
    no_repro = {t: mrow(t, i < 3, 0.8) for i, t in enumerate(tasks[:9])}  # t9 missing
    no_review = {t: mrow(t, int(t[1:]) < 6, 0.9) for t in tasks[1:]}  # t0 missing
    variants = {"no_reproducer": no_repro, "no_reviewer": no_review}
    rows = mt.ablation_analysis(default, variants, b=B)
    assert [r.variant for r in rows] == ["default", "no_reproducer", "no_reviewer"]
    assert all(r.n_shared == 8 for r in rows)  # t1..t8
    d, nr, nv = rows
    assert d.resolve_rate == pytest.approx(5 / 8)  # t1..t5 pass in the default
    assert nr.resolve_rate == pytest.approx(2 / 8)
    assert nr.resolve_rate_diff.point == pytest.approx(-3 / 8)
    assert nr.cost_per_task_diff.point == pytest.approx(-0.2)
    assert nv.resolve_rate_diff.point == pytest.approx(0.0)
    assert nr.resolve_rate_diff.lo <= nr.resolve_rate_diff.point <= nr.resolve_rate_diff.hi


def test_ablation_with_no_shared_tasks_is_empty() -> None:
    assert mt.ablation_analysis({"a": mrow("a", True)}, {"v": {"b": mrow("b", True)}}) == []


def test_noise_flip_rate_is_the_share_of_repeated_cells_whose_outcome_changed() -> None:
    first = {(f"t{i}", "c"): row(f"t{i}", "c", true_success=i % 2 == 0) for i in range(10)}
    repeat = {
        (f"t{i}", "c"): row(f"t{i}", "c", seed=1, true_success=(i % 2 == 0) != (i < 3))
        for i in range(10)
    }  # the first three cells flip
    [n] = mt.noise_estimate(first, repeat, ["c", "unseen"], b=B)
    assert n.arm == "c" and n.n_pairs == 10 and n.flips == 3
    assert n.flip_rate.point == pytest.approx(0.3) and n.flip_rate.lo <= 0.3 <= n.flip_rate.hi


def test_noise_ignores_repeat_cells_without_a_first_cell_and_arms_with_no_pairs() -> None:
    first = {("t0", "c"): row("t0", "c")}
    repeat = {("t0", "c"): row("t0", "c", seed=1), ("t9", "c"): row("t9", "c", seed=1)}
    [n] = mt.noise_estimate(first, repeat, ["c"], b=B)
    assert n.n_pairs == 1
    assert mt.noise_estimate(first, {}, ["c"], b=B) == []
