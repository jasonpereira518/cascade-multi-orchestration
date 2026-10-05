"""Sizing from a pilot ledger and calibration from pilot rows. Synthetic data only.

Every number the sizing report prints must be recomputable from the ledger, so these tests build a
ledger whose per-attempt costs are known and check the report against independent arithmetic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cascade.agents.pipeline import PipelineConfig
from cascade.config import ModelsConfig, load_models
from cascade.harness.ledger import Ledger
from cascade.routing import calibrate as cal
from cascade.routing import sizing as sz
from cascade.routing.matrix import MatrixRow
from tests.matrix_helpers import (
    CHEAP,
    MATRIX_MODELS,
    MAX,
    MID,
    matrix_budget,
    matrix_models,
    new_ledger,
    row,
    spend,
)

# --- percentiles --------------------------------------------------------------------------------


def test_percentile_is_linear_interpolation() -> None:
    assert sz.percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert sz.percentile(list(range(1, 11)), 0.9) == pytest.approx(9.1)
    assert sz.percentile([7.0], 0.9) == 7.0
    assert sz.percentile([4, 1, 3, 2], 0.0) == 1 and sz.percentile([4, 1, 3, 2], 1.0) == 4
    assert sz.median([1, 2, 3, 4]) == 2.5
    with pytest.raises(ValueError, match="no values"):
        sz.percentile([], 0.5)


# --- reading the pilot ledger -------------------------------------------------------------------


def pilot_ledger(tmp_path: Path, n: int = 10) -> Ledger:
    """n pilot attempts per arm; attempt i costs (i+1) cents on cheap/mid, 2x that on max."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    ledger = new_ledger(tmp_path)
    for i in range(n):
        for arm in (CHEAP, MID):  # luna: $1/M, so 10k input tokens = $0.01
            spend(
                ledger, arm, 10_000 * (i + 1), task_id=f"t{i}", attempt_id=arm,
                call_id=f"{arm}-{i}", bucket="pilot",
            )  # fmt: skip
        spend(  # sol: $10/M, so 2k input tokens = $0.02
            ledger, MAX, 2_000 * (i + 1), task_id=f"t{i}", attempt_id=MAX,
            call_id=f"{MAX}-{i}", bucket="pilot",
        )  # fmt: skip
    return ledger


def test_attempt_costs_sum_calls_per_attempt_and_only_in_the_bucket(tmp_path: Path) -> None:
    ledger = new_ledger(tmp_path)
    spend(ledger, CHEAP, 10_000, attempt_id="a1", call_id="c1", bucket="pilot")
    spend(ledger, CHEAP, 20_000, attempt_id="a1", call_id="c2", bucket="pilot")  # same attempt
    spend(ledger, CHEAP, 10_000, attempt_id="a1|try1", call_id="c3", bucket="pilot")  # a retry
    spend(ledger, CHEAP, 50_000, attempt_id="b1", call_id="c4", bucket="matrix")  # other bucket
    costs = sz.attempt_costs(tmp_path / "ledger.sqlite", "pilot")
    assert sorted(costs[CHEAP]) == [pytest.approx(0.01), pytest.approx(0.03)]
    assert sz.bucket_task_ids(tmp_path / "ledger.sqlite", "pilot") == {"t1"}
    ledger.close()


def test_arm_cost_stats_match_independent_arithmetic(tmp_path: Path) -> None:
    ledger = pilot_ledger(tmp_path)
    costs = sz.attempt_costs(tmp_path / "ledger.sqlite", "pilot")
    stats = sz.arm_cost_stats(costs, matrix_models())
    assert [a.arm_id for a in stats] == [CHEAP, MID, MAX]
    assert [a.label for a in stats] == ["cheap", "mid", "max"]
    cheap = stats[0]
    assert cheap.attempts == 10 and cheap.mean == pytest.approx(0.055)
    assert cheap.median == pytest.approx(0.055) and cheap.p90 == pytest.approx(0.091)
    assert stats[2].mean == pytest.approx(0.11) and stats[2].p90 == pytest.approx(0.182)
    ledger.close()


def test_an_arm_with_no_pilot_attempts_is_an_error() -> None:
    with pytest.raises(ValueError, match="no pilot attempts logged"):
        sz.arm_cost_stats({CHEAP: [0.1]}, matrix_models())


# --- the report ---------------------------------------------------------------------------------


def report(
    tmp_path: Path,
    *,
    matrix_cap: float = 20.0,
    audited: int = 60,
    pilot: int = 10,
    models: ModelsConfig | None = None,
) -> sz.SizingReport:
    ledger = pilot_ledger(tmp_path)
    m = models or matrix_models()
    stats = sz.arm_cost_stats(sz.attempt_costs(tmp_path / "ledger.sqlite", "pilot"), m)
    budget = matrix_budget(tmp_path, buckets={"pilot": 5.0, "matrix": matrix_cap})
    r = sz.build_report(
        stats, m, budget, matrix_spent_usd=ledger.bucket_spent_usd("matrix"),
        global_spent_usd=ledger.spent_usd(), audited_tasks=audited, pilot_tasks=pilot,
    )  # fmt: skip
    ledger.close()
    return r


def test_cost_per_task_is_the_sum_of_arm_costs_and_affordable_n_is_a_floor(tmp_path: Path) -> None:
    r = report(tmp_path)
    full = r.options[0]
    assert full.per_task_mean == pytest.approx(0.055 + 0.055 + 0.11)
    assert full.per_task_p90 == pytest.approx(0.091 + 0.091 + 0.182)
    assert full.new_tasks_mean == int(20.0 // 0.22) and full.new_tasks_p90 == int(20.0 // 0.364)
    assert full.covered_mean == full.new_tasks_mean + 10  # pilot tasks are already covered
    assert r.budget_usd == pytest.approx(20.0) and "bucket 'matrix'" in r.budget_basis


def test_the_budget_basis_falls_back_to_the_global_cap_without_a_matrix_bucket(
    tmp_path: Path,
) -> None:
    ledger = pilot_ledger(tmp_path)
    costs = sz.attempt_costs(tmp_path / "ledger.sqlite", "pilot")
    stats = sz.arm_cost_stats(costs, matrix_models())
    budget = matrix_budget(tmp_path, buckets={})
    r = sz.build_report(
        stats, matrix_models(), budget, matrix_spent_usd=0.0,
        global_spent_usd=ledger.spent_usd(), audited_tasks=60, pilot_tasks=10,
    )  # fmt: skip
    assert "global cap" in r.budget_basis
    assert r.budget_usd == pytest.approx(50.0 - ledger.spent_usd())
    ledger.close()


def test_flex_halves_the_cost_and_gets_no_pilot_credit(tmp_path: Path) -> None:
    r = report(tmp_path)
    full, flex = r.options[0], r.options[1]
    assert flex.available and flex.per_task_mean == pytest.approx(full.per_task_mean * 0.5)
    assert flex.new_tasks_mean == int(20.0 // (0.22 * 0.5))
    assert flex.covered_mean == flex.new_tasks_mean  # the standard-tier pilot cannot be reused


def test_flex_is_unavailable_when_a_models_flex_multiplier_is_null(tmp_path: Path) -> None:
    edited = tmp_path / "models.yaml"
    text = MATRIX_MODELS.read_text().replace("flex_multiplier: 0.5", "flex_multiplier: null", 1)
    edited.write_text(text)
    r = report(tmp_path / "x", models=load_models(edited))
    flex = r.options[1]
    assert not flex.available and "no flex_multiplier" in flex.note
    assert flex.covered_p90 == 0
    assert "no flex_multiplier" in sz.format_report(r)


def test_dropping_the_mid_arm_removes_exactly_its_cost(tmp_path: Path) -> None:
    r = report(tmp_path)
    drop = r.options[2]
    assert drop.name.startswith("drop the mid arm") and MID in drop.name
    assert drop.arms == [CHEAP, MAX]
    assert drop.per_task_mean == pytest.approx(0.055 + 0.11)


def test_the_subset_option_is_capped_at_the_audited_task_count(tmp_path: Path) -> None:
    r = report(tmp_path, audited=25)
    subset = r.options[3]
    assert subset.name.startswith("subset of tasks") and subset.covered_mean == 25
    assert r.options[0].covered_mean > 25  # the uncapped option can exceed what exists


def test_under_40_tasks_at_p90_requires_a_decision_and_picks_nothing(tmp_path: Path) -> None:
    r = report(tmp_path, matrix_cap=5.0)  # 5.0 // 0.364 = 13 new tasks + 10 pilot = 23 < 40
    assert r.options[0].covered_p90 == 23 and r.decision_required
    text = sz.format_report(r)
    assert "DECISION REQUIRED" in text and "No option has been chosen" in text
    assert "recommend" not in text.lower() and "choose option" not in text.lower()
    for header in ("full 3-arm matrix", "flex", "drop the mid arm", "subset of tasks"):
        assert header in text


@pytest.mark.parametrize(("pilot", "covered", "required"), [(10, 40, False), (9, 39, True)])
def test_the_decision_boundary_is_exactly_40_tasks_at_the_p90(
    tmp_path: Path, pilot: int, covered: int, required: bool
) -> None:
    r = report(tmp_path, matrix_cap=11.0, pilot=pilot)  # 11.0 // 0.364 = 30 new tasks
    assert r.options[0].new_tasks_p90 == 30 and r.options[0].covered_p90 == covered
    assert r.decision_required is required


def test_at_or_above_40_no_decision_is_forced_but_none_is_picked_either(tmp_path: Path) -> None:
    r = report(tmp_path, matrix_cap=20.0)  # 20 // 0.364 = 54 new + 10 pilot
    assert not r.decision_required
    text = sz.format_report(r)
    assert "DECISION REQUIRED" not in text and "No option is chosen automatically" in text


def test_the_report_states_its_method_limits(tmp_path: Path) -> None:
    text = sz.format_report(report(tmp_path))
    assert "p90 over ~10 attempts" in text and "audited-ok tasks: 60" in text
    assert "mean" in text and "median" in text and "p90" in text


# --- calibration --------------------------------------------------------------------------------

CFG = PipelineConfig().canonical()
PILOT = {f"t{i}" for i in range(10)}


def arm_rows(arm: str, passed: int, *, claimed: int | None = None, **over: Any) -> list[MatrixRow]:
    out: list[MatrixRow] = []
    for i in range(10):
        ok = i < passed
        said = ok if claimed is None else i < claimed
        out.append(row(f"t{i}", arm, true_success=ok, verifier_claimed=said, **over))
    return out


def calibrate(rows: list[MatrixRow], models: ModelsConfig | None = None) -> cal.CalibrationReport:
    return cal.calibrate(rows, models or matrix_models(), task_ids=PILOT, pipeline_config=CFG)


def test_counts_rates_and_verifier_errors_per_arm() -> None:
    rows = [
        *arm_rows(CHEAP, 3, claimed=5),  # 3 pass; claimed 5: the 3 true ones plus 2 false positives
        *arm_rows(MID, 5),
        *arm_rows(MAX, 6, claimed=4),  # claimed only the first 4 of 6 passes: 2 false negatives
    ]
    cheap, mid, top = calibrate(rows).arms
    assert (cheap.n, cheap.successes, cheap.claimed) == (10, 3, 5)
    assert (cheap.false_pos, cheap.false_neg) == (2, 0)
    assert cheap.rate == 0.3 and cheap.claimed_rate == 0.5
    assert cheap.fp_rate == pytest.approx(2 / 7) and cheap.fn_rate == 0.0
    assert (mid.false_pos, mid.false_neg) == (0, 0)
    assert (top.successes, top.false_neg, top.false_pos) == (6, 2, 0)
    assert top.fn_rate == pytest.approx(2 / 6) and top.fp_rate == 0.0


def test_rates_are_none_when_the_denominator_is_empty() -> None:
    r = calibrate([*arm_rows(CHEAP, 10), *arm_rows(MID, 0), *arm_rows(MAX, 5)])
    assert r.arms[0].fp_rate is None  # no failures to be falsely claimed on
    assert r.arms[1].fn_rate is None  # no passes to be falsely rejected


def test_only_pilot_tasks_the_base_seed_and_the_default_config_count() -> None:
    other_cfg = PipelineConfig(reproducer=False).canonical()
    rows = [
        *arm_rows(CHEAP, 5),
        row("t0", CHEAP, true_success=False, pipeline_config=other_cfg),  # an ablation row
        row("t1", CHEAP, true_success=False, seed=1),  # a repeat row
        row("not-pilot", CHEAP, true_success=True),  # a task outside the pilot
    ]
    cheap = calibrate(rows).arms[0]
    assert cheap.n == 10 and cheap.successes == 5


@pytest.mark.parametrize(("passed", "flagged"), [(3, True), (4, False), (7, False), (8, True)])
def test_the_max_arm_is_flagged_outside_40_to_70_percent(passed: int, flagged: bool) -> None:
    r = calibrate([*arm_rows(CHEAP, 5), *arm_rows(MID, 5), *arm_rows(MAX, passed)])
    hit = [f for f in r.flags if f.startswith("MAX ARM OUTSIDE 40-70%")]
    assert bool(hit) == flagged
    if flagged:
        assert ("too hard" if passed < 4 else "too easy") in hit[0]


@pytest.mark.parametrize(("passed", "ceiling"), [(8, False), (9, True), (10, True)])
def test_the_cheap_arm_is_flagged_at_90_percent_or_more(passed: int, ceiling: bool) -> None:
    r = calibrate([*arm_rows(CHEAP, passed), *arm_rows(MID, 8), *arm_rows(MAX, 5)])
    assert any(f.startswith("CHEAP ARM AT CEILING") for f in r.flags) == ceiling


@pytest.mark.parametrize(("passed", "near_zero"), [(0, True), (1, True), (2, False)])
def test_the_cheap_arm_near_zero_recommends_a_swap_but_switches_nothing(
    passed: int, near_zero: bool
) -> None:
    r = calibrate([*arm_rows(CHEAP, passed), *arm_rows(MID, 5), *arm_rows(MAX, 5)])
    hit = [f for f in r.flags if f.startswith("CHEAP ARM NEAR ZERO")]
    assert bool(hit) == near_zero
    if near_zero:
        assert "swap_in_cheap" in hit[0] and "Nothing is switched automatically" in hit[0]
        assert all(a.label != "swap_in_cheap" for a in matrix_models().matrix_arms())


def test_the_swap_advice_says_so_when_models_yaml_has_no_swap_arm(tmp_path: Path) -> None:
    edited = tmp_path / "models.yaml"
    line = "  - {label: swap_in_cheap, model: mock-noflex, optional: true}\n"
    text = MATRIX_MODELS.read_text()
    assert line in text
    edited.write_text(text.replace(line, ""))
    rows = [*arm_rows(CHEAP, 0), *arm_rows(MID, 5), *arm_rows(MAX, 5)]
    r = calibrate(rows, load_models(edited))
    assert any("defines no 'swap_in_cheap' arm" in f for f in r.flags)


def test_optional_arms_are_never_part_of_the_calibration() -> None:
    r = calibrate([*arm_rows(CHEAP, 5), *arm_rows(MID, 5), *arm_rows(MAX, 5)])
    assert [a.arm_id for a in r.arms] == [CHEAP, MID, MAX]


def test_wilson_interval_and_the_printed_table() -> None:
    lo, hi = cal.wilson(5, 10)
    assert (lo, hi) == (pytest.approx(0.2366, abs=1e-3), pytest.approx(0.7634, abs=1e-3))
    assert cal.wilson(0, 10)[0] == 0.0 and cal.wilson(10, 10)[1] == 1.0
    assert cal.wilson(0, 0) == (0.0, 1.0)
    low = calibrate([*arm_rows(CHEAP, 3), *arm_rows(MID, 5), *arm_rows(MAX, 5)])
    text = cal.format_calibration(low)
    assert "3/10" in text and "cheap = mock-luna@low" in text and "n is small" in text
    fine = calibrate([*arm_rows(CHEAP, 5), *arm_rows(MID, 5), *arm_rows(MAX, 5)])
    assert "no flags raised" in cal.format_calibration(fine)
