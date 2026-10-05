"""Cost formula (cached tokens, long-context multiplier, flex), buckets, and per-arm caps.

Prices here are SYNTHETIC (tests/fixtures/synthetic/models_matrix.yaml). The shipped
configs/models.yaml still has TODO prices, so the last test checks the formula against the shipped
file only when the human has filled it in, and says so when it skips.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from cascade.config import ResolvedBudget, load_models
from cascade.errors import BudgetExceededError, CascadeError
from cascade.harness.ledger import Ledger, cost_of_usage
from cascade.harness.providers.base import Usage
from tests.conftest import ROOT
from tests.matrix_helpers import (
    CHEAP,
    MAX,
    MID,
    ctx,
    matrix_budget,
    matrix_models,
    new_ledger,
    spend,
)

M = 1_000_000


def test_plain_call_is_input_output_and_reasoning_at_their_prices() -> None:
    p = matrix_models().prices_for("mock-sol")
    u = Usage(input_tokens=1000, output_tokens=200, reasoning_tokens=300)
    assert cost_of_usage(p, u) == pytest.approx((1000 * 10.0 + 200 * 40.0 + 300 * 50.0) / M)


def test_reasoning_bills_at_the_output_price_when_no_separate_price_is_configured() -> None:
    p = matrix_models().prices_for("mock-luna")
    u = Usage(input_tokens=0, output_tokens=100, reasoning_tokens=100)
    assert cost_of_usage(p, u) == pytest.approx(200 * 4.0 / M)


def test_cached_tokens_bill_at_the_cached_price_and_are_a_subset_of_input() -> None:
    p = matrix_models().prices_for("mock-luna")  # input 1.0, cached 0.25
    u = Usage(input_tokens=1000, output_tokens=0, cached_tokens=600)
    assert cost_of_usage(p, u) == pytest.approx((400 * 1.0 + 600 * 0.25) / M)


def test_cache_write_tokens_bill_at_their_price_and_cached_falls_back_to_input() -> None:
    p = matrix_models().prices_for("mock-sol")  # input 10, cache write 12.5, no cached price
    u = Usage(input_tokens=1000, output_tokens=0, cached_tokens=100, cache_write_tokens=200)
    assert cost_of_usage(p, u) == pytest.approx((700 * 10.0 + 100 * 10.0 + 200 * 12.5) / M)


def test_cached_and_written_tokens_never_exceed_input() -> None:
    p = matrix_models().prices_for("mock-luna")
    u = Usage(input_tokens=100, output_tokens=0, cached_tokens=500, cache_write_tokens=500)
    assert cost_of_usage(p, u) == pytest.approx(100 * 0.25 / M)


@pytest.mark.parametrize(
    ("input_tokens", "in_mult", "out_mult"),
    [(100_000, 1.0, 1.0), (100_001, 2.0, 1.5), (500_000, 2.0, 1.5)],
)
def test_long_context_multiplier_applies_strictly_above_the_threshold(
    input_tokens: int, in_mult: float, out_mult: float
) -> None:
    p = matrix_models().prices_for("mock-luna")  # threshold 100_000
    u = Usage(input_tokens=input_tokens, output_tokens=1000, reasoning_tokens=1000)
    expected = (input_tokens * 1.0 * in_mult + 2000 * 4.0 * out_mult) / M
    assert cost_of_usage(p, u) == pytest.approx(expected)


def test_per_model_thresholds_differ() -> None:
    u = Usage(input_tokens=150_000, output_tokens=0)
    assert cost_of_usage(matrix_models().prices_for("mock-luna"), u) == pytest.approx(
        150_000 * 1.0 * 2.0 / M
    )
    assert cost_of_usage(matrix_models().prices_for("mock-sol"), u) == pytest.approx(
        150_000 * 10.0 / M  # sol's threshold is 200_000, so no multiple yet
    )


def test_a_model_without_the_rule_never_gets_a_multiplier() -> None:
    p = matrix_models().prices_for("mock-noflex")
    u = Usage(input_tokens=5_000_000, output_tokens=1_000_000)
    assert cost_of_usage(p, u) == pytest.approx((5_000_000 * 2.0 + 1_000_000 * 8.0) / M)


def test_flex_halves_the_cost_and_composes_with_long_context() -> None:
    p = matrix_models().prices_for("mock-luna")
    u = Usage(input_tokens=200_000, output_tokens=1000)
    standard = cost_of_usage(p, u, "standard")
    assert cost_of_usage(p, u, "flex") == pytest.approx(standard * 0.5)
    assert standard == pytest.approx((200_000 * 1.0 * 2.0 + 1000 * 4.0 * 1.5) / M)


def test_flex_on_a_model_without_a_multiplier_is_refused_and_so_is_an_unknown_tier() -> None:
    p = matrix_models().prices_for("mock-noflex")
    with pytest.raises(CascadeError, match="no flex_multiplier"):
        cost_of_usage(p, Usage(input_tokens=1, output_tokens=1), "flex", model="mock-noflex")
    with pytest.raises(CascadeError, match="unknown processing tier"):
        cost_of_usage(p, Usage(input_tokens=1, output_tokens=1), "priority")


def test_worst_case_includes_the_long_context_multiple_and_the_flex_discount(
    tmp_path: Path,
) -> None:
    ledger = new_ledger(tmp_path)
    short = ledger.worst_case_cost("mock-luna", 1000, 1000)
    assert short == pytest.approx((1000 * 1.0 + 1000 * 4.0) / M)
    long = ledger.worst_case_cost("mock-luna", 200_000, 1000)
    assert long == pytest.approx((200_000 * 1.0 * 2.0 + 1000 * 4.0 * 1.5) / M)
    assert ledger.worst_case_cost("mock-luna", 1000, 1000, "flex") == pytest.approx(short * 0.5)
    ledger.close()


def test_the_ledger_logs_cost_tier_bucket_and_cache_columns(tmp_path: Path) -> None:
    ledger = new_ledger(tmp_path)
    rec = ledger.record(
        ctx(MAX, bucket="pilot", tier="flex"), provider="mock", model="mock-sol",
        usage=Usage(input_tokens=1000, output_tokens=100, cached_tokens=10, cache_write_tokens=20),
        latency_s=0.5,
    )  # fmt: skip
    assert rec.processing_tier == "flex" and rec.bucket == "pilot"
    assert rec.cached_tokens == 10 and rec.cache_write_tokens == 20
    assert rec.cost_usd == pytest.approx(
        (970 * 10.0 + 10 * 10.0 + 20 * 12.5 + 100 * 40.0) / M * 0.5
    )
    totals = ledger.attempt_totals(task_id="t1", attempt_id="a1")
    assert (totals.cached_tokens, totals.cache_write_tokens) == (10, 20)
    ledger.close()


def test_an_old_ledger_file_is_migrated_in_place_with_defaults(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE calls (call_id TEXT PRIMARY KEY, ts TEXT NOT NULL, run_id TEXT NOT NULL,
          task_id TEXT NOT NULL, attempt_id TEXT NOT NULL, arm_id TEXT NOT NULL, role TEXT NOT NULL,
          provider TEXT NOT NULL, model TEXT NOT NULL, input_tokens INTEGER NOT NULL,
          output_tokens INTEGER NOT NULL, reasoning_tokens INTEGER NOT NULL,
          cost_usd REAL NOT NULL, latency_s REAL NOT NULL, status TEXT NOT NULL, error TEXT);
        INSERT INTO calls VALUES ('c0','2025-01-01T00:00:00+00:00','r','t0','a0','m@low','patcher',
          'mock','mock-luna',10,1,0,0.5,0.1,'ok',NULL);
        """
    )
    conn.commit()
    conn.close()
    ledger = Ledger(db, budget=matrix_budget(tmp_path), models=matrix_models())
    [old] = ledger.rows()
    assert old.processing_tier == "standard" and old.bucket == "default" and old.cached_tokens == 0
    assert ledger.spent_usd() == pytest.approx(0.5)  # old spend still counts against the caps
    spend(ledger, CHEAP, 1000, call_id="c9")  # and new rows write fine
    assert len(ledger.rows()) == 2
    ledger.close()


# --- caps ---------------------------------------------------------------------------------------


def test_bucket_cap_enforced_on_worst_case_and_other_buckets_are_unaffected(tmp_path: Path) -> None:
    ledger = new_ledger(tmp_path, buckets={"pilot": 0.001, "matrix": 20.0})
    call = {
        "est_input_tokens": 1000,
        "max_output_tokens": 1000,
        "real": False,
        "model": "mock-luna",
    }
    with pytest.raises(BudgetExceededError, match="bucket cap"):
        ledger.authorize(ctx(CHEAP, bucket="pilot"), **call)  # worst case $0.005 > $0.001
    ledger.authorize(ctx(CHEAP, bucket="matrix", call_id="c2"), **call)  # a roomy bucket is fine
    ledger.authorize(ctx(CHEAP, bucket="no-such-bucket", call_id="c3"), **call)  # uncapped
    ledger.close()


def test_bucket_spend_counts_only_that_buckets_calls(tmp_path: Path) -> None:
    ledger = new_ledger(tmp_path)
    a = spend(ledger, CHEAP, 1000, bucket="pilot", call_id="c1")
    b = spend(ledger, CHEAP, 2000, bucket="matrix", call_id="c2", attempt_id="a2")
    assert ledger.bucket_spent_usd("pilot") == pytest.approx(a)
    assert ledger.bucket_spent_usd("matrix") == pytest.approx(b)
    assert ledger.bucket_spent_usd("ablate") == 0.0
    ledger.close()


def test_each_arm_has_its_own_per_attempt_cap_and_the_general_cap_is_the_fallback(
    tmp_path: Path,
) -> None:
    ledger = new_ledger(tmp_path)
    assert ledger.attempt_cap(CHEAP) == 0.2 and ledger.attempt_cap(MID) == 0.5
    assert ledger.attempt_cap(MAX) == 1.0
    assert ledger.attempt_cap("unlabelled@x") == 1.0  # per_attempt_cap_usd
    ledger.close()


def test_the_cheap_arm_is_stopped_at_its_own_small_cap_while_mid_may_spend_more(
    tmp_path: Path,
) -> None:
    ledger = new_ledger(tmp_path)
    # worst case of this call: 100k input * $1/M + 50k out * $4/M = $0.30 (cheap cap $0.20)
    call = {"est_input_tokens": 100_000, "max_output_tokens": 50_000, "real": False}
    with pytest.raises(BudgetExceededError, match="attempt cap"):
        ledger.authorize(ctx(CHEAP), model="mock-luna", **call)
    ledger.authorize(ctx(MID, call_id="c2"), model="mock-luna", **call)  # mid cap is $0.50
    ledger.close()


def test_per_arm_caps_may_not_exceed_the_general_attempt_cap(tmp_path: Path) -> None:
    data = matrix_budget(tmp_path).model_dump()
    with pytest.raises(ValueError, match="exceeds per_attempt_cap_usd"):
        ResolvedBudget.model_validate({**data, "per_arm_attempt_cap_usd": {"max": 9.0}})


def test_buckets_may_not_sum_above_the_global_cap(tmp_path: Path) -> None:
    data = matrix_budget(tmp_path).model_dump()
    with pytest.raises(ValueError, match="sum to at most"):
        ResolvedBudget.model_validate({**data, "buckets": {"a": 30.0, "b": 30.0}})


def test_cost_formula_matches_the_shipped_luna_and_sol_prices_when_filled() -> None:
    models = load_models(ROOT / "configs" / "models.yaml")
    todo = [p for p in models.unfilled() if "models[0]" in p or "models[1]" in p]
    if todo:
        pytest.skip(
            "configs/models.yaml still has TODO prices for gpt-6-luna / gpt-6.1-sol "
            f"({len(todo)} field(s)); fill them and this test checks the formula against them"
        )
    for name in ("gpt-6-luna", "gpt-6.1-sol"):
        p = models.prices_for(name)
        u = Usage(input_tokens=1000, output_tokens=100, reasoning_tokens=50)
        base = (
            1000 * p.input_usd_per_mtok
            + 100 * p.output_usd_per_mtok
            + 50 * p.reasoning_usd_per_mtok
        ) / M
        assert cost_of_usage(p, u) == pytest.approx(base)
        if p.long_context_threshold_tokens:
            n = p.long_context_threshold_tokens + 1
            big = Usage(input_tokens=n, output_tokens=100)
            expected = (
                n * p.input_usd_per_mtok * p.long_context_input_multiplier
                + 100 * p.output_usd_per_mtok * p.long_context_output_multiplier
            ) / M
            assert cost_of_usage(p, big) == pytest.approx(expected)
