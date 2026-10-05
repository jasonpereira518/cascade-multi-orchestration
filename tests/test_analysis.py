"""Paired comparisons for ablations (same seed, different pipeline) and repeats (same pipeline,
different seed). Synthetic rows. Each count has a distinct value so a swapped side cannot hide."""

from __future__ import annotations

from cascade.agents.pipeline import PipelineConfig
from cascade.routing import analysis as an
from cascade.routing.matrix import MatrixRow
from tests.matrix_helpers import MID, row

FULL = PipelineConfig()
NO_REPRO = an.ablation_configs(FULL)["no_reproducer"]
BASE = (0, FULL.canonical())
OTHER = (0, NO_REPRO.canonical())


def rows(
    base: dict[str, bool],
    other: dict[str, bool],
    *,
    cost_base: float = 0.1,
    cost_other: float = 0.3,
    claimed_base: set[str] | None = None,
    claimed_other: set[str] | None = None,
) -> list[MatrixRow]:
    said_base, said_other = claimed_base or set(), claimed_other or set()
    out = [
        row(t, MID, true_success=ok, verifier_claimed=t in said_base, cost_usd=cost_base)
        for t, ok in base.items()
    ]
    out += [
        row(
            t, MID, true_success=ok, verifier_claimed=t in said_other, cost_usd=cost_other,
            pipeline_config=NO_REPRO.canonical(),
        )
        for t, ok in other.items()
    ]  # fmt: skip
    return out


def pair(data: list[MatrixRow], ids: set[str]) -> an.Paired:
    return an.pair(data, arm_id=MID, label="x", task_ids=ids, base=BASE, other=OTHER)


def test_counts_are_split_by_side_and_each_side_has_its_own_value() -> None:
    base = {"a": True, "b": True, "c": True, "d": False, "e": False, "f": True}
    other = {"a": True, "b": False, "c": False, "d": True, "e": False, "f": False}
    p = pair(rows(base, other), set(base))
    assert p.tasks_with_both == 6
    assert (p.base_pass, p.other_pass) == (4, 2)
    assert p.both_pass == 1  # only "a"
    assert p.only_base == 3  # b, c, f passed with the full pipeline and failed without
    assert p.only_other == 1  # d passed only without the reproducer
    assert p.agreement == 1 - 4 / 6  # disagreements are only_base + only_other


def test_only_tasks_with_both_rows_and_inside_the_task_set_are_compared() -> None:
    base = {"a": True, "b": True, "c": True}
    other = {"a": False, "b": True, "z": True}  # c has no ablation row; z has no baseline
    p = pair(rows(base, other), {"a", "b", "c", "z", "outside"})
    assert p.tasks_with_both == 2 and (p.base_pass, p.other_pass) == (2, 1)
    narrow = pair(rows(base, other), {"a"})
    assert narrow.tasks_with_both == 1 and narrow.only_base == 1


def test_costs_and_claims_are_means_and_counts_over_the_paired_tasks_only() -> None:
    base = {"a": True, "b": False}
    other = {"a": True, "b": False}
    data = rows(
        base, other, cost_base=0.2, cost_other=0.5, claimed_base={"a", "b"}, claimed_other={"a"}
    )
    data.append(row("solo", MID, cost_usd=99.0))  # unpaired: must not enter the means
    p = pair(data, {"a", "b", "solo"})
    assert p.base_cost == 0.2 and p.other_cost == 0.5
    assert (p.base_claimed, p.other_claimed) == (2, 1)


def test_no_pairs_gives_zeroes_and_no_agreement_instead_of_dividing_by_zero() -> None:
    p = pair(rows({"a": True}, {}), {"a"})
    assert p.tasks_with_both == 0 and p.agreement is None
    assert (p.base_cost, p.other_cost) == (0.0, 0.0)
    assert "0 paired tasks" in an.format_paired("T", "full", [p])


def test_the_seed_and_arm_select_the_rows() -> None:
    cfg = FULL.canonical()
    data = [
        row("a", MID, seed=0, true_success=True),
        row("a", MID, seed=1, true_success=False),
        row("a", "other@arm", seed=1, true_success=True),
    ]
    p = an.pair(data, arm_id=MID, label="r", task_ids={"a"}, base=(0, cfg), other=(1, cfg))
    assert (p.tasks_with_both, p.only_base, p.only_other) == (1, 1, 0)


def test_the_two_ablations_each_remove_exactly_one_role_and_keep_everything_else() -> None:
    base = PipelineConfig(
        max_repairs=3, max_output_tokens=8000, test_timeout_s=77.0, max_test_runs_per_role=9
    )
    configs = an.ablation_configs(base)
    assert set(configs) == {"no_reproducer", "no_reviewer"}
    no_repro, no_review = configs["no_reproducer"], configs["no_reviewer"]
    assert (no_repro.reproducer, no_repro.reviewer) == (False, True)
    assert (no_review.reproducer, no_review.reviewer) == (True, False)
    for c in (no_repro, no_review):
        assert (c.max_repairs, c.max_output_tokens, c.test_timeout_s) == (3, 8000, 77.0)
        assert c.max_test_runs_per_role == 9
    assert len({base.canonical(), no_repro.canonical(), no_review.canonical()}) == 3


def test_the_printed_table_carries_each_comparison_and_its_caveat() -> None:
    p = pair(rows({"a": True}, {"a": False}), {"a"})
    text = an.format_paired("ABLATIONS", "full", [p])
    assert text.startswith("ABLATIONS") and "small n" in text
    assert "base-only" in text and "other-only" in text
