"""Features, the two small models, the lookup policies, and the predictive router's selection rule.

Policy tests use hand-built tasks whose outcomes are chosen to make each rule's behaviour obvious.
Everything is SYNTHETIC.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from cascade.routing import features as ft
from cascade.routing import policies as pol
from cascade.routing.learn import LogisticModel, RidgeLogCost
from cascade.routing.policies import Cell, TaskRecord
from tests.phase3_helpers import make_task

Spec = tuple[bool, bool, float, float]  # success, claimed, cost, wall

# --- features -----------------------------------------------------------------------------------


def test_feature_groups_split_issue_side_from_gold_derived() -> None:
    assert set(ft.ISSUE_FEATURES).isdisjoint(ft.GOLD_FEATURES)
    assert {"src_files", "src_lines", "test_runtime_s"} <= set(ft.GOLD_FEATURES)
    assert {"issue_chars", "code_blocks", "has_traceback"} <= set(ft.ISSUE_FEATURES)
    assert set(ft.ALL_FEATURES) == set(ft.ISSUE_FEATURES) | set(ft.GOLD_FEATURES)
    assert ft.FEATURE_SETS["issue"] == ft.ISSUE_FEATURES
    assert ft.FEATURE_SETS["all"] == ft.ALL_FEATURES
    assert all(ft.BY_NAME[n].gold_derived for n in ft.GOLD_FEATURES)


def test_the_four_spec_features_are_present() -> None:
    for name in ("src_files", "src_lines", "issue_chars", "test_runtime_s"):
        assert name in ft.BY_NAME


def test_feature_matrix_shape_order_and_log_transform() -> None:
    t = make_task("o/a", 1)
    x = ft.feature_matrix([t], ["issue_chars", "src_files"])
    assert x.shape == (1, 2)
    assert x[0, 0] == pytest.approx(np.log1p(t.features.issue_chars))  # log feature
    assert x[0, 1] == float(t.features.src_files)  # raw feature
    assert ft.feature_matrix([], ft.ALL_FEATURES).shape == (0, len(ft.ALL_FEATURES))


def test_text_features_are_deterministic_functions_of_the_text() -> None:
    t = make_task(
        "o/a", 1, issue_title="Crash in parse",
        issue_body="See https://x.io/a and v1.2.3. Why? It breaks.",
    )  # fmt: skip
    names = ["n_urls", "mentions_version", "n_sentences", "title_chars", "issue_lines"]
    a = ft.feature_matrix([t], names)
    assert np.array_equal(a, ft.feature_matrix([t], names))
    assert a[0, 0] == 1.0 and a[0, 1] == 1.0
    assert a[0, 3] == pytest.approx(np.log1p(len("Crash in parse")))


def test_unknown_feature_names_are_an_error() -> None:
    with pytest.raises(KeyError, match="nope"):
        ft.feature_matrix([make_task()], ["issue_chars", "nope"])


def test_no_feature_depends_on_a_model_provider() -> None:
    source = inspect.getsource(ft)
    assert "openai" not in source and "from agents" not in source and "import agents" not in source


# --- the models ---------------------------------------------------------------------------------


def _toy(n: int = 80, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] * 2.0 + rng.normal(scale=0.5, size=n)) > 0
    return x, y


def test_logistic_regression_finds_the_signal_and_is_deterministic() -> None:
    x, y = _toy()
    m1, m2 = LogisticModel.fit(x, y, c=1.0), LogisticModel.fit(x, y, c=1.0)
    assert np.array_equal(m1.weights, m2.weights) and m1.intercept == m2.intercept
    assert abs(m1.weights[0]) > 3 * max(abs(m1.weights[1]), abs(m1.weights[2]))
    p = m1.predict_proba(x)
    assert np.all((p > 0) & (p < 1))
    assert float(np.mean((p > 0.5) == y)) > 0.85


def test_stronger_regularization_shrinks_the_weights() -> None:
    x, y = _toy()
    weak = LogisticModel.fit(x, y, c=10.0)
    strong = LogisticModel.fit(x, y, c=0.01)
    assert np.linalg.norm(strong.weights) < 0.2 * np.linalg.norm(weak.weights)


def test_a_class_that_is_all_one_value_still_gives_finite_probabilities() -> None:
    x, _ = _toy()
    for label in (True, False):
        m = LogisticModel.fit(x, np.full(len(x), label), c=0.1)
        p = m.predict_proba(x)
        assert np.all(np.isfinite(p)) and (p.mean() > 0.5) == label


def test_a_constant_feature_column_does_not_break_the_fit() -> None:
    x, y = _toy()
    x[:, 1] = 7.0
    m = LogisticModel.fit(x, y, c=1.0)
    assert np.all(np.isfinite(m.weights)) and m.weights[1] == pytest.approx(0.0, abs=1e-6)


def test_logistic_input_validation() -> None:
    x, y = _toy(10)
    with pytest.raises(ValueError, match="positive"):
        LogisticModel.fit(x, y, c=0.0)
    with pytest.raises(ValueError, match="differ"):
        LogisticModel.fit(x, y[:5], c=1.0)


def test_ridge_log_cost_recovers_a_multiplicative_cost_and_never_goes_negative() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 2))
    cost = np.exp(0.5 * x[:, 0] - 2.0 + rng.normal(scale=0.8, size=200))  # noisy: smearing matters
    m = RidgeLogCost.fit(x, cost, lam=1.0)
    pred = m.predict(x)
    assert np.all(pred > 0)
    assert float(np.corrcoef(np.log(pred), np.log(cost))[0, 1]) > 0.4
    assert pred.mean() == pytest.approx(cost.mean(), rel=0.1)  # smearing keeps the mean unbiased
    with pytest.raises(ValueError, match="negative"):
        RidgeLogCost.fit(x, cost, lam=-1.0)


# --- the lookup policies ------------------------------------------------------------------------


def tri(c0: Spec, c1: Spec, c2: Spec) -> tuple[Cell, Cell, Cell]:
    return Cell(*c0), Cell(*c1), Cell(*c2)


def rec(c0: Spec, c1: Spec, c2: Spec, repo: str = "o/a", n: int = 1) -> TaskRecord:
    return TaskRecord(make_task(repo, n), tri(c0, c1, c2))


def test_fixed_policies_read_one_arm() -> None:
    r = rec((False, False, 1.0, 10.0), (True, True, 2.0, 20.0), (True, True, 5.0, 50.0))
    assert pol.fixed("cheap")(r) == pol.Outcome(1.0, False, 10.0, 1, 0)
    assert pol.fixed("mid")(r) == pol.Outcome(2.0, True, 20.0, 1, 1)
    assert pol.fixed("max")(r) == pol.Outcome(5.0, True, 50.0, 1, 2)
    with pytest.raises(pol.PolicyError):
        pol.fixed("huge")


def test_cascade_stops_at_the_first_claimed_success_and_pays_only_for_arms_tried() -> None:
    r = rec((True, True, 1.0, 10.0), (True, True, 2.0, 20.0), (True, True, 5.0, 50.0))
    assert pol.cascade(r) == pol.Outcome(1.0, True, 10.0, 1, 0)


def test_cascade_escalates_while_the_verifier_rejects_and_sums_costs_and_latency() -> None:
    r = rec((False, False, 1.0, 10.0), (True, True, 2.0, 20.0), (True, True, 5.0, 50.0))
    assert pol.cascade(r) == pol.Outcome(3.0, True, 30.0, 2, 1)
    r2 = rec((False, False, 1.0, 10.0), (False, False, 2.0, 20.0), (True, True, 5.0, 50.0))
    assert pol.cascade(r2) == pol.Outcome(8.0, True, 80.0, 3, 2)


def test_a_verifier_false_positive_stops_the_cascade_and_counts_as_a_failure() -> None:
    r = rec((False, True, 1.0, 10.0), (True, True, 2.0, 20.0), (True, True, 5.0, 50.0))
    out = pol.cascade(r)
    assert out.success is False and out.arms_tried == 1 and out.cost == 1.0  # never reached mid


def test_a_verifier_false_negative_costs_an_unnecessary_escalation() -> None:
    r = rec((True, False, 1.0, 10.0), (True, True, 2.0, 20.0), (True, True, 5.0, 50.0))
    out = pol.cascade(r)
    assert out.success is True and out.arms_tried == 2 and out.cost == 3.0


def test_the_last_arm_ends_the_cascade_and_its_true_outcome_is_the_result() -> None:
    r = rec((False, False, 1.0, 10.0), (False, False, 2.0, 20.0), (False, False, 5.0, 50.0))
    assert pol.cascade(r) == pol.Outcome(8.0, False, 80.0, 3, 2)
    r2 = rec((False, False, 1.0, 10.0), (False, False, 2.0, 20.0), (True, False, 5.0, 50.0))
    assert pol.cascade(r2).success is True  # outcome is true_success, not the claim


def test_oracle_picks_the_cheapest_successful_arm_by_observed_cost() -> None:
    r = rec((False, False, 1.0, 10.0), (True, True, 4.0, 20.0), (True, True, 3.0, 50.0))
    out = pol.oracle(r)  # max is cheaper than mid on this task, and both succeed
    assert (out.arm, out.cost, out.success) == (2, 3.0, True)


def test_oracle_with_no_success_takes_the_cheapest_arm_and_ties_go_to_the_lower_arm() -> None:
    r = rec((False, False, 2.0, 10.0), (False, False, 1.0, 20.0), (False, False, 3.0, 50.0))
    assert pol.oracle(r).arm == 1 and pol.oracle(r).success is False
    tie = rec((True, True, 1.0, 10.0), (True, True, 1.0, 20.0), (True, True, 1.0, 50.0))
    assert pol.oracle(tie).arm == 0


def test_the_oracle_succeeds_whenever_any_arm_does_and_costs_the_minimum_of_that_set() -> None:
    cases = [
        ((False, False), (False, False), (True, True)),
        ((True, True), (True, True), (True, True)),
        ((False, False), (False, False), (False, False)),
        ((False, False), (True, True), (True, True)),
    ]
    recs: list[TaskRecord] = []
    for i, (a, b, c) in enumerate(cases):
        recs.append(rec((*a, 1.0 + i, 10.0), (*b, 2.0 + i, 20.0), (*c, 0.5 + 3 * i, 50.0), n=i))
    for r, o in zip(recs, pol.evaluate(pol.oracle, recs), strict=True):
        winners = [c.cost for c in r.cells if c.success]
        assert o.success == bool(winners)
        assert o.cost == min(winners or [c.cost for c in r.cells])
    for arm in pol.ARMS:
        fixed_wins = sum(x.success for x in pol.evaluate(pol.fixed(arm), recs))
        assert sum(x.success for x in pol.evaluate(pol.oracle, recs)) >= fixed_wins


def test_arrays_and_choice_outcomes_agree_with_the_cells() -> None:
    recs = [rec((True, True, 1.0, 5.0), (False, False, 2.0, 6.0), (True, True, 3.0, 7.0))]
    s, k, w = pol.arrays(recs)
    assert s.tolist() == [[True, False, True]] and k.tolist() == [[1.0, 2.0, 3.0]]
    assert w.tolist() == [[5.0, 6.0, 7.0]]
    assert pol.outcomes_of_choices(recs, [1])[0] == pol.Outcome(2.0, False, 6.0, 1, 1)


def test_choose_arms_maximizes_p_times_value_minus_cost_and_breaks_ties_toward_cheaper() -> None:
    p = np.array([[0.2, 0.5, 0.9], [0.5, 0.5, 0.5]])
    c = np.array([[1.0, 2.0, 5.0], [1.0, 1.0, 1.0]])
    assert pol.choose_arms(p, c, np.array([10.0, 10.0])) == [2, 0]  # 9-5 > 5-2 > 2-1; tie -> 0
    assert pol.choose_arms(p, c, np.array([1.0, 1.0])) == [0, 0]  # a low value prefers cheap


# --- the predictive router ----------------------------------------------------------------------


def synthetic_records(
    n_per_repo: int = 12, repos: tuple[str, ...] = ("o/a", "o/b", "o/c")
) -> list[TaskRecord]:
    rng = np.random.default_rng(3)
    out: list[TaskRecord] = []
    for repo in repos:
        for i in range(n_per_repo):
            hard = bool(rng.random() < 0.5)
            t = make_task(repo, i)
            feat = t.features.model_copy(
                update={"issue_chars": 2000 if hard else 200, "issue_words": 300 if hard else 30}
            )
            ok = (False, False, True) if hard else (True, True, True)
            scale = 1.5 if hard else 1.0
            cells = tri(
                (ok[0], ok[0], 0.05 * scale, 10.0),
                (ok[1], ok[1], 0.12 * scale, 20.0),
                (ok[2], ok[2], 0.60 * scale, 30.0),
            )
            out.append(TaskRecord(t.model_copy(update={"features": feat}), cells))
    return out


def test_loro_needs_two_repos_and_predicts_each_task_without_its_own_repo() -> None:
    with pytest.raises(pol.PolicyError, match="at least two repos"):
        pol.loro_predictions(synthetic_records(repos=("o/a",)), "issue", 0.1)
    recs = synthetic_records()
    oof = pol.loro_predictions(recs, "issue", 0.1)
    assert oof.p.shape == (len(recs), 3) and np.all(oof.value_scale > 0)
    # corrupting one repo's outcomes must leave that repo's own out-of-fold predictions unchanged:
    # the model that predicts a repo never saw that repo's labels
    flipped = [
        TaskRecord(
            r.task,
            tri(*[(not c.success, not c.claimed, c.cost, c.wall_s) for c in r.cells]),  # type: ignore[arg-type]
        )
        if r.repo == "o/a"
        else r
        for r in recs
    ]
    oof2 = pol.loro_predictions(flipped, "issue", 0.1)
    idx = [i for i, r in enumerate(recs) if r.repo == "o/a"]
    assert np.allclose(oof.p[idx], oof2.p[idx])
    other = [i for i, r in enumerate(recs) if r.repo != "o/a"]
    assert not np.allclose(oof.p[other], oof2.p[other])  # others DO see the flipped labels


def test_selection_picks_the_cheapest_feasible_candidate_and_is_deterministic() -> None:
    recs = synthetic_records()
    a = pol.select_config(recs, "issue", margin=0.05)
    b = pol.select_config(recs, "issue", margin=0.05)
    assert a == b and a.chosen is not None
    assert len(a.candidates) == len(pol.CS) * len(pol.VALUE_MULTIPLIERS)
    feasible = [c for c in a.candidates if c.feasible]
    assert a.chosen in feasible
    assert a.chosen.mean_cost == min(c.mean_cost for c in feasible)
    assert all(c.resolve_rate >= a.max_resolve_rate - 0.05 - 1e-9 for c in feasible)
    infeasible = [c for c in a.candidates if not c.feasible]
    assert all(c.resolve_rate < a.max_resolve_rate - 0.05 + 1e-9 for c in infeasible)


def test_with_an_impossible_margin_no_candidate_is_feasible_and_the_router_is_always_max() -> None:
    recs = synthetic_records()
    sel = pol.select_config(recs, "issue", margin=-1.0)  # nothing can beat max by 100 points
    assert sel.chosen is None and not any(c.feasible for c in sel.candidates)
    router = pol.fit_final_router(recs, "issue", margin=-1.0)
    assert router.model is None and router.choose(recs) == [pol.MAX_ARM] * len(recs)


def test_ties_in_cost_go_to_the_smaller_c_then_the_smaller_multiplier() -> None:
    recs = synthetic_records()
    sel = pol.select_config(recs, "issue", margin=0.5)  # almost everything feasible
    best = min(c.mean_cost for c in sel.candidates if c.feasible)
    tied = [c for c in sel.candidates if c.feasible and round(c.mean_cost, 12) == round(best, 12)]
    assert sel.chosen == min(tied, key=lambda c: (c.c, c.multiplier))


def test_nested_cv_needs_three_repos_and_chooses_for_every_task() -> None:
    assert pol.nested_choices(synthetic_records(repos=("o/a", "o/b")), "issue", 0.05) is None
    recs = synthetic_records()
    choices = pol.nested_choices(recs, "issue", 0.05)
    assert choices is not None and len(choices) == len(recs)
    assert set(choices) <= {0, 1, 2}
    assert choices == pol.nested_choices(recs, "issue", 0.05)


def test_nested_selection_never_sees_the_repo_it_is_choosing_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recs = synthetic_records()
    seen: list[set[str]] = []
    original = pol.select_config

    def spy(
        records: list[TaskRecord], feature_set: str, margin: float, **kw: object
    ) -> pol.Selection:
        seen.append({r.repo for r in records})
        return original(records, feature_set, margin, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(pol, "select_config", spy)
    pol.nested_choices(recs, "issue", 0.05)
    all_repos = {"o/a", "o/b", "o/c"}
    assert len(seen) == 3 and all(len(s) == 2 and s < all_repos for s in seen)
    assert {frozenset(all_repos - s) for s in seen} == {frozenset({r}) for r in all_repos}


def test_the_value_scale_is_the_mean_max_arm_cost_of_the_training_tasks() -> None:
    recs = synthetic_records()
    model = pol.PredictiveModel.fit(recs, "issue", 0.1)
    assert model.value_scale == pytest.approx(np.mean([r.cells[2].cost for r in recs]))
    assert model.value_scale != pytest.approx(np.mean([r.cells[0].cost for r in recs]))


def test_the_router_learns_to_send_easy_tasks_to_cheaper_arms_than_hard_ones() -> None:
    recs = synthetic_records()
    router = pol.fit_final_router(recs, "issue", margin=0.05)
    choices = router.choose(recs)
    hard = [a for a, r in zip(choices, recs, strict=True) if r.task.features.issue_chars > 1000]
    easy = [a for a, r in zip(choices, recs, strict=True) if r.task.features.issue_chars <= 1000]
    assert np.mean(easy) <= np.mean(hard)
    outcomes = pol.outcomes_of_choices(recs, choices)
    always_max = pol.evaluate(pol.fixed("max"), recs)
    assert sum(o.cost for o in outcomes) < sum(o.cost for o in always_max)


def test_fitting_on_nothing_is_refused() -> None:
    with pytest.raises(pol.PolicyError, match="no tasks"):
        pol.PredictiveModel.fit([], "issue", 0.1)
