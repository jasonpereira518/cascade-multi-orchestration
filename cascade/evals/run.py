"""Run an evaluation of the routing policies on a dataset. Pure data out; ``report`` renders it.

Two modes, one code path:

- **dev** (``mode="dev"``): evaluates on TRAIN tasks. Fixed, cascade, and oracle are plain lookups.
  The predictive router is evaluated by NESTED leave-one-repo-out CV, so its numbers are honest
  out-of-sample estimates (selecting the configuration on the same folds that are then reported
  would flatter it); the single-level CV sweep is kept for the plot and the selection table.
  Results are labelled TRAIN/CV and are not evidence for any claim.
- **final** (``mode="final"``): the router is fit on ALL train tasks (configuration chosen by
  leave-one-repo-out CV on train only) and evaluated ONCE on the test tasks. The test split is used
  for nothing else. ``evals.final`` is the only caller and guards that this happens exactly once.

The two predictive variants are reported separately: ``predictive_issue`` uses issue-side features a
deployed router could compute, ``predictive_all`` also uses gold-derived features and is an upper
bound, not a deployable router (see routing/features.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from cascade.evals import bootstrap as bs
from cascade.evals import metrics as mt
from cascade.evals import svg
from cascade.evals.data import Dataset
from cascade.routing import policies as pol
from cascade.routing.policies import Outcome, TaskRecord

PREDICTIVE = {"predictive_issue": "issue", "predictive_all": "all"}
_FIXED = {"always_cheap": "cheap", "always_mid": "mid", "always_max": "max"}


@dataclass(frozen=True)
class EvalResult:
    mode: str
    stamp: str
    provenance: dict[str, Any]
    n_tasks: int
    n_repos: int
    margin: float
    margin_basis: str
    resamples: int
    seed: int
    metrics: list[mt.PolicyMetrics]
    comparisons: list[mt.Comparison]
    per_repo: list[mt.RepoRow]
    verifier: list[mt.VerifierRow]
    ablation: list[mt.AblationRow]
    noise: list[mt.NoiseRow]
    selections: dict[str, dict[str, Any]]  # feature set -> chosen configuration and CV summary
    notes: list[str]
    pareto: list[svg.Point]


def _single_level_choices(
    train: Sequence[TaskRecord], feature_set: str, sel: pol.Selection
) -> list[int]:
    """Out-of-fold choices of the selected configuration (optimistic: selected on these folds)."""
    if sel.chosen is None:
        return [pol.MAX_ARM] * len(train)
    oof = pol.loro_predictions(train, feature_set, sel.chosen.c)
    return pol.choose_arms(oof.p, oof.cost_hat, sel.chosen.multiplier * oof.value_scale)


def policy_outcomes(
    train: Sequence[TaskRecord],
    evaluate_on: Sequence[TaskRecord],
    *,
    mode: str,
    selection_margin: float,
) -> tuple[dict[str, list[Outcome]], dict[str, pol.Selection], list[str]]:
    """Outcomes of every policy on ``evaluate_on``; the router is learned from ``train`` only."""
    out: dict[str, list[Outcome]] = {
        name: pol.evaluate(pol.fixed(arm), evaluate_on) for name, arm in _FIXED.items()
    }
    out["cascade"] = pol.evaluate(pol.cascade, evaluate_on)
    out[mt.ORACLE] = pol.evaluate(pol.oracle, evaluate_on)
    selections: dict[str, pol.Selection] = {}
    notes: list[str] = []
    for name, feature_set in PREDICTIVE.items():
        if mode == "dev":
            sel = pol.select_config(train, feature_set, selection_margin)
            nested = pol.nested_choices(train, feature_set, selection_margin)
            if nested is None:
                notes.append(
                    f"{name}: nested CV needs at least 3 repos in train, so the single-level CV "
                    "choices of the selected configuration are reported instead (optimistic)"
                )
            choices = (
                nested if nested is not None else _single_level_choices(train, feature_set, sel)
            )
            selections[feature_set] = sel
        else:
            router = pol.fit_final_router(train, feature_set, selection_margin)
            selections[feature_set] = router.selection
            choices = router.choose(evaluate_on)
        out[name] = pol.outcomes_of_choices(evaluate_on, choices)
    return out, selections, notes


def _pareto(metrics: Sequence[mt.PolicyMetrics], sels: dict[str, pol.Selection]) -> list[svg.Point]:
    pts: list[svg.Point] = []
    for m in metrics:
        if m.name.startswith("always"):
            kind = "fixed"
        elif m.name == "cascade":
            kind = "cascade"
        elif m.name == mt.ORACLE:
            kind = "oracle"
        else:
            kind = "predictive"
        pts.append(
            svg.Point(
                m.name, kind, m.cost_per_task.point, m.resolve_rate.point,
                m.cost_per_task.lo, m.cost_per_task.hi, m.resolve_rate.lo, m.resolve_rate.hi,
            )
        )  # fmt: skip
    issue = sels.get("issue")
    if issue is not None and issue.candidates:
        best_c = issue.chosen.c if issue.chosen else min(c.c for c in issue.candidates)
        pts += [
            svg.Point(f"cv m={c.multiplier:g}", "sweep", c.mean_cost, c.resolve_rate)
            for c in issue.candidates
            if c.c == best_c
        ]
    return pts


def evaluate_dataset(
    *,
    mode: str,
    train: Dataset,
    target: Dataset,
    margin: float,
    margin_basis: str,
    b: int = bs.B_DEFAULT,
    seed: int = bs.SEED_DEFAULT,
) -> EvalResult:
    """Evaluate every policy on ``target`` (train for dev, test for final)."""
    records = target.records
    outs, sels, notes = policy_outcomes(
        train.records, records, mode=mode, selection_margin=margin / 2.0
    )  # the plan's rule: select at half the evaluation margin
    w_task = bs.task_weights(len(records), b, seed)
    w_repo = bs.cluster_weights([r.repo for r in records], b, seed)
    metrics = [mt.policy_metrics(n, o, w_task, w_repo) for n, o in outs.items()]
    base, orc = outs[mt.BASELINE], outs[mt.ORACLE]
    comparisons = [
        mt.compare(n, o, base, orc, w_task, w_repo, margin)
        for n, o in outs.items()
        if n != mt.BASELINE
    ]
    ablation: list[mt.AblationRow] = []
    noise: list[mt.NoiseRow] = []
    if mode == "dev":
        mid = target.arm_ids["mid"]
        default_mid = {t: r for (t, a), r in target.default_rows.items() if a == mid}
        if target.ablation_rows:
            ablation = mt.ablation_analysis(default_mid, target.ablation_rows, b=b, seed=seed)
        arms = [target.arm_ids["cheap"], mid]
        noise = mt.noise_estimate(target.default_rows, target.repeat_rows, arms, b=b, seed=seed)
    summary = {
        fs: {
            "chosen": None if s.chosen is None else asdict(s.chosen),
            "always_max_resolve_rate_in_cv": s.max_resolve_rate,
            "selection_margin": s.margin,
            "n_candidates": len(s.candidates),
            "n_feasible": sum(c.feasible for c in s.candidates),
        }
        for fs, s in sels.items()
    }
    return EvalResult(
        mode=mode, stamp=target.provenance.stamp(), provenance=target.provenance.as_dict(),
        n_tasks=len(records), n_repos=len({r.repo for r in records}), margin=margin,
        margin_basis=margin_basis, resamples=b, seed=seed, metrics=metrics, comparisons=comparisons,
        per_repo=mt.per_repo(outs, records), verifier=mt.verifier_errors(records),
        ablation=ablation, noise=noise, selections=summary, notes=notes,
        pareto=_pareto(metrics, sels),
    )  # fmt: skip
