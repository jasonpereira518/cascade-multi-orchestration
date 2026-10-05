"""Routing policies, all evaluated by OFFLINE LOOKUP in the outcome matrix.

The matrix holds, for every task, what each arm did when run alone on it: ``true_success`` (graded
against hidden tests), ``verifier_claimed`` (the pipeline's own verdict), cost, and wall time. A
policy is then just a rule for which cells it would have used, and its cost, success, and latency
are read off those cells. No model is called here.

Policies:
- ``fixed(arm)``: always that arm (always-cheap, always-mid, always-max).
- ``cascade``: try cheap, then mid, then max, escalating while ``verifier_claimed`` is false.
  Cost is the SUM of the costs of the arms tried, latency the sum of their wall times (they run in
  sequence), and the outcome is ``true_success`` of the LAST arm tried, so a Verifier false positive
  (claimed but wrong) stops the cascade early and counts as a failure.
- ``oracle``: with hindsight, the cheapest arm (by this task's observed cost) whose ``true_success``
  holds; if none succeeds, the cheapest arm by observed cost. It is an unachievable bound.
- ``predictive`` (Tier 2): per-arm success and cost models on triage features, choosing the arm that
  maximizes ``p_success * value - cost``. Hyperparameters and the value are selected by
  leave-one-repo-out CV on train only, by the rule fixed in docs/EVAL_PLAN.md.

INDEPENDENCE ASSUMPTION (cascade, and any policy that combines arms): each matrix cell is one
independent run of that arm alone. The cascade lookup assumes a later arm would have done exactly
what its stand-alone cell did, with no information from the earlier arm's failed attempt (a real
cascade could pass the failed patch or the Verifier's findings forward, which could help or hurt)
and that a single sampled outcome per cell is the arm's outcome (the repeat-subset noise estimate
quantifies how far that is from true). Costs are therefore the cost of RUNNING each tried arm on its
own; nothing is shared or cached between them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cascade.errors import CascadeError
from cascade.routing.features import FEATURE_SETS, feature_matrix
from cascade.routing.learn import LogisticModel, Matrix, RidgeLogCost
from cascade.taskgen.schema import Task

ARMS = ("cheap", "mid", "max")
MAX_ARM = len(ARMS) - 1

# Predictive-router grids. Fixed in docs/EVAL_PLAN.md; changing them changes the plan.
CS = (0.01, 0.03, 0.1, 0.3, 1.0)
VALUE_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
RIDGE_LAMBDA = 10.0
_EPS = 1e-12


class PolicyError(CascadeError):
    """A policy cannot be built or evaluated from the data given."""


@dataclass(frozen=True)
class Cell:
    success: bool  # true_success (hidden tests); already False for cap-exceeded attempts
    claimed: bool  # verifier_claimed
    cost: float
    wall_s: float


@dataclass(frozen=True)
class TaskRecord:
    task: Task
    cells: tuple[Cell, Cell, Cell]  # cheap, mid, max

    @property
    def repo(self) -> str:
        return self.task.repo


@dataclass(frozen=True)
class Outcome:
    cost: float
    success: bool
    latency: float
    arms_tried: int
    arm: int  # index of the arm whose result is the policy's outcome


Policy = Callable[[TaskRecord], Outcome]


# --- the three lookup policies ------------------------------------------------------------------


def arm_index(label: str) -> int:
    if label not in ARMS:
        raise PolicyError(f"unknown arm {label!r}; expected one of {ARMS}")
    return ARMS.index(label)


def fixed(label: str) -> Policy:
    arm = arm_index(label)

    def policy(rec: TaskRecord) -> Outcome:
        c = rec.cells[arm]
        return Outcome(c.cost, c.success, c.wall_s, 1, arm)

    return policy


def cascade(rec: TaskRecord) -> Outcome:
    cost = latency = 0.0
    for i, c in enumerate(rec.cells):
        cost += c.cost
        latency += c.wall_s
        if c.claimed or i == MAX_ARM:
            return Outcome(cost, c.success, latency, i + 1, i)
    raise AssertionError("unreachable: the last arm always ends the cascade")


def oracle(rec: TaskRecord) -> Outcome:
    order = sorted(range(len(rec.cells)), key=lambda i: (rec.cells[i].cost, i))
    winners = [i for i in order if rec.cells[i].success]
    arm = winners[0] if winners else order[0]
    c = rec.cells[arm]
    return Outcome(c.cost, c.success, c.wall_s, 1, arm)


def evaluate(policy: Policy, records: Sequence[TaskRecord]) -> list[Outcome]:
    return [policy(r) for r in records]


def arrays(records: Sequence[TaskRecord]) -> tuple[NDArray[np.bool_], Matrix, Matrix]:
    """(success, cost, wall_s), each (n_tasks, n_arms), for vectorized lookups."""
    shape = (len(records), len(ARMS))
    s = np.array([[c.success for c in r.cells] for r in records], dtype=np.bool_).reshape(shape)
    k = np.array([[c.cost for c in r.cells] for r in records], dtype=np.float64).reshape(shape)
    w = np.array([[c.wall_s for c in r.cells] for r in records], dtype=np.float64).reshape(shape)
    return s, k, w


def outcomes_of_choices(records: Sequence[TaskRecord], choices: Sequence[int]) -> list[Outcome]:
    """Outcomes of a single-arm-per-task policy (the predictive router and fixed policies)."""
    out: list[Outcome] = []
    for r, a in zip(records, choices, strict=True):
        c = r.cells[a]
        out.append(Outcome(c.cost, c.success, c.wall_s, 1, a))
    return out


# --- the predictive router ----------------------------------------------------------------------


def choose_arms(p: Matrix, cost_hat: Matrix, value: Matrix) -> list[int]:
    """argmax over arms of p * value - cost_hat; ties go to the cheaper (lower) arm."""
    return [int(i) for i in np.argmax(p * value[:, None] - cost_hat, axis=1)]


@dataclass(frozen=True)
class PredictiveModel:
    feature_set: str
    success: tuple[LogisticModel, ...]
    cost: tuple[RidgeLogCost, ...]
    value_scale: float  # mean max-arm cost on the training tasks; value = multiplier * this

    @classmethod
    def fit(cls, records: Sequence[TaskRecord], feature_set: str, c: float) -> PredictiveModel:
        if not records:
            raise PolicyError("cannot fit a router on no tasks")
        x = feature_matrix([r.task for r in records], FEATURE_SETS[feature_set])
        s, k, _ = arrays(records)
        success = tuple(LogisticModel.fit(x, s[:, a], c) for a in range(len(ARMS)))
        cost = tuple(RidgeLogCost.fit(x, k[:, a], RIDGE_LAMBDA) for a in range(len(ARMS)))
        return cls(feature_set, success, cost, float(k[:, MAX_ARM].mean()))

    def predict(self, records: Sequence[TaskRecord]) -> tuple[Matrix, Matrix]:
        x = feature_matrix([r.task for r in records], FEATURE_SETS[self.feature_set])
        p = np.column_stack([m.predict_proba(x) for m in self.success])
        c = np.column_stack([m.predict(x) for m in self.cost])
        return p, c

    def choose(self, records: Sequence[TaskRecord], multiplier: float) -> list[int]:
        p, c = self.predict(records)
        value = np.full(len(records), multiplier * self.value_scale)
        return choose_arms(p, c, value)


@dataclass(frozen=True)
class OutOfFold:
    p: Matrix
    cost_hat: Matrix
    value_scale: Matrix  # per task: the training fold's value scale for the fold it was held out in


def repos_of(records: Sequence[TaskRecord]) -> list[str]:
    return sorted({r.repo for r in records})


def loro_predictions(records: Sequence[TaskRecord], feature_set: str, c: float) -> OutOfFold:
    """Leave-one-repo-out predictions: every task is predicted by models fitted without its repo."""
    repos = repos_of(records)
    if len(repos) < 2:
        raise PolicyError("leave-one-repo-out CV needs tasks from at least two repos")
    n = len(records)
    p = np.zeros((n, len(ARMS)))
    ch = np.zeros((n, len(ARMS)))
    vs = np.zeros(n)
    for repo in repos:
        held = [i for i, r in enumerate(records) if r.repo == repo]
        rest = [r for r in records if r.repo != repo]
        model = PredictiveModel.fit(rest, feature_set, c)
        pp, cc = model.predict([records[i] for i in held])
        p[held], ch[held], vs[held] = pp, cc, model.value_scale
    return OutOfFold(p, ch, vs)


@dataclass(frozen=True)
class Candidate:
    feature_set: str
    c: float
    multiplier: float
    resolve_rate: float
    mean_cost: float
    feasible: bool


@dataclass(frozen=True)
class Selection:
    candidates: list[Candidate]
    chosen: Candidate | None  # None: no candidate was feasible, so the router is always-max
    max_resolve_rate: float
    max_mean_cost: float
    margin: float


def select_config(
    records: Sequence[TaskRecord],
    feature_set: str,
    margin: float,
    *,
    cs: Sequence[float] = CS,
    multipliers: Sequence[float] = VALUE_MULTIPLIERS,
) -> Selection:
    """The selection rule of docs/EVAL_PLAN.md, applied to leave-one-repo-out predictions.

    Among candidates (C, value multiplier) whose pooled out-of-fold resolve rate is at least the
    always-max resolve rate minus ``margin``, choose the one with the lowest mean cost; ties go to
    the smaller C, then the smaller multiplier. If none qualifies the router is always-max.
    """
    s, k, _ = arrays(records)
    rows = np.arange(len(records))
    rr_max = float(s[:, MAX_ARM].mean())
    cost_max = float(k[:, MAX_ARM].mean())
    cands: list[Candidate] = []
    for c in cs:
        oof = loro_predictions(records, feature_set, c)
        for m in multipliers:
            choice = np.array(choose_arms(oof.p, oof.cost_hat, m * oof.value_scale))
            rr = float(s[rows, choice].mean())
            mc = float(k[rows, choice].mean())
            cands.append(Candidate(feature_set, c, m, rr, mc, rr >= rr_max - margin - _EPS))
    feasible = [x for x in cands if x.feasible]
    chosen = (
        min(feasible, key=lambda x: (round(x.mean_cost, 12), x.c, x.multiplier))
        if feasible
        else None
    )
    return Selection(cands, chosen, rr_max, cost_max, margin)


def nested_choices(
    records: Sequence[TaskRecord], feature_set: str, margin: float
) -> list[int] | None:
    """Honest out-of-sample choices for the dev report: nested leave-one-repo-out.

    For each outer repo, the (C, multiplier) is selected by inner leave-one-repo-out CV on the OTHER
    repos only, the router is refit on them, and it chooses arms for the held-out repo. Reporting
    the single-level CV of the selected candidate would be optimistic (it was selected on those
    same folds). Needs at least three repos; otherwise returns None.
    """
    repos = repos_of(records)
    if len(repos) < 3:
        return None
    choices = [MAX_ARM] * len(records)
    for repo in repos:
        held = [i for i, r in enumerate(records) if r.repo == repo]
        rest = [r for r in records if r.repo != repo]
        sel = select_config(rest, feature_set, margin)
        if sel.chosen is None:
            continue  # fallback: always-max for this repo
        model = PredictiveModel.fit(rest, feature_set, sel.chosen.c)
        picked = model.choose([records[i] for i in held], sel.chosen.multiplier)
        for i, a in zip(held, picked, strict=True):
            choices[i] = a
    return choices


@dataclass(frozen=True)
class FinalRouter:
    """A router fitted on all training tasks with the selected configuration (or always-max)."""

    feature_set: str
    selection: Selection
    model: PredictiveModel | None  # None: always-max

    def choose(self, records: Sequence[TaskRecord]) -> list[int]:
        if self.model is None or self.selection.chosen is None:
            return [MAX_ARM] * len(records)
        return self.model.choose(records, self.selection.chosen.multiplier)


def fit_final_router(records: Sequence[TaskRecord], feature_set: str, margin: float) -> FinalRouter:
    sel = select_config(records, feature_set, margin)
    if sel.chosen is None:
        return FinalRouter(feature_set, sel, None)
    return FinalRouter(feature_set, sel, PredictiveModel.fit(records, feature_set, sel.chosen.c))
