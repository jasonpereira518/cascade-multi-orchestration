# Evaluation plan

**Status: PROPOSED, awaiting the human's confirmation of section 5 (the non-inferiority margin).**
This file is committed before anything touches the test split. `cascade eval --final` refuses to run
unless it is committed and the working tree is clean, and it records this file's git blob hash in
`FINAL.lock.json`. Anything not decided here is not decided; a change after the final run is a
recorded deviation, not an edit.

## 1. Question and claim

Thesis (CLAUDE.md): a cost-aware router can match an always-max-effort baseline's resolve rate at
materially lower cost per resolved issue. **A null result is acceptable. A fabricated or leaky one is
not.** The evaluation is built to report a null or an inconclusive result as plainly as a positive one.

## 2. Data, provenance, and what counts as a row

- **Real vs synthetic.** Real data is the Phase 4 export (`cascade export`): a matrix whose ledger
  shows only `openai` calls. The labeled-synthetic matrix lives only under `tests/synthetic/` and
  carries a `SYNTHETIC.json` marker. They are never combined, a mock-provider matrix is refused as real,
  and `--final` refuses synthetic data.
- **Stamp.** Every output (report, JSON, figure) carries: data source, git commit (and whether the
  tree was dirty), SHA-256 of `matrix.parquet`, processing tier, and the splits used.
- **One tier.** Every row in the matrix file must share one `processing_tier`. Mixed tiers are refused
  (nothing is evaluated), because costs on different tiers are not comparable.
- **Default-pipeline rows only** feed policy fitting and evaluation: seed 0, the full pipeline
  (reproducer and reviewer on). Ablation rows and repeat-seed rows are excluded from policies and used
  only in sections 7.3 and 7.4. If more than one distinct default pipeline config exists, the load is
  refused as ambiguous.
- **Cap-exceeded cells are failures.** An attempt that hit a per-attempt cap counts as a failed cell
  (`true_success` false and `verifier_claimed` false, so a cascade escalates past it) with the cost it
  incurred, even if its partial patch passed the hidden tests. The number of cells so treated is reported.
- **Complete tasks only.** A task needs a default row for all three arms (cheap, mid, max). Others are
  excluded and counted in the report.
- **Splits.** The split is the frozen repo-level split (`tasks/split.json`). Dev runs load TRAIN rows
  only (rows of other splits are filtered out at read time and never materialized). The final run loads
  train for fitting and test for the one evaluation. Only audited-ok tasks are used.

## 3. Policies (all evaluated by offline lookup in the matrix)

Arms: cheap = gpt-6-luna/low, mid = gpt-6-luna/high, max = gpt-6.1-sol/medium. Cascade order is cheap, mid, max.

- **fixed(arm):** always-cheap, always-mid, always-max (always-max is the baseline).
- **cascade:** try cheap, then mid, then max; escalate while `verifier_claimed` is false. Cost is the
  sum of the costs of the arms tried; latency is the sum of their wall times. The outcome is
  `true_success` of the last arm tried, so a Verifier false positive (claimed but wrong) stops the
  cascade and counts as a failure.
- **oracle:** with hindsight, the cheapest arm (by this task's observed cost) with `true_success`; if
  none succeeds, the cheapest arm by observed cost. Unachievable; used only as a bound.
- **predictive:** section 4.

**Independence assumption (cascade and any combination of arms).** Each matrix cell is one independent
run of that arm alone. The lookup assumes a later arm would have done exactly what its stand-alone
cell did, with no information from an earlier arm's failed attempt (a real cascade could forward the
failed patch or the Verifier's findings, which could help or hurt), and that one sampled outcome per
cell is the arm's outcome. The repeat-subset flip rate (7.4) measures how far that last assumption is
from true. The cascade's cost is therefore the cost of running each tried arm separately.

## 4. The predictive router (Tier 2)

- **Models.** Per arm: an L2-regularized logistic regression for P(true success) and a ridge
  regression of log cost with a smearing correction (predicted mean cost), on standardized features.
  Implemented in plain numpy, deterministic.
- **Decision.** For each task pick the arm maximizing `p_success x value - predicted_cost`; ties go to
  the cheaper arm. `value` is a multiple of the training tasks' mean always-max cost.
- **Two feature sets, reported separately.** *issue* (primary): features computable from the issue text
  before any attempt. *all*: issue plus features measured on the gold patch or the test run
  (`src_files`, `src_lines`, `test_runtime_s`, hunks, fail_to_pass count, symbols the fix touches). The
  SPEC lists some of these as triage features, but a deployed router would not have them, so a router
  using them is an **upper bound, not a deployable router**. `predictive_issue` is the one that can
  support a deployability claim; `predictive_all` is context.
- **Grids (fixed).** C in {0.01, 0.03, 0.1, 0.3, 1.0}; value multiplier in {0.25, 0.5, 1, 2, 4, 8, 16,
  32, 64, 128}; ridge lambda 10 (not tuned).
- **Selection rule (fixed).** On TRAIN only, by leave-one-repo-out CV (every task predicted by models
  fitted without its repo; predictions pooled), compute for each (C, multiplier) the pooled
  out-of-fold resolve rate and mean cost. A candidate is *feasible* if its resolve rate is at least the
  always-max resolve rate on the same tasks minus **delta/2** (half the margin of section 5). Choose the
  feasible candidate with the lowest mean cost; ties go to the smaller C, then the smaller multiplier.
  If no candidate is feasible the router is always-max (and is reported as such). The selection is made
  per feature set. The test split plays no part in it.
- **Dev honesty.** Dev reports use NESTED leave-one-repo-out CV (the configuration for each outer repo
  is selected on the other repos only), because reporting the CV of a configuration chosen on those
  same folds would flatter it. Nested CV needs at least 3 repos in train; with fewer the report says
  so and shows the optimistic single-level number instead.
- **Final.** The router is fit on all TRAIN tasks with the selected configuration and applied once to
  the test tasks.

## 5. Headline and the non-inferiority margin (FOR THE HUMAN TO CONFIRM)

**Headline: cost reduction versus always-max at equal-or-better resolve rate.** Made precise:

- *resolve-rate difference* = resolve rate of the policy minus resolve rate of always-max, paired on the
  same tasks;
- *cost reduction* = 1 - (mean cost per task of the policy) / (mean cost per task of always-max);
- **non-inferior** iff the lower bound of the 95% bootstrap CI of the resolve-rate difference is above
  **-delta**;
- **saves** iff the lower bound of the 95% bootstrap CI of the cost reduction is above 0;
- **verdict MET** iff non-inferior and saves; **INFERIOR** iff the whole CI of the resolve-rate
  difference is below -delta; otherwise **INCONCLUSIVE**.

Two policies are primary and each is judged on its own: `cascade` and `predictive_issue`. The other
rows (fixed arms, oracle, `predictive_all`) are descriptive. No multiplicity correction is applied; with
two primary comparisons, a MET on one of them is weaker evidence than the intervals alone suggest, and
the write-up must say so.

**Proposed margin rule.** delta = the smallest of {0.05, 0.10, 0.15} for which the power to show
non-inferiority is at least 0.5 when the policy is exactly as good as always-max, capped at 0.15. The
number of audited test tasks `n` is read from the manifest when the lock is created (a count, not an
outcome), so the margin is fixed by the data size and not chosen after seeing results.

Power is `Phi(delta / SE - 1.96)` with `SE = sqrt(d / n)`, where `d` is the share of tasks on which the
policy and always-max outcomes differ. **`d = 0.20` is an assumption, not a measurement**; the dev
analysis estimates it, but the rule uses the fixed 0.20 so it cannot be tuned. Power under that
assumption:

| test tasks n | delta 0.05 | delta 0.10 | delta 0.15 | delta 0.20 | rule gives |
|---:|---:|---:|---:|---:|---:|
| 13 | 0.06 | 0.12 | 0.23 | 0.36 | 0.15 |
| 20 | 0.07 | 0.17 | 0.32 | 0.52 | 0.15 |
| 25 | 0.08 | 0.20 | 0.39 | 0.61 | 0.15 |
| 30 | 0.09 | 0.23 | 0.45 | 0.69 | 0.15 |
| 40 | 0.11 | 0.29 | 0.56 | 0.81 | 0.15 |
| 60 | 0.14 | 0.41 | 0.74 | 0.93 | 0.15 |
| 100 | 0.20 | 0.61 | 0.92 | 0.99 | 0.10 |

**Why 0.15, and what it costs.** A margin is a statement of how much resolve rate you would give up
for the saving. 0.15 (15 percentage points) is already generous; the cap exists so the margin is never
loosened to manufacture a pass. At the test-split size this project is likely to have (the repo-level
split points to roughly 13 to 30 tasks), even a router exactly as good as always-max has only about a
23% to 45% chance of being shown non-inferior at 0.15. **The headline is therefore expected to be
INCONCLUSIVE for many true situations, and that outcome will be reported as inconclusive, not retried
and not re-margined.** Smaller margins (0.05, 0.10) are not usable at this size; they would be
INCONCLUSIVE almost regardless of the truth.

*Please confirm or change this rule before the final run.* If you change it, change
`MARGIN_CANDIDATES`, `MARGIN_CAP`, `ASSUMED_DISCORDANCE`, or `POWER_TARGET` in
`cascade/evals/bootstrap.py` to match, update this table, and commit both together. A test checks that
this table and the code agree.

**Oracle-savings share** (how much of the achievable saving a policy captures):

`share = (C_always_max - C_policy) / (C_always_max - C_oracle)`

where C is mean cost per task. It is 1 for the oracle and 0 for always-max; it can exceed 1 for a policy
that is cheaper than the oracle only by failing more often, and it is undefined (reported as such) when
the denominator is about zero. It is reported with its bootstrap CI and is not a headline.

## 6. Metrics and intervals

- **Resolve rate, cost per task and total cost, cost per resolved issue (total cost / number resolved),
  median latency (a cascade's latency is the sum over arms tried).**
- **Bootstrap:** 95% percentile intervals over tasks, 10,000 resamples, seed 4242. Comparisons between
  policies are **paired**: both policies are evaluated on the same resamples. Resamples in which a
  statistic is undefined (cost per resolved when nothing was resolved) are excluded and counted.
- **Sensitivity:** a repo-clustered bootstrap (resample whole repos with replacement). With fewer than
  about 5 repos in the evaluated split it is unreliable and is shown only as a check.

## 7. Secondary analyses (dev split only)

1. Pareto plot of cost per task against resolve rate, with whiskers (SVG).
2. Per-repo breakdown (point estimates; a repo has too few tasks for an interval).
3. Ablations (mid arm): default vs no_reproducer vs no_reviewer, paired on the tasks that have all
   three, with bootstrap CIs on the differences. Train tasks only.
4. Noise: the share of repeat-subset cells (cheap and mid, a second seed) whose `true_success` differs
   from the base seed, with a bootstrap CI. It bounds how seriously a single sampled cell should be taken.
5. Verifier false-positive rate P(claimed | failed) and false-negative rate P(not claimed | passed) per
   arm, with Wilson intervals.

## 8. Test-split confidence intervals will be wide

The test split is whole repos held out by hash, so it is small: likely about 13 to 30 tasks from one to
three repos. Intervals on resolve rate will be tens of percentage points wide, intervals on differences
between policies nearly as wide, and the repo-clustered intervals wider still. **Differences smaller than
the intervals are not distinguishable from noise, and a point estimate in the right direction is not a
finding.** The report prints this warning whenever there are fewer than 40 tasks.

## 9. The final run

`cascade eval --final`, exactly once: (1) this file is committed and the working tree is clean; (2) no
`FINAL.lock.json` exists in the tree or anywhere in git history; (3) the data is real, one tier, and
has train and test tasks, else it refuses before starting; (4) it writes the lock (commit, this file's
blob hash, matrix hash, margin, n) before evaluating; (5) it fits on train only, evaluates the test
tasks once, writes `results/final/{report.md, results.json, pareto.svg}`, and records their hashes in
the lock. If evaluation fails after the lock is written the single run is used up and is not retried
automatically. Development uses `cascade eval` without `--final`, which never loads a test row.

## 10. Limits stated in advance

- The cascade lookup rests on the independence assumption of section 3.
- One sampled outcome per cell (plus a small repeat subset); true per-arm success probabilities are
  unknown.
- The gold-derived features make `predictive_all` an upper bound; the primary predictive claim uses
  issue-side features only.
- A repo-level split is coarse: results may reflect which repos landed in test.
- The tasks come from a handful of Python repos and from after the arms' knowledge cutoffs; nothing
  here says how a router would do on other languages, larger tasks, or other repos.
- The assumed discordance in the margin rule is a planning number, not a finding.
