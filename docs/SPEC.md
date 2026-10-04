## Repo layout
cascade/taskgen (mine, filter, build_env, validate, split, audit) | harness (sandbox, ledger, providers/{openai,mock}) | agents (triager, reproducer, localizer, patcher, reviewer, verifier, pipeline) | routing (arms, matrix, features, policies) | evals (metrics, plots, report) | pilot (draft)
configs/{models,repos,budget}.yaml | tasks/{manifest,audit}.jsonl | tests/fixtures

## Task schema
task_id, repo, base_commit, issue_title, issue_body, gold_patch, test_patch, fail_to_pass, pass_to_pass, image, features{src_files, src_lines, issue_chars, test_runtime_s}, split, audit{status, notes}

## Task pipeline ($0, no LLM)
Repos: Python+pytest, permissive license, active, mid-popularity, installs without GPU, tests under ~2 min. PRs via GraphQL closingIssuesReferences, merged inside a configurable date window (after the newest arm's knowledge cutoff), touching source and tests, 1-5 source files, ~150 source lines or fewer, no bot/docs-only/dependency PRs. Validation: at base+test patch the failing tests are fail_to_pass; with the source patch all fail_to_pass pass and pass_to_pass still pass; run 3x, drop flaky tests, drop tasks over the runtime cap; record attrition per stage. Split holds out whole repos, frozen by hash. Only audited-ok tasks enter experiments.

## Agents
Roles: Triager, Reproducer (writes a failing test), Localizer, Patcher, Reviewer (separate critic), Verifier (no LLM: executes tests). Pipeline: triage -> reproduce -> localize -> patch -> review -> verify, with a bounded repair loop and per-attempt token and dollar caps. Reproducer and Reviewer are toggleable (for ablations). An arm = (model, reasoning effort) applied to the whole pipeline. Agents see only the issue text and the repo at the base commit.
Two outcomes per attempt: true_success (hidden fail_to_pass and pass_to_pass pass) and verifier_claimed_success (agent's reproducer fails-then-passes and visible tests don't regress).

## Ledger
SQLite record of every call: input/output/reasoning tokens, cost from models.yaml prices, latency. Global, per-task, per-attempt caps from budget.yaml, plus a kill-switch file.

## Outcome matrix row
task_id, arm_id, seed, true_success, verifier_claimed, cost_usd, input_tokens, output_tokens, reasoning_tokens, wall_s, turns, patch_sha, pipeline_config, error

## Policies (offline lookup in the matrix)
fixed(arm) incl. always-cheapest/always-max; cascade(order): cheapest-first, escalate on verifier-claimed failure, cost = sum of arms tried, outcome = last arm's true outcome (document the independence assumption); predictive: per-arm success/cost models from triage features, pick the arm maximizing p*value - cost; oracle: best arm per task. Hyperparameters via leave-one-repo-out CV on train only; one final evaluation on the frozen test split.

## Metrics
Resolve rate, cost per resolved issue, total cost, median latency with bootstrap CIs; paired comparisons between policies on the same tasks (headline); cost reduction vs always-max at equal-or-better resolve rate; share of oracle savings captured; Verifier FP/FN; per-repo breakdown; ablations.
