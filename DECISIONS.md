# Decisions

Non-obvious choices, newest last. Each says what was chosen, why, and what would change it.

## Foundation (STOP 1)

1. **Python 3.11 pinned in `.python-version`.** The spec floor is 3.11 and CI should test the
   floor. The dev machine has 3.14 as its default; uv supplies 3.11. `tarfile` extraction filters
   need 3.11.4+, which uv's 3.11.15 satisfies.
2. **Only four runtime deps for now: pydantic, pyyaml, typer (plus dev tools).** The OpenAI Agents
   SDK, the OpenAI SDK, pandas, and pyarrow are not installed yet. Rule 8 says to check an API
   against the installed version, and nothing uses them yet. Pin each when its first consumer lands.
3. **Unfilled config fields are the literal string `TODO`, and everything needing a real value
   fails closed.** Configs load with TODOs so `cascade config-status` can list them. `prices()` and
   `BudgetConfig.resolve()` raise `ConfigNotFilledError`, and `Ledger` only accepts a resolved
   budget, so no run can start on unfilled config. `TODO-<hint>` is also treated as a placeholder
   where names must stay distinct (the three template arms).
4. **Token usage is non-overlapping: `output_tokens` excludes `reasoning_tokens`.** Some provider
   APIs report reasoning as a subset of output. The adapter must subtract it so the ledger never
   double-bills. Both are priced at the output price unless `reasoning_usd_per_mtok` is set.
5. **The ledger authorizes on worst-case figures and records actuals.** Pre-call check uses
   `est_input_tokens * input_price + max_output_tokens * max(output_price, reasoning_price)`, with
   input estimated pessimistically at 3 chars/token. A call that passes cannot breach a cap by
   itself. Known limit: authorize and record are not one atomic reservation, so parallel workers can
   overshoot by their in-flight calls. Keep parallelism low or leave cap headroom.
6. **A repeated `call_id` is an error, not a no-op.** If a real call were repeated under the same
   id and the ledger silently ignored the second row, spend would be undercounted. Resumability
   (rule 9) therefore lives one layer up: the pipeline persists call outputs and skips calls whose
   id is already in the ledger. `call_id_for()` derives ids deterministically for that purpose.
7. **Caps apply to mock calls too, using synthetic prices.** One code path for mock and real keeps
   the smoke test exercising the same authorize/record logic that real runs use.
8. **The leak guard is layered, and always on.** The workspace comes from `git archive` of the base
   commit, so it has no `.git` and no history at all. `LeakGuard.scan` runs inside every
   `SandboxRunner` call with no opt-out. It flags `.git`, paths the patches add, and whole files
   byte-identical to either patch. Limit: it does not detect an excerpt of a patch pasted into an
   existing file. The defence for that is that nothing but the base-commit tree is ever mounted.
   `git archive` honours `export-ignore` and `export-subst` attributes, so a repo using them could
   differ slightly from a plain checkout; revisit if a chosen repo does.
9. **Grading is a separate code path, not built yet.** Grading legitimately applies the test patch,
   so it must not reuse the agent-side runner behind a flag. It will get its own runner and tests.
10. **Container hardening beyond "network off".** Read-only root, `/tmp` tmpfs, all capabilities
    dropped, `no-new-privileges`, memory/swap/pids/cpu limits, host uid:gid, and an empty
    environment apart from `HOME=/tmp` and an allow-list. Env names that look like secrets
    (KEY/TOKEN/SECRET/PASSWORD/OPENAI/GITHUB/...) are refused outright.
11. **Docker tests are gated by `CASCADE_TEST_DOCKER_IMAGE`.** The real-container test needs a
    daemon and a locally present image, and it must not pull one implicitly. It is skipped otherwise
    and was skipped when this was written (no daemon running); see STATUS.md.
12. **Mock model names in `tests/fixtures/synthetic/models.yaml` do not violate rule 5.** They are
    labeled synthetic, are not real models, and the production `configs/models.yaml` has none.
13. **Ruff runs with the `S` (bandit) rules and pyright is strict on `cascade/`.** The
    `subprocess` calls carry `noqa: S603` with a reason (argv lists, validated input, no shell).
14. **CI actions are pinned by major tag, not SHA.** Versions were read from the GitHub releases
    API when written (`actions/checkout` v7, `astral-sh/setup-uv` v10). SHA pinning is stricter;
    switch if this repo ever handles anything sensitive in CI. CI sets `CASCADE_ALLOW_SPEND=0`.

## Phase 1: task pipeline for pallets/click (STOP A)

15. **A PR's base commit is its merge commit's first parent; the patch is `git diff P M`.** Checked
    against click's real history: most PRs land as true merge commits (344 two-parent merges in
    two years, PR title and `(#N)` in the subject), a few as squash commits, and the many
    single-parent commits without a PR number are branch commits inside merged PRs, never a PR's
    `mergeCommit`. First-parent diff is the full PR effect for both shapes. Rebase-merges are not
    supported and are caught by decision 16.
16. **Patches are cross-checked against GitHub's per-file list and line counts.** Same file set is
    not enough: a rebase-merged multi-commit PR whose last commit touches the same files would
    pass with a partial diff. Mismatches drop the PR (`diff_mismatch`, `diff_counts_mismatch`,
    which also covers renames). This is stricter than needed in rare overlapping-change cases;
    that costs a few tasks, never correctness. Found by reading real history, not by tests first.
17. **Gold patch = every non-test file in the diff, including docs and changelog hunks;** test patch
    = every test file (SWE-bench-style). Size features count source `.py` files only. Classification
    order matters: dependency/CI files before docs (`requirements/*.txt` is not documentation), and
    docs-only is reported before no-tests so it appears as its own exclusion. Both were test-found bugs.
18. **Filters beyond the spec:** exactly one closing issue, in the same repo, with a non-empty body;
    PRs into a non-default branch are dropped (stable-branch backports duplicate mainline fixes).
    The issue text is the issue as it is *now*; edits made after the fix are not detected (an audit
    item). Click's `closingIssuesReferences` includes manually linked issues (GitHub's default).
19. **Test outcomes come from a tiny pytest plugin (`docker/cascade_pytest_report.py`), not `-rA`
    text.** Real `-rA` output omits node ids for skipped tests and parametrized ids contain spaces
    and " - ", so parsing text is ambiguous. The plugin writes exact ids; a malformed report raises
    instead of being skipped. It is located relative to the repo root, so `taskgen` runs from a
    checkout, not an installed wheel.
20. **The repo is installed editable at image build time, pointing at `/workspace`.** Click's own
    tests call `importlib.metadata.version("click")`, so `PYTHONPATH` alone fails (found by running
    its suite natively). The source is deleted from the image afterwards; at test time the task tree
    is bind-mounted at `/workspace`. Caveat: package metadata (e.g. the version string) comes from
    the image's reference commit, so a test comparing it to the task tree can fail identically before
    and after the gold patch; such tests land in neither list.
21. **Validation semantics.** Two configurations (base+tests, base+tests+gold) x 3 fresh-workspace
    runs. A test is flaky if its outcome (including "missing") differs between runs of one
    configuration, and is excluded. fail_to_pass = not passing before (failed, error, or missing, so
    tests in a module that cannot import yet count) and passing after. pass_to_pass = passing both.
    A test passing before and failing after gold drops the whole task (`gold_regresses`, strict on
    purpose). Skipped/xfailed tests belong to neither list. The container timeout equals the runtime
    cap, and any timeout or over-cap run drops the task. Usable pytest exit codes are 0 and 1.
22. **Env stage = run the test files the test patch touches, with gold applied, and require that
    they collect and run.** Cheap and catches missing dependencies per base commit. It is not a
    full collection run.
23. **One image per repo, built at the newest passing candidate's base commit** (`uv` 0.11.2 and
    `debian:bookworm-slim` tags were verified to exist; Python pinned to an exact patch release
    from `uv python list`). The tag hashes the Dockerfile, reference commit, plugin, and test
    command, so identical inputs reuse the image. Tasks older than the reference commit run with
    its dependencies; mismatches show up as env or validation drops, not silently.
24. **State is SQLite; reruns skip final rows and retry `error` rows.** A month slice is marked done
    only once wholly in the past, so the slice containing today is refetched. State is bound to a
    config fingerprint and refuses to mix configurations. Reruns never rebuild an existing image.
25. **`github.py` carries a local `# pyright: basic`.** It parses untyped external JSON, and every
    value is validated into pydantic models immediately; strict mode there only adds casts. The rest
    of `cascade/` stays strict.
26. **No GITHUB_TOKEN was available, and I did not borrow the `gh` CLI's token.** The brief says to
    use `GITHUB_TOKEN`; that credential was not provided, and `gh`'s OAuth token has wider scope than
    a read-only miner needs. The client reads only the env var. CI fixtures are therefore synthetic
    (labeled, shaped from GitHub's published schema); `GitHubClient(record_dir=...)` captures real
    responses into `tests/fixtures/recorded/` for later.
27. **The date window was left `TODO` on purpose.** It must start after the newest arm's knowledge
    cutoff, which comes from `models.yaml`, which is still unfilled. Repo-specific fields for click
    (license, Python 3.12.13, install and test commands) were verified and filled.
28. **A grading runner exists now, separate from the agent runner** (closes the deferral in #9). It
    shares every isolation flag via `ContainerRunner` and differs only in what it will mount: it
    skips the leak guard because validation workspaces legitimately contain the patches. A test
    asserts the two produce identical commands for a clean workspace.

## Phase 2: agent pipeline and walking skeleton (STOP B)

29. **Agents SDK integration is a model-layer wrapper, and it replaced the Phase 0 provider layer.**
    The SDK drives the agent loop (tools, structured output, turns); `MeteredModel` wraps any SDK
    `Model` and runs ledger authorize -> call -> record around every call, so no role, and no turn of
    a tool loop, can bypass the caps. The old `Provider`/`MockProvider`/`MeteredClient` layer only
    did single completions and would have duplicated the SDK, so it was removed rather than kept
    in parallel. Its guarantees (spend gate, kill switch, caps, error rows) moved to
    `harness/models.py` with ported tests.
30. **SDK facts verified against openai-agents 0.23.1 / openai 3.24.0 (rule 8), by reading source
    and prototyping:** strict tool schemas make every parameter required and drop defaults, so tools
    take no defaults; `output_tokens` includes reasoning tokens (subtracted for the ledger); usage
    details need `cache_write_tokens` (new in openai 3.x); exceptions raised in the model layer reach
    the caller unwrapped (so budget and kill-switch errors are catchable); `MaxTurnsExceeded` and
    `ModelBehaviorError` are catchable. `ModelSettings.max_tokens`, `reasoning`, `store` exist.
31. **SDK tracing is off on every run, and `store=False`.** Default tracing uploads prompts to a
    third-party backend, and the Responses API retains responses server-side by default. Local JSONL
    traces replace the first; the second is just disabled.
32. **Cached input tokens are billed as ordinary input.** models.yaml has no cached-token price, so
    cost can only be overstated. Failed calls are logged as zero-cost error rows; if a provider
    bills a failed request, the ledger would undercount it (unverified; watch the first paid run).
33. **The mock is the SDK's own `ScriptedModel`, one instance per role,** so a plan can span repair
    rounds. With no plan it answers any output schema with a minimal valid instance and calls no
    tools, which makes `cascade run` work offline and fail honestly (empty patch).
34. **The graded patch is source-only: every non-test file the agent changed.** Test files, including
    the reproducer, are excluded because hidden tests are applied on top and edits to tests would
    conflict. Patches are built with `git diff --no-index --no-prefix` on filtered copies and
    header-normalized. Both `--no-prefix` and the normalization were found by tests: plain
    `--no-index a b` emits `a/a/<path>` (inapplicable) and `b/x b/x` headers for added files.
35. **verifier_claimed definition:** reproducer fails before the patch and passes after, AND no test of
    the visible suite regresses. A regression must reproduce on a second run before it counts (one
    flaky test cannot turn a good patch into a "failure"). An empty patch is never claimed and runs no
    suite. With the Reproducer disabled the first condition is dropped, a weaker claim that shows up
    as such in the pipeline config of the matrix row.
36. **The pipeline verifies the reproducer itself.** After the Reproducer role it runs the file on the
    untouched code and requires a genuine `failed` test (an import or collection error does not
    count). The agent's own statement that it fails is not trusted.
37. **Review happens only after the Verifier is satisfied, and only drives repair.** A reviewer that
    cannot answer does not block a verified patch. Repairs are bounded by `max_repairs`.
38. **Failure semantics.** Cap or kill switch: stop calling models, but still diff the workspace and
    run the Verifier (no LLM) so the row is honest rather than empty. Role-level model failures
    (`AgentsException`, `openai.OpenAIError`) are survivable. The spend gate and genuine bugs
    propagate, so a config mistake or a bug is never recorded as an attempt result.
39. **true_success grading** applies the agent's source patch to a fresh base export, then the hidden
    test patch, runs the full suite once under the grading runner, and requires every fail_to_pass
    and pass_to_pass test to pass. No grading information flows back into any prompt.
40. **Attempt bookkeeping.** attempt_id = arm|seed|config id (+ `|tryN` for retries, which get a
    fresh per-attempt cap while task and global totals stay cumulative). The matrix row is written
    last and keyed by (task, arm, seed, config id), so ablations of the same attempt are separate
    rows and an interrupted attempt leaves none. `wall_s` is the agent pipeline's wall time, not
    grading. Every numeric field is a sum over logged ledger calls.
41. **Tools are pure Python and bounded:** BM25 over 40-line chunks plus regex grep (no ripgrep: no
    external dependency, fully deterministic), line and character caps, no default parameters,
    dotfiles and `.git` unreachable, symlinks resolving outside the root refused.
42. **The leak audit decodes nested JSON and exempts one check.** Trace lines nest JSON several
    levels, so strings are decoded recursively rather than guessing escape levels (a bug found by a
    planted-leak test). Hidden test *names* are those the test patch adds, not existing visible ids
    (agents legitimately see those in test output). The whole-gold-patch check is skipped only on
    reviewer calls, whose prompt is the agent's own patch; a one-line fix can equal the gold patch
    byte for byte. A hit means "investigate".
43. **`NativeExecutor` exists only in tests.** It runs pytest on the host so the pipeline can be tested
    without Docker, but it still calls the real runner's `validate` so the leak guard runs. The CLI
    only ever wires the Docker executor.
44. **Docker was verified for real this phase.** The sandbox test passes against a daemon (the first
    version asserted "only lo" and was wrong: `--network none` still lists inert tunnel stubs; the
    check is now "no route"). The click image builds and runs the suite offline. The first run showed
    24 pager-test failures because the slim base image lacks `less`, hence per-repo `system_packages`.
45. **The real OpenAI path has never been called.** The source builds `OpenAIResponsesModel` (verified
    to construct with no network) but no request has been made; there is no key and no spend
    approval. Likely first-run issues are listed in NEXT.md.
46. **The Phase 2 acceptance ran on the synthetic fixture task,** because no validated task exists
    (the Phase 1 live run is still pending a token). A separate exploratory run of the real CLI
    through real Docker on a real click PR (scratch task, placeholder issue text) confirmed the
    plumbing; see STATUS.md.
47. **E501 is relaxed for `tests/` only** (long ids and literals read better unwrapped); production
    code stays at 100 columns.

## Additional repos (human's picks: jinja, attrs, structlog, marshmallow)

48. **All four were built and run for real before being listed.** Each image was built through the
    real `build_image`, and the suite was run three times offline with the network off. Results:
    jinja 911 tests, attrs 1401, structlog 884, marshmallow 1190; all passing, 1.4 to 6 s. Eight real
    issue-linked PRs from attrs, structlog, and marshmallow were then validated end to end in Docker
    (3x pre/post): 8 of 8 validated. Jinja was not validated per PR because it yields none (49).
49. **Jinja is dormant and is listed only for completeness.** Its last commit to `main` is
    2025-06-14, so it has no PR landings in the last 16 months and fails the spec's "active"
    criterion. Config is verified but the run instructions skip it.
50. **`github_license_id` for dual-licensed repos.** GitHub's API reports structlog (MIT OR Apache-2.0)
    as `NOASSERTION` (verified), so the mine-time license check would have rejected it. The optional
    field says what GitHub should report; the check still runs and still fails on any other value.
51. **Yield estimates now combine two independent sources.** PRs that pass my git-checkable filters
    (source and test files, size limits, not docs/dependency/branch-sync) are intersected with GitHub's
    own `linked:issue` search, run through the GitHub MCP tool as read-only queries on public
    repos. This replaced the commit-message proxy, which badly underestimated (click: 18 by proxy,
    73 linked per GitHub). The pipeline itself still reads only `GITHUB_TOKEN`; the MCP queries were
    exploratory. `linked:issue` is not exactly `closingIssuesReferences`, and the single-issue,
    same-repo, and non-empty-body rules are not applied, so the counts are upper bounds.
52. **Per-repo test commands.** attrs runs serially: its tox config uses `-n auto` (xdist), but
    parallel workers would race on the plugin's single report file. structlog runs with
    `-p no:randomly`: its test group includes pytest-randomly, and a fixed order keeps the 3x flake
    check meaningful. Tests whose parametrize ids embed the current time (six in marshmallow) have
    node ids that differ per run, so the flake check drops them, which is the conservative outcome.
53. **Large fail_to_pass sets mean an import failure, not a big fix.** When a PR's test module cannot
    import before the fix (structlog #761, #812: ~175 tests), every test in it counts as
    fail_to_pass by the "missing before" rule. Grading is equivalent, but those tasks are worth a
    glance during audit, since their F2P size says nothing about difficulty.

## Phase 3 (scale, split, audit)

54. **The split is a hash-ordered prefix of repos, frozen in `tasks/split.json`.** Repos are sorted by
    SHA-256 of `salt:repo` (never Python's `hash()`, which is randomized per process; a test runs the
    choice under three `PYTHONHASHSEED`s). The test set is the shortest prefix whose share of tasks is
    closest to the target fraction (default 0.4), always leaving at least one repo in train. The
    only inputs are repo names, task counts, the salt, and the target, so no outcome can influence it.
    `freeze` refuses to overwrite; `--force` exists and is documented as invalidating results.
55. **Repos that appear after the freeze get a hash bucket, appended, never recomputed.** A late repo
    is test if `bucket(repo, salt) < target`, else train, and is recorded under `late_assignments` with
    a timestamp. Existing assignments never change. `fitting_tasks` returns train tasks only and
    refuses a manifest with any unsplit task, so fitting code cannot be handed unsplit data by mistake.
56. **A repo-level split is coarse, and with this repo set it cannot hit 25-30 test tasks.** Click
    supplies about 60% of the projected tasks. With the upper-bound counts (STATUS.md) the default salt
    puts attrs and structlog in test, roughly 21% of tasks (about 13 of 60). Hitting 25-30 would need
    click in test, which leaves train small. This is a human decision, left open in NEXT.md; nothing is
    frozen until the real counts exist. Choosing a salt by looking at counts is allowed (counts are not
    outcomes) but must be made once, logged here, and made before any result is seen.
57. **The audit log is append-only and the latest line wins.** `tasks/audit.jsonl` is the source of
    truth; a changed mind is another line, an interrupted session loses nothing, and skipping records
    nothing. `cascade tasks finalize` copies the latest verdict into each task's `audit` field. Only
    `ok` enters experiments (`experiment_tasks`). Tasks are shown in a hash order that mixes repos
    (so fatigue does not fall on one repo), identical on every resume.
58. **Audit hints are model-free and advisory.** Verbatim overlap between the fix and the issue (leak),
    new public names that tests use but the issue never states (underspecified), literals the tests
    assert that the issue never states (arbitrary), and a note on huge fail_to_pass sets (decision 53).
    They are shown above the issue and are never verdicts; the 25-character minimum for a verbatim
    line exists because short lines like `return x` appear in many issues by chance.
59. **Difficulty features need no model and are recomputed at finalize.** Everything but
    `test_runtime_s` is a deterministic function of the issue text, patches, and test id lists, so a
    stale or hand-edited value cannot survive `finalize`. `test_runtime_s` is measured, so it is
    preserved. Old manifest rows without the new fields still load (they default to zero).
60. **Two-level parallelism, with one writer per shared file.** `run-all` runs repos in threads
    (default 4) and, within a repo, PRs in threads (default 2). Each repo has its own state, clone, and
    image; the shared SQLite store takes a lock per statement. The manifest merge and STATUS.md
    refresh happen afterward in one thread, so workers never race on them. Results do not depend on the
    degree of parallelism (a test compares serial and parallel runs). One repo failing is reported and
    the others finish; finished rows from a failed repo are still merged.
61. **The manifest merge keeps a task's split and audit.** Re-mining a repo replaces that repo's lines
    only, and carries over each task's existing `split` and `audit`, so rerunning the pipeline can
    never undo a human verdict or the frozen split. Consequence: `run-all`'s per-repo `PipelineSummary`
    reports `manifest_tasks == 0` because the orchestrator, not the repo run, writes the manifest; the
    report uses the attrition funnel instead.
62. **Disabled repos are skipped by default.** `enabled: false` in `repos.yaml` (jinja, which is
    dormant) excludes a repo from `run-all` unless `--repo` names it or `--include-disabled` is given.
63. **Triage classifies from logged facts only, and has seen no real data.** `cascade triage` reads
    the matrix and each attempt's `result.json` and labels it a pipeline bug, Verifier false positive
    or negative, malformed model output, provider/config error, cap/kill switch, or genuine model
    failure, with the evidence used, and flags leak-audit hits separately. Grading infrastructure
    errors outrank everything (they are never blamed on the model). One mutant (the false-positive
    branch also matching a case that already returned success) is equivalent and cannot be killed.

## Phase 4 (matrix runner, sizing, calibration, ablations)

64. **The inputs the brief said to take from artifacts did not exist, so nothing was assumed.**
    `tasks/manifest.jsonl` and `tasks/audit.jsonl` are 0 bytes, there is no `tasks/split.json`, no
    skeleton run was ever made (so no measured cost), and `configs/models.yaml` / `budget.yaml` were
    unfilled templates. The audited counts are therefore 0, and the "sizing table from skeleton-run
    priors" cannot be produced: STATUS.md says so and shows only arithmetic (the break-even cost per
    task for N >= 40 at a given budget), never a cost figure. `cascade size` computes the real table
    from the pilot ledger.
65. **models.yaml carries the names and arms from the brief, nothing else.** `gpt-6-luna` (low, high)
    and `gpt-6.1-sol` (medium) with labels cheap, mid, max, taken from the brief. Every price and
    cutoff stays `TODO`. The optional arms are declared but disabled: `optional: true` means an arm
    never runs unless `enabled: true` is set by the human. `swap_in_cheap` and `astra_ref` have
    placeholder `TODO-` models because the brief does not define them; `sol_high_ref` is assumed from
    its name to be sol at effort high and is marked for confirmation.
66. **Pricing rules live in config, not code.** Optional per-model fields: `cached_input_usd_per_mtok`,
    `cache_write_usd_per_mtok`, `long_context_threshold_tokens` with input and output multipliers,
    and `flex_multiplier`. Formula per call: cached and cache-write tokens (subsets of input) bill at
    their own price when configured, else at the input price (an overstatement, never an
    understatement); a call whose input is strictly above the threshold bills input and output at the
    multiples, for the whole call; flex multiplies the result; reasoning bills at the output price
    unless a reasoning price is set. `flex_multiplier: null` means flex is unavailable for the model,
    and a flex call on such a model is refused before anything is sent. The worst-case pre-call check
    applies the same multiples, so a call cannot overshoot a cap on its own.
67. **Budget buckets and per-arm caps.** `budget.yaml` gains `buckets` (pilot, matrix, ablate,
    repeat, summing to at most the global cap) and `per_arm_attempt_cap_usd` (keyed by arm label,
    each at most `per_attempt_cap_usd`). The brief's "total_usd" is the existing `global_cap_usd`; no
    second field was added for the same quantity. The ledger enforces both on worst-case figures and
    tags every call with its bucket, tier, and cached tokens (old ledger files migrate in place).
68. **Runner: what is dropped and what is not.** An attempt that reaches its own per-attempt cap is a
    failed cell, recorded with the cost incurred, never dropped. An attempt cut short by the kill
    switch or by the global or bucket cap says nothing about the model, so its row is removed and the
    cell reruns on resume (its spend stays in the ledger and counts against every cap); keeping it
    would bake a budget stop in as a model failure. Before a task's first pending cell the runner
    requires the global, bucket, and per-task budgets to cover all its pending cells at their
    per-attempt caps, and stops there if not, so a stop never leaves a task with only some arms
    (which would bias the arm comparison toward the cheap arm). A per-task cap below one task's
    planned attempts stops the run instead of starving the max arm. Cells run sequentially, because
    authorize and record are not one atomic reservation (see the ledger's known limit).
69. **Pilot rows are matrix rows.** The pilot uses the matrix's pipeline config, seed 0, and standard
    tier, so the matrix skips those cells and `cascade size` credits the pilot tasks as already
    covered. Consequence for flex: a matrix uses one tier, so a flex matrix cannot reuse the standard
    pilot rows; it needs a fresh `--matrix-db`, gets no pilot credit in the sizing table, and every
    task is paid for again. `cascade matrix run` refuses to mix tiers in one store.
70. **Flex: implemented, off by default, and unverified against the live API.** From OpenAI's Flex
    documentation: it is requested per call with `service_tier="flex"`; tokens bill at Batch rates;
    capacity shortages return `429 Resource Unavailable`, which is not charged; the default timeout is
    10 minutes and should be raised; SDKs retry 408 twice; the docs say nothing about tools,
    streaming, or prompt caching under flex, and say supported models are listed on the pricing page.
    The installed openai 3.24.0 / openai-agents 0.23.1 accept `service_tier` (including "flex") and
    pass `ModelSettings.extra_args` through to `responses.create`, so each turn of an agent loop is
    an ordinary flex request. Implemented: `--tier flex`, `service_tier` on every call, exponential
    backoff on resource-unavailable (5 s, doubling, 120 s cap, 8 retries) with no fallback to standard,
    and a longer client timeout for flex. Not verifiable offline: whether gpt-6-luna and gpt-6.1-sol
    support flex, and how often agent loops hit 429s. So `flex_multiplier` is `null` until you
    confirm support on the pricing page, standard stays the default, and the recorded
    `processing_tier` is the tier requested (the response's echoed tier is not captured).
71. **Sampling.** Pilot, ablation, and repeat draws go through one train-only function
    (`draw_train`), stratified round-robin over repo x size tercile (changed source lines), seeded and
    deterministic per purpose, and re-checked on output. Defaults: pilot 10, ablate 20, repeat 10.
    Ablations run the mid arm with the Reproducer or the Reviewer off; their baseline is the matrix's
    full-pipeline mid row for the same task, and only tasks with both rows are compared (the count is
    always printed).
72. **Pre-flight gates real runs.** `cascade preflight` runs the leakage checks on every audited-ok
    task (base-commit workspace and every rendered prompt) and writes `runs/preflight.json`. A real
    (non-mock) run refuses to start unless that record covers every task it is about to run.
    Docker-level properties (network off, read-only root) are covered by the sandbox and grading
    tests and are not repeated per task.
73. **`cascade size` never picks an option.** Costs per attempt are summed from the pilot ledger
    (bucket `pilot`); per arm it prints mean, median, and p90 (linear interpolation; with about 10
    attempts the p90 is roughly the second-largest attempt, and the report says so); cost per task is
    the sum over arms; affordable N is the floor of remaining matrix budget over cost per task, shown
    at the mean and at the p90, with the pilot tasks counted as covered under standard tier only. If
    the full matrix covers fewer than 40 tasks at the p90 it prints DECISION REQUIRED and stops.
74. **Calibration thresholds.** Flags: max arm outside 40-70% (inclusive bounds), cheap arm at 90% or
    more, cheap arm "near zero" at 10% or less (my choice of what "near" means). FP rate is
    P(claimed | failed), FN rate is P(not claimed | passed). The cheap-near-zero flag mentions
    `swap_in_cheap` and says nothing is switched; no code path switches an arm.
75. **One pipeline config for every matrix command**, built by `default_pipeline_config`: the repo
    config's runtime cap plus `--max-output-tokens`, default 8000 here (a per-call ceiling, reasoning
    included). It differs from `cascade run`'s default of 4096, so rows from a bare `cascade run` are
    a different cell than pilot or matrix rows. An earlier draft rebuilt the config differently in
    `size` and `calibrate`, which would have found no rows if the runtime cap differed from its
    default; caught in review and fixed.
76. **New dependency: pyarrow 25.0.1, pinned,** for the parquet export (a straight copy of the matrix
    and ledger rows, deterministic order, nothing derived). The export directory defaults to
    `<runs-dir>/export`; a first draft defaulted to the working directory and wrote files into the
    repo during a test, which a regression test now prevents.

## Phase 5 (routing policies and evals)

77. **No real matrix exists, so no real analysis was run.** `runs/export/` is absent, the manifest and
    audit log are empty, and nothing has been spent. Everything was developed and tested on the
    labeled-synthetic matrix under `tests/synthetic/`. The brief's "dev analysis on the real matrix if
    it exists" therefore did not happen; `cascade eval` on the real paths refuses with a clear error.
78. **Provenance is established from artifacts, not asserted.** Data is `synthetic` only if the
    directory carries the `SYNTHETIC.json` marker the test generator writes. Otherwise it is `real`
    only if `ledger.parquet` exists, is non-empty, and every logged call came from the `openai`
    provider, and the matrix's total cost does not exceed the ledger's (rows must be backed by logged
    calls). A mock-provider matrix, a matrix without a ledger, and a hand-made parquet are all refused.
    One directory, one source: synthetic and real are never combined. Every output is stamped with the
    data source, git commit and dirty flag, the SHA-256 of the matrix file's raw bytes, the tier, and
    the splits.
79. **What a "default-pipeline row" is.** Seed 0 and the full pipeline (reproducer and reviewer on).
    If more than one distinct such config exists the load is refused as ambiguous (a changed
    `--max-output-tokens` mid-experiment would otherwise silently mix configurations). Ablation rows
    are mid-arm rows that differ from the default only by switching exactly one of those two roles off;
    repeat rows are default-config rows with another seed. Neither feeds a policy.
80. **Cap-exceeded cells are failures that keep their cost.** The runner does not emit the string
    `cap_exceeded`; it records `BudgetExceededError: attempt cap ...` and similar. The loader treats an
    error containing `cap_exceeded`, `attempt cap`, `attempt token cap`, or `task cap` as a cap hit and
    sets `true_success` and `verifier_claimed` to false (so a cascade escalates past it), because a
    partial patch that happened to pass is not a result the pipeline produced within its budget. The
    count is reported. Global, bucket, and kill-switch interruptions never reach the matrix (Phase 4).
81. **Mixed tiers are refused over the whole file, without reading outcomes.** The `processing_tier`
    column of EVERY row is checked, including test-split rows, because a tier mix anywhere makes costs
    incomparable. Reading one column of a split is not using that split's outcomes. A mix is refused
    before anything is evaluated and, for `--final`, before the lock is written.
82. **Dev runs never materialize a test row.** The parquet read pushes a `task_id IN (train ids)`
    filter down to pyarrow, so test rows stay on disk; a test spies on the read and asserts it. The
    margin rule needs only the COUNT of audited test tasks, taken from the manifest. The matrix hash is
    of file bytes, so computing it parses nothing.
83. **Cascade, oracle, and the independence assumption.** Cascade: escalate while `verifier_claimed`
    is false; cost is the sum over arms tried; outcome is `true_success` of the last arm tried, so a
    false positive stops it and counts as a failure. Oracle: the cheapest arm by that task's observed
    cost among successful arms, else the cheapest by observed cost (observed cost can reorder arms on a
    task). The cascade lookup assumes each cell is an independent stand-alone run (no information
    forwarded from a failed attempt) and that one sampled outcome is the arm's outcome; the repeat-seed
    flip rate measures the second assumption. Written into `policies.py` and `docs/EVAL_PLAN.md`.
84. **DEVIATION FLAGGED: two feature sets, and the primary one excludes the gold-derived features.**
    The brief lists `src_files`, `src_lines`, and `test_runtime_s` as triage features. They are
    measured on the gold patch or the test run, so a deployed router would not have them before the fix
    exists. `routing/features.py` builds all of them, tagged by group; the plan reports `predictive_issue`
    (issue-side features only) as the primary, deployable router and `predictive_all` as an upper bound.
    If you want the gold-derived features in the primary router, say so before the plan is confirmed.
85. **The two small models are plain numpy, not scikit-learn.** An L2 logistic regression by Newton's
    method (standardized features, unpenalized intercept; the penalty keeps it finite on an all-success
    or all-failure arm) and a ridge regression of log cost with Duan smearing. Deterministic, typed,
    and no heavyweight dependency. The only dependency added this phase is numpy 2.4.6, pinned.
86. **The selection rule and nested CV.** Leave-one-repo-out on train; a candidate (C, value
    multiplier) is feasible if its pooled out-of-fold resolve rate is at least always-max's minus
    delta/2 (a stricter bar than the evaluation margin, to leave room for test noise); choose the
    cheapest feasible; ties go to smaller C then smaller multiplier; none feasible means always-max.
    Dev numbers use NESTED CV (selection repeated inside each outer fold) because the CV of a
    configuration chosen on the same folds is optimistic; with fewer than 3 repos nested CV is
    impossible and the report says it is showing the optimistic number.
87. **The margin rule, and why the headline will often be inconclusive.** delta is the smallest of
    {0.05, 0.10, 0.15} with at least 50% power (when the policy is exactly as good as always-max,
    assuming 20% of tasks differ), capped at 0.15. At 13 to 30 test tasks the power at 0.15 is only
    23% to 45%. The cap exists so the margin is never loosened to manufacture a pass; the plan states
    that an inconclusive verdict will be reported as inconclusive, not retried or re-margined. The
    plan's power table is checked against the code by a test. **Awaiting your confirmation.**
88. **Bootstrap design.** Every resample is a weight matrix so policies are compared on identical
    resamples (paired). 10,000 resamples, seed 4242, 95% percentile intervals; a repo-clustered scheme
    as a sensitivity check (flagged as unreliable under 5 repos); statistics undefined in a resample
    (cost per resolved with nothing resolved) are excluded and counted. A review-found bug: the weighted
    median returned the lower middle value for an even count, unlike `np.median`; fixed, with
    brute-force tests for even and odd counts.
89. **The final guard.** `cascade eval --final` requires: the plan committed (present in HEAD), a clean
    working tree, no `FINAL.lock.json` in the tree or anywhere in git history (so deleting and rerunning
    is refused), real data, one tier, and train and test tasks. It loads train and test in two calls,
    writes the lock with an exclusive create (commit, plan blob hash, matrix hash, margin, n) BEFORE
    evaluating, fits on train only, evaluates the test tasks once, and records output hashes. A failure
    after the lock is written marks it `failed` and the run stays used: recovery is a deliberate human
    act. Refusals that happen before the lock (synthetic, mixed tier, missing data, dirty tree) do not
    spend the run. The lock is committed by the human, which is what makes the history check bite.
90. **Pareto plot is hand-written SVG.** No plotting dependency, byte-for-byte reproducible, and the
    provenance stamp is printed on the figure. Dev outputs go to `results/dev/<source>-<hash>/`
    (gitignored); the final outputs go to `results/final/` and the lock is at the repo root, tracked.

