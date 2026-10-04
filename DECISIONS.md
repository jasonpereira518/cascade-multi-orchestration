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
