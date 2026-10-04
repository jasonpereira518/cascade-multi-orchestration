# Status

**Stage:** STOP C (Phase 3: scale, split, audit) reached. Halted for the human; see `NEXT.md`.
**Updated:** 2026-10-04

**Acceptance (Phase 3):** audit CLI built and tested, split deterministic and tested, difficulty
features computed without a model, per-stage attrition and per-repo counts printable, `make check` and
`make smoke` pass. **Not met, and why:** the *candidate manifest was not produced*. Mining needs
`GITHUB_TOKEN` (unset) and the date window in `configs/repos.yaml` is still `TODO`. I have not borrowed
another credential. **Step 1 (triage the real skeleton run) had no data**: no real `cascade run` has
happened, so there were no failures to classify. I built the triage tool (`cascade triage`) so it is
ready when results arrive.

## Phase 3 (this stop)

- **Scale** (`taskgen/orchestrate.py`, `cascade taskgen run-all`): all enabled repos in parallel
  (repos x PRs), resumable and idempotent, a failing repo does not stop the rest, one writer for the
  manifest and this file. `cascade taskgen report` prints per-stage attrition, drop reasons, and
  per-repo task counts (with split) from saved state, with no network and no Docker.
- **Split** (`taskgen/split.py`, `cascade split freeze|show`): repo-level holdout by SHA-256 hash,
  frozen in `tasks/split.json`, refuses to overwrite, late repos by hash bucket, `fitting_tasks`
  guard. **Not frozen yet**: it needs the real per-repo counts (see risk 6 for a coarseness problem).
- **Audit** (`taskgen/audit.py`, `cascade audit review|status`): one task at a time (issue, gold patch,
  test patch, model-free hints, tests that must pass), verdicts ok / underspecified / leaks_fix /
  arbitrary_test appended to `tasks/audit.jsonl`, resumable, `--redo`, `--full`, filters.
- **Features** (`taskgen/features.py`): 15 model-free difficulty features, recomputed by
  `cascade tasks finalize`, which also stamps the split and audit verdicts onto the manifest.
- **Triage** (`agents/triage.py`, `cascade triage`): see `DECISIONS.md` 63.
- **Config**: `enabled` flag per repo; jinja (dormant) is `enabled: false`.

| Check | Result |
|---|---|
| `make check` (ruff, format, pyright strict, pytest) | exit 0; 607 passed, 1 skipped (real-Docker test), smoke test run separately |
| `make smoke` | exit 0 |
| New tests | split 23, audit 33, manifest/store/config 17, orchestrate 12, triage 26, CLI 19, features 11 |
| Mutation check on the new logic (20 deliberate breakages, scratch edits restored) | 19 caught after one new test; 1 equivalent mutant (no observable difference) |
| Real Docker / real GitHub / real model | **not exercised this phase**; orchestration is tested against a fake GitHub, real local git upstreams, a fake docker, and a fake executor, all labeled synthetic |

Bugs found by the Phase 3 tests: none in the product code. Two test assumptions were wrong and fixed
(run-all deliberately reports `manifest_tasks == 0` per repo; two separately built git upstreams have
different shas). The mutation check found one weak test (the 25-character leak threshold was never
exercised); it is fixed.

## Done

- **Phase 0 and 1** (unchanged): foundation, the $0 task pipeline for pallets/click, ledger, sandbox.
- **Agents** (`cascade/agents/`): five LLM roles as OpenAI Agents SDK agents with pydantic structured
  outputs (Triager, Reproducer, Localizer, Patcher, Reviewer) and the sixth, the Verifier, as
  deterministic code. `pipeline.py` has toggles for Reproducer and Reviewer, a bounded repair loop,
  and stops cleanly when a cap or the kill switch trips (the patch and Verifier report are still
  recorded). Deterministic tools: BM25 and regex search, bounded file read/edit with strict path
  safety, a test-failure parser, and a source-only patch builder.
- **Metering** (`harness/models.py`): `MeteredModel` wraps any SDK model so every call, including
  every turn of a tool loop, is authorized against the ledger (global/task/attempt dollar caps and
  the attempt token cap, on worst-case figures), priced from the provider's reported usage and
  models.yaml, and logged. This replaced the Phase 0 provider layer. SDK tracing is off.
- **Providers**: mock built on the SDK's `ScriptedModel` (default); real OpenAI source behind
  `CASCADE_ALLOW_SPEND=1` and `OPENAI_API_KEY`.
- **Both outcomes per attempt** (`agents/runner.py`, `routing/matrix.py`): `verifier_claimed` from the
  Verifier, `true_success` from hidden-test grading in a fresh workspace, in a matrix row whose every
  number is a sum over logged ledger calls. Rows are idempotent and keyed by task, arm, seed, config.
- **CLI**: `cascade run --task --arm [--seed --no-reproducer --no-reviewer --max-repairs --cap-usd
  --provider --force]` and `cascade ledger-summary`.
- **Leakage**: tests on the sandbox contents, on every prompt of a full attempt, on the tools, on the
  live leak guard, and a post-hoc trace audit with planted-leak controls.
- `system_packages` per repo (click needs `less`), found by the first real Docker run.

## Verified

| Check | Result |
|---|---|
| `make check` at STOP B (ruff, format, pyright strict, pytest) | exit 0; 462 passed, 1 skipped (the real-Docker test, which passes when given an image) |
| `make smoke` (full pipeline, fixture, mock) | exit 0 |
| Real-Docker sandbox test (`CASCADE_TEST_DOCKER_IMAGE=ubuntu:24.04`) | passes: 0 routes off the box, no `.git`, workspace writable, root read-only |
| click image built for real; click's suite run offline in it | 2241 passed, 24 skipped, 1 xfailed in ~4 s |
| Phase 2 logic mutation check (30 deliberate breakages on a scratch copy) | 29 caught; 1 equivalent mutant (no observable difference). Two real gaps found and closed with new tests |
| SDK API facts (strict schemas, usage fields, exception behavior, scripted model) | read from source and prototyped against openai-agents 0.23.1 / openai 3.24.0 |

Bugs found by the tests along the way, all fixed: patches built with `git diff --no-index a b` had
`a/a/<path>` headers and could never apply; added files got `b/x b/x` headers; the leak audit missed
patches escaped several JSON levels deep; the leak audit cried wolf when a correct one-line fix
equalled the gold patch byte for byte; the sandbox test asserted "only `lo`" which is wrong.

## Evidence short of a live run

**Exploratory run of the real CLI through real Docker on a real click PR (#3642), scratch task,
placeholder issue text, scripted mock models. A plumbing check, not a result.**

- Real validation in Docker: 1 fail_to_pass, 1681 pass_to_pass, 0 flaky, 43 s.
- `cascade run` (Reproducer off, `--max-repairs 0`): the scripted Patcher applied the fix through its
  own `edit_file` tool; the Verifier ran click's suite in the sandbox; hidden-test grading ran in
  Docker. Output: `true_success=True verifier_claimed=True`, 5 model calls, 24 s wall, `$0.0021`
  (synthetic prices on scripted token counts).
- `cascade ledger-summary` printed the by-arm, by-role, and per-attempt tables as it will for you.

This exercised the network-off sandbox, the leak guard on real click files, the Docker executor,
patch application, grading, and the CLI. It did not exercise a real model, the Reproducer through
Docker, or any prompt quality.

<!-- attrition:pallets/click:begin -->
### pallets/click

**Not yet run.** This block is generated by `cascade taskgen run` from the saved stage results
and is the only place a pipeline attrition table belongs. Until that runs there is no table.
<!-- attrition:pallets/click:end -->

## Repos added at the human's request (jinja, attrs, structlog, marshmallow)

All four are in `configs/repos.yaml`. Each was built and run for real; none has been mined, because
there is still no `GITHUB_TOKEN`. `make check`: 466 passed, 1 skipped.

| Repo | License | Suite in the image (offline) | Real PRs validated in Docker | Issue-linked and passes my filters, 12 months to 2026-10-04 | Prior 12 months |
|---|---|---|---|---:|---:|
| pallets/click | BSD-3-Clause | 2241 tests, ~4 s | 1 of 1 (+6 of 6 native) | 32 | 23 |
| marshmallow-code/marshmallow | MIT | 1190 tests, ~1.5 s | 3 of 3 | 9 | 0 |
| hynek/structlog | MIT OR Apache-2.0 | 884 tests, ~1.4 s | 3 of 3 | 7 | 10 |
| python-attrs/attrs | MIT | 1401 tests, ~6 s | 2 of 2 | 4 | 11 |
| pallets/jinja | BSD-3-Clause | 911 tests, ~2 s | n/a | **0 (dormant)** | 0 |
| **Total** |  |  |  | **52** | **44** |

Method and limits (exploratory, not a pipeline result): the counts are PRs that pass my
git-checkable filters (source and test files, size limits, not docs, dependency, or branch-sync)
**and** that GitHub's `linked:issue` search says are linked to an issue. They are upper bounds: the
single-closing-issue, same-repo, non-empty-body, env-build, and validation rules are not applied, and
`linked:issue` is close to but not exactly `closingIssuesReferences`. Details: `DECISIONS.md` 48-53.

Findings worth knowing:
- **Jinja is dormant** (last commit to `main` 2025-06-14): no tasks; skip it.
- **GitHub reports structlog's license as `NOASSERTION`**, which would have failed the mine-time
  license check; fixed with the new optional `github_license_id`.
- Marshmallow's default branch is `dev`; the pipeline reads it from the API, so this just works.
- Click's issue-linked count (73) is far above the old commit-message proxy (18).

## Projected tasks (an upper-bound bracket, not a measurement)

Across the four live repos, **about 52 candidates for the latest 12 months and 44 for the year
before** (table above), before the single-issue, env-build, and validation rules. Real validation
attrition looked small (15 of 15 real PRs validated across click and the three new repos), but those
were taken from the passing set, not drawn at random, so that is weak evidence. If 70 to 90 percent
survive, one 12-month window gives roughly **35 to 45 tasks**, which is right at Tier 1's 40+ target.
A window of 18 months or more (if your arms' knowledge cutoffs allow it) gives comfortable margin.
Click supplies most of it (32 of 52), so any click-specific problem matters a lot.

## Spend

**$0.** No real provider call has ever been made, by this phase or any earlier one. Costs in tests
and in the exploratory run are scripted token counts priced with synthetic prices.

GitHub access this round: read-only search queries on public repositories through the connected
GitHub MCP tool (merged PRs linked to issues, repository license). No writes, nothing opened or
commented, and no token handled by me. The pipeline itself still reads only `GITHUB_TOKEN`.

## Risks and unverified items

1. **The real OpenAI path has never run.** Construction is verified; a request never was. The first
   paid run is the first test of: the model and reasoning-effort strings you configure, strict
   structured outputs with these schemas, strict tool schemas, whether `max_output_tokens` leaves
   room for reasoning, and the usage fields the cost is computed from. Reconcile the ledger total
   against your provider dashboard after the first run. Details in `NEXT.md`.
2. **Real cost per attempt is unmeasured.** Mock token counts are scripted. A real attempt can run
   dozens of tool turns, each re-sending context. The caps are hard, but a cap that is too tight
   ends attempts early, and the pre-call check is pessimistic by design (`DECISIONS.md` 5).
3. **No validated tasks exist.** STOP B needs about 5, so the Phase 1 live run (`NEXT.md` part 1)
   must happen first. Phase 1 risks stand: GraphQL queries never executed against the live API; the
   closing-issue rule is the biggest yield unknown.
4. **Prompts are untuned.** A mock cannot judge prompt quality. Expect the first real runs to show
   malformed outputs, wasted turns, or a Reproducer that never lands a valid failing test. That is
   what STOP B is for.
5. **The Reproducer's `run_tests` through Docker was not exercised** (the exploratory run turned the
   Reproducer off). It uses the same executor class the Verifier used, and is unit-tested with the
   native executor, but it has not run in a container.
6. **Failed provider calls are logged at $0.** If a provider bills a failed request, the ledger
   would undercount it (`DECISIONS.md` 32). The reconciliation in item 1 would show it.
7. **Equal-patch coincidence.** A real fix can equal the gold patch byte for byte; the audit exempts
   only the whole-patch check on reviewer calls (`DECISIONS.md` 42).
8. **Test suite takes about 2.5 minutes** (real pytest subprocesses); CI will be correspondingly slow.
9. **Earlier limits still apply:** ledger overshoot under parallelism, leak-guard excerpt limit, the
   unexecuted GraphQL path, issue text being current text.
6. **A repo-level split cannot reach 25-30 test tasks with this repo set.** Click is about 60% of the
   projected supply, so the hash split puts a small repo or two in test (roughly 13 of 60 with the
   default salt on upper-bound counts) unless click itself is held out, which starves train. This is a
   structural cost of holding out whole repos, not a bug. Decision needed (`NEXT.md`).
7. **The audit is the bottleneck and is a human judgment.** About 60 tasks to read, each with a patch.
   The hints are heuristics (verbatim overlap, undeclared names) and will miss subtle leaks and flag
   harmless ones. Verdicts are the only thing that decides which tasks enter experiments.
8. **Nothing in Phase 3 touched the live services.** GraphQL against the real API, the real clone
   and image build at scale, and Docker load at `workers x jobs` are untested; start with the
   defaults (4 x 2) and watch the first run.
