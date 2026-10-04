# Status

**Stage:** STOP A (Phase 1: task pipeline for pallets/click) reached. Halted for the human; see `NEXT.md`.
**Updated:** 2026-10-04

**Acceptance (at least 3 validated tasks from click): not met, and that is stated plainly.** No
live run has happened, so there are zero pipeline-validated tasks and no pipeline attrition table
yet. Three things block a live run: no `GITHUB_TOKEN` in this environment, the Docker daemon is
not running, and `date_window` in `configs/repos.yaml` is still `TODO`. Everything else is built
and tested. What I can offer instead is in "Evidence short of a live run" below, labeled by what
each piece is and is not.

## Done

- **Phase 0 foundation** (unchanged): configs and schemas, budget ledger, providers with the spend
  gate, agent-side sandbox with the leak guard, smoke test. See `DECISIONS.md` 1-14.
- **Phase 1 pipeline** (`cascade/taskgen/`), all stages resumable and idempotent:
  - `github.py`, `mine.py`: GraphQL client (token from `GITHUB_TOKEN` only, retry/backoff, optional
    response recorder) and monthly-slice mining of merged PRs with closing issues.
  - `filter.py`: 16 rule codes (date window, default branch, bot, dependency, docs-only, issue
    rules, test/source presence, 1-5 source files, 150 source lines).
  - `gitrepo.py`: bare clone, base commit and gold/test patches from the merge commit's first
    parent, cross-checked against GitHub's per-file counts; patch application.
  - `build_env.py`: per-repo Dockerfile (Python pinned and installed by uv, editable install),
    deterministic image tag, `docker build`, and a test executor that runs with the network off.
  - `validate.py`: 3 pre + 3 post runs in fresh workspaces, flaky detection, runtime cap,
    fail_to_pass / pass_to_pass.
  - `store.py`, `attrition.py`, `pipeline.py`, CLI: SQLite stage store, attrition table written into
    this file, `cascade taskgen run|attrition`.
  - `harness/grading.py`: grading runner sharing every isolation flag with the agent runner.
  - `docker/cascade_pytest_report.py`: pytest plugin recording exact node ids and outcomes.
- `configs/repos.yaml`: `pallets/click` filled (BSD-3-Clause, Python 3.12.13, editable install
  with the `tests` group, `python -m pytest`). `date_window` left `TODO`.
- CI uses fixtures only: no network, no Docker, no token.

## Verified

| Check | Result |
|---|---|
| `make check` (ruff, format, pyright strict, pytest) | exit 0; 257 passed, 1 skipped (real-Docker test) |
| `make smoke` | exit 0 |
| `uv lock --check` | current |
| Guard + Phase 1 logic mutation check (28 deliberate breakages on a scratch copy) | 28 of 28 caught (one escaped the first time, a pending-count bug; test added, rerun caught it) |
| GraphQL fields (`closingIssuesReferences`, `files`, `mergeCommit`, `search`, `PatchStatus`...) | checked against GitHub's published schema; one of my assumptions was wrong (`changeType` is `PatchStatus`) and was corrected before coding |
| `uv:0.11.2` image tag, `debian:bookworm-slim` | both exist (registry queries) |
| Click installs and tests natively with `uv pip install --group tests -e .` | yes (see below) |

The mutation script is a one-off on a scratch copy, not committed and not in CI.

## Evidence short of a live run

**1. Native dry run of the real code on real click PRs (not sandboxed, not a pipeline result).**
`derive_patches` + `validate_task` on six real click PRs, using a host venv and the real plugin,
nothing saved to pipeline state. This is *not* the sanctioned path (the network was on, no
container), so none of it may be cited as a validated task.

| PR | validated | fail_to_pass | pass_to_pass | flaky excluded | median post run |
|---|---|---:|---:|---:|---:|
| #3739 | yes | 6 | 1950 | 0 | 3.8 s |
| #3728 | yes | 1 | 1946 | 5 | 3.6 s |
| #3681 | yes | 30 | 1863 | 0 | 3.7 s |
| #3677 | yes | 38 | 1710 | 0 | 3.9 s |
| #3642 | yes | 1 | 1680 | 0 | 3.6 s |
| #3769 | yes | 6 | 1962 | 0 | 3.5 s |

It shows the validation logic behaves on real tests (the five #3728 flakes were found and
excluded while the real fail_to_pass test stayed) and that click's suite is fast and stable: 2240
passed, 25 skipped, 1 xfailed, about 4-5 s, identical per-test outcomes across three native runs.
It does not show that the Docker image builds or that tests pass with the network off.

**2. Exploratory funnel estimate from click's real git history (not a pipeline result).** Applies
the project's own classifier and size rules to PR-landing commits (subject ends in `(#N)`) on
main's first-parent history. It cannot see bot authors or closing-issue links (GraphQL only); a
closing keyword in commit messages is a weak proxy, so the last column is a rough floor.

| Window | PR-landing commits | pass git-checkable filters | with a closing keyword in commit messages |
|---|---:|---:|---:|
| 2025-10-04 to 2026-10-04 | 126 | 48 | 18 |
| 2024-10-04 to 2025-10-04 | 96 | 33 | 4 |

Dominant drops, last 12 months: `no_test_files` 31, `docs_only` 30, `no_source_files` 8,
`branch_sync_merge` 5, `too_many_source_files` 2, `too_many_source_lines` 1. The size limits
cost almost nothing, so loosening them would not raise yield.

<!-- attrition:pallets/click:begin -->
### pallets/click

**Not yet run.** This block is generated by `cascade taskgen run` from the saved stage results
and is the only place a pipeline attrition table belongs. Until that runs there is no table.
<!-- attrition:pallets/click:end -->

## Projected tasks per repo (an order-of-magnitude bracket, not a measurement)

For a click-sized repo over a 12-month window: at most about 48 PRs clear the source/test/size
rules; the closing-issue requirement (the spec's mining rule, and the least certain filter here)
plausibly brings that to the 5-20 range before env and validation attrition. Validation attrition
looked small in the dry run (6 of 6), but that is six hand-picked PRs. So expect **roughly 5-20
tasks per click-sized repo per year of window**, and treat that as unverified until a live run.

Consequence for the plan: Tier 1 needs 40+ audited tasks, and the window is capped from below by
the arms' knowledge cutoffs, so click alone will not get there. Expect to need about 3-6 repos of
this size. This is the main finding of this stage.

## Spend

**$0.** No LLM call exists in this phase. GitHub API calls: none by the pipeline (no token). The
only network use was reading public files, the schema, registry tags, and cloning public
pallets/click, plus package downloads for the scratch venv.

## Risks and unverified items

1. **Docker is untested end to end.** Dockerfile text, tag, build command, and executor command
   are covered by unit tests with a fake docker; a real `docker build` and a real network-off run
   have never happened. Likely trouble spots: base image needing CA certificates for `uv python
   install` (uv bundles roots, unconfirmed), `--user` host uid against Docker Desktop bind mounts,
   `--read-only` root with the editable `.pth`. `make docker-test` still skips without an image.
2. **The live GraphQL queries have not executed.** Field names match the published schema, and
   `parse_pr` is tested on shaped-but-synthetic responses. A real response could differ in a way
   the schema check cannot show (null handling, `files` truncation, huge bodies).
3. **Closing-issue rule is the biggest yield unknown.** Click PRs often reference issues in
   descriptions in ways GitHub may or may not treat as closing; the live `closing` count decides.
4. **Native dry run != sandbox.** Network was on and the interpreter was a host Python 3.12.13.
5. **Image reference commit drift.** One image serves the whole window (`DECISIONS.md` 23); tasks
   far from the reference commit may fail env or validation. That would show as attrition.
6. **Issue text is current text,** not the text at fix time (`DECISIONS.md` 18). Audit must watch
   for issues edited to contain the solution.
7. **Strict diff check may drop some valid PRs** (`DECISIONS.md` 16). Visible as `diff_*mismatch`.
8. **Phase 0 limits still apply** (ledger overshoot under parallelism, leak-guard excerpt limit).
