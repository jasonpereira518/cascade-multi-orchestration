# Next

The pipeline is built and tested but has not run live. Three things from you unblock that, then
one command does the work. Decisions I need are at the bottom.

## 1. Give the pipeline a GitHub token (required)

The brief says to use `GITHUB_TOKEN`. It is not set in the environment I run in, and I did not
borrow the `gh` CLI's stored token (wider scope than a read-only miner needs). Pick one:

- **Recommended:** create a fine-grained personal access token with *public repositories, read-only*
  access (no extra permissions), then in the shell you launch from:

```bash
export GITHUB_TOKEN=<paste token>
```

- Or, if you accept `gh`'s broader OAuth token for this read-only job, your call:

```bash
export GITHUB_TOKEN=$(gh auth token)
```

Never commit it; `.env.example` lists the variable. The code reads it from the environment only.

## 2. Fill the date window (required)

In `configs/repos.yaml` set `date_window.start` and `end` (YYYY-MM-DD). `start` must be after the
newest arm's knowledge cutoff, which comes from `configs/models.yaml` (still unfilled; that file
and `budget.yaml` are not needed for this phase, but the cutoffs decide the window).

For scale: over 2025-10-04 to 2026-10-04 an estimate from click's git history found 126
PR-landing commits, 48 passing the source/test/size rules, and a rough floor of 18 with a closing
keyword in commit messages. The real closing-issue count comes from GraphQL. See `STATUS.md`.

```bash
uv run cascade config-status   # repos.yaml: date_window lines should be gone
```

## 3. Run the token-only stages first (no Docker needed)

This mines, filters, and derives patches, then writes the attrition block into `STATUS.md`. It
makes GraphQL calls only (read-only) and a public `git clone` of pallets/click.

```bash
uv run cascade taskgen run --repo pallets/click --stop-after patch
```

Re-running is safe: finished months and PRs are skipped. If it stops on a rate limit it says when
it resets; just rerun.

## 4. Start Docker, verify the sandbox, run everything

Start Docker Desktop, then check the sandbox flags against a real daemon (this has never run):

```bash
docker pull alpine:3
CASCADE_TEST_DOCKER_IMAGE=alpine:3 make docker-test
```

Then the full pipeline: build the click image, smoke-run each PR's tests, validate (3 runs before
and after the gold patch), and write passing tasks to `tasks/manifest.jsonl`:

```bash
uv run cascade taskgen run --repo pallets/click
uv run cascade taskgen attrition --repo pallets/click --status-md STATUS.md
```

The first image build downloads Python and click's test dependencies and can take a few minutes.
If the build or a test run fails, send me the error text; the Dockerfile is the likeliest culprit
(`STATUS.md` risk 1).

## Decisions I need from you

1. **More repos.** Click alone will not reach Tier 1's 40+ audited tasks (expect roughly 5-20 per
   click-sized repo per year of window). Do you want me to shortlist 3-6 more candidates (Python +
   pytest, permissive license, active, comparable size) using read-only `gh` queries? You pick.
2. **Token choice** (item 1): fine-grained PAT, or `gh`'s token?
3. **Date window** (item 2), and the arm knowledge cutoffs behind it.
4. **Yield levers, only if the live numbers disappoint.** Raising the size limits would gain almost
   nothing (they drop 1-4 PRs a year). The real levers are a longer window (bounded by cutoffs),
   more repos, or relaxing the closing-issue rule (for example using the PR description as the
   issue text). I recommend against the last one: it changes what an "issue" is and weakens the
   benchmark. I will not do it unless you ask.
5. Still open from before: budget numbers, and approval to add the OpenAI Agents SDK as a dependency
   in the agent phase.

## After the live run (STOP B)

Frozen train/test split by hash and by repo, the audit sheet for you to review (only audited-ok
tasks enter experiments), and the real task count against the 40+ target.
