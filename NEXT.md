# Next

Nothing further runs until you complete items 1 to 3. Items 4 and 5 are optional but recommended.

## 1. Fill `configs/models.yaml` (required)

Replace every `TODO` with values from the provider's own docs and pricing page. I did not fill any
of it, because model names and prices are yours to supply (hard rule 5).

- Per model: `name`, `input_usd_per_mtok`, `output_usd_per_mtok`, `knowledge_cutoff`,
  `reasoning_efforts`. Set `reasoning_usd_per_mtok` only if reasoning tokens bill differently from
  output tokens; otherwise leave it `null`.
- `arms`: Tier 1 needs 3 arms, each a (model, reasoning_effort). The template has
  small@low, large@low, large@high. Change them if you want a different spread.
- The newest `knowledge_cutoff` among your arms sets the earliest allowed PR merge date.

## 2. Fill `configs/budget.yaml` (required)

Set four values: `global_cap_usd`, `per_task_cap_usd`, `per_attempt_cap_usd`,
`per_attempt_max_tokens`. They must satisfy per_attempt <= per_task <= global. Until all four are
set the ledger refuses to start, so nothing can spend. To halt everything at any time:

```bash
touch KILL_SWITCH
```

## 3. Choose the first repo and fill `configs/repos.yaml` (required)

**Decision for you:** which repo? Criteria from the spec: Python + pytest, permissive license,
active, mid-popularity, installs without a GPU, test suite under about 2 minutes, and enough merged
PRs that close issues inside your date window. Also set `date_window.start/end` (start after the
newest arm cutoff from step 1), plus `license`, `python_version`, `install_cmd`, `test_cmd`.

If you would like a shortlist, say so and I will draft candidates at the next session using
read-only `gh` queries (the GitHub MCP server failed to connect in this session). You pick one.

Then check what is left (it prints one line per unfilled field and exits 1 until all are filled):

```bash
uv run cascade config-status
```

## 4. Verify the sandbox against a real Docker daemon (recommended)

This is the one thing I could not verify: no Docker daemon was running. Start Docker Desktop, then
pick any small local image that has `sh`, `cat`, `ls`, and `touch` (for example `alpine`):

```bash
docker pull alpine:3
CASCADE_TEST_DOCKER_IMAGE=alpine:3 make docker-test
```

It asserts: only the loopback interface exists (network off), no `.git` in the workspace, the
workspace is writable, and the root filesystem is read-only. Tell me if it fails and paste the
output; the flags are the likely culprit.

## 5. Optional housekeeping

- Create a GitHub remote if you want CI to run (`gh repo create`, then push). I have not pushed
  or created anything remote.
- Provide `GITHUB_TOKEN` (read-only, public repos) in a local `.env` for task mining. Never commit
  it; `.env.example` shows the variable names.

## Decisions I need from you

1. First repo (item 3).
2. Budget numbers (item 2). For orientation only: the worst-case pre-call check is deliberately
   pessimistic, so caps that are too tight will refuse calls that would have fit.
3. Confirm the next stage may add and pin the OpenAI Agents SDK and OpenAI SDK as dependencies. I
   will check their APIs against the installed versions before using them.

## What happens next (after steps 1 to 3)

STOP 2 is the $0 task pipeline for the first repo: mining via GraphQL, filtering, per-repo image
build, validation (3 runs, flake removal, runtime cap, attrition per stage), the frozen split, and
an audit sheet for you to review. No LLM calls and no spend. Only audited-ok tasks go forward.
