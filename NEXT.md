# Next (STOP D): pilot, size, calibrate, matrix, ablate

Nothing in Phase 4 has spent money or touched a real service: everything was dry-run on the mock
provider with synthetic tasks (`STATUS.md`). The seven steps you asked for are below, in order. They
cannot start yet, because **the audited tasks do not exist and the configs are not filled**. Those
prerequisites come first, and they are all yours.

## Before step 1: what only you can supply

**A. Audited tasks.** `tasks/manifest.jsonl` and `tasks/audit.jsonl` are empty, and there is no split.
Do "Prerequisite: the audit" further down (token, date window, `taskgen run-all`, split, audit,
`tasks finalize`). The matrix needs every audited-ok task to have a split.

**B. Fill `configs/models.yaml`.** `uv run cascade config-status` lists every field still `TODO`.
For `gpt-6-luna` and `gpt-6.1-sol` fill the prices and `knowledge_cutoff` from the pricing page. Also
look for these rules and fill them only if the page has them (an unset rule means "no such rule"):
cached-input price, cache-write price, and the long-context rule (threshold, input multiple, output
multiple). **Check how the long-context rule is applied:** I implemented "a call whose input is
strictly above the threshold bills the whole call's input and output at the multiples". If the
provider prices only the tokens above the threshold, or applies a different multiple to output, tell
me and I will change the formula; do not fill the numbers in until we agree.
Set `flex_multiplier` (the brief expects 0.5) **only** after confirming that those models support
Flex; leave it `null` otherwise. The optional arms (`swap_in_cheap`, `sol_high_ref`, `astra_ref`)
never run unless you add `enabled: true`; `swap_in_cheap` and `astra_ref` have placeholder models
because the brief does not define them, and `sol_high_ref` is my guess from its name (sol, high).

**C. Fill `configs/budget.yaml`.** `global_cap_usd` is the "total_usd". Guidance so the caps cannot
cut the work short (the runner stops cleanly at a task boundary if a budget cannot cover a whole
task, so too-small caps show up as an early stop, not as bad data):
- `per_arm_attempt_cap_usd` (`cheap`, `mid`, `max`): the most one attempt of that arm may cost.
- `buckets.pilot` at least 10 x (cheap + mid + max caps); `buckets.matrix` = what you want to spend
  on the matrix; `buckets.ablate` at least 20 x 2 x mid cap; `buckets.repeat` at least 10 x (cheap +
  mid caps). The buckets must sum to at most `global_cap_usd`.
- `per_task_cap_usd` at least 2 x cheap + 4 x mid + max caps: one train task can appear in the
  pilot/matrix, both ablations, and the repeat. The run stops with `task_cap_too_low` otherwise.
- `uv run cascade config-status` should then show only the `repos.yaml` lines you already know about.

**D. Preflight.** `uv run cascade preflight` checks every audited-ok task for leaks (workspace and
every prompt) and writes `runs/preflight.json`. A real run refuses to start without a passing record
for the tasks it will run. If a task fails, the output names the violation; fix it or mark the task
`leaks_fix` in the audit.

## The seven steps

1. **Set an OpenAI dashboard spend limit slightly above `global_cap_usd`.** That is the last line of
   defence if anything here is wrong; the ledger caps are not a substitute for it.

2. **Run the pilot.**

```bash
export OPENAI_API_KEY=<your key>
export CASCADE_ALLOW_SPEND=1
uv run cascade pilot --provider openai
```

   `--provider openai` is required: the default provider is the mock (hard rule 2), so without it the
   pilot silently runs for $0 on the mock. The pilot draws 10 TRAIN tasks (stratified by repo and
   size), runs all three arms at the standard tier, and prints per-task cost, tokens, and both
   outcomes (`true_success`, `verifier_claimed`). It is resumable (rerun the same command) and
   `touch KILL_SWITCH` stops it. Look at the first few lines before letting it finish; the first
   real calls are the first test of the model names, the effort strings, and strict schemas
   (see "What I expect to go wrong first").

3. **Size and calibrate, then paste both outputs into the chat before continuing.**

```bash
uv run cascade size
uv run cascade calibrate
```

   `size` prints cost per attempt per arm (mean, median, p90), cost per task, and how many tasks each
   option can cover. It **does not choose**. If the full 3-arm matrix covers fewer than 40 tasks at
   the p90 it says `DECISION REQUIRED`; do not start step 5 until we have decided. `calibrate` prints
   pass rate with counts and an interval (10 tasks is noisy), the Verifier's claimed rate and
   false-positive/false-negative counts, and flags.

4. **If anything is flagged, decide:**
   - *Cheap arm near 0%* (or at 90%+): swap it (define `swap_in_cheap` in `models.yaml`, add
     `enabled: true`, and tell me; nothing switches automatically), rebalance the task set, or accept.
   - *Max arm outside 40-70%*: the tasks may be too easy or too hard to compare policies; options are
     different tasks (more repos or a different window) or going ahead and saying so in the write-up.
   - *Too few affordable tasks*: reduce N (a random prefix of the audited tasks, which the task-major
     order supports), drop the mid arm, raise the budget, or use Flex (below).
   - **Flex** is opt-in and unverified: it halves the price (if `flex_multiplier` is set) but adds
     queueing and `429 Resource Unavailable` retries, and a Flex matrix cannot reuse the standard-tier
     pilot rows (one matrix, one tier): it needs `--tier flex --matrix-db runs/matrix_flex.sqlite`
     and pays for every task again. Decide before step 5; never mix tiers.

5. **Run the matrix (resumable) and watch the ledger.**

```bash
uv run cascade matrix run --provider openai
uv run cascade matrix status
uv run cascade ledger-summary
```

   Cells run one task at a time, all arms (cheap to max) before the next task, in a fixed shuffled
   order, so any prefix is a random sample with complete arm coverage. It skips finished cells
   (including the pilot's), stops cleanly at the global or bucket cap, and writes
   `runs/matrix_status.json`. **The kill switch is the stop file:** `touch KILL_SWITCH` (remove it to
   resume). If it stops with `global_cap` or `bucket_cap`, nothing was lost: rerun after raising the
   cap. `--max-cells N` runs only N cells.

6. **Run the ablations and the repeat subset.**

```bash
uv run cascade ablate --provider openai
uv run cascade repeat-subset --provider openai
```

   `ablate` runs the mid arm with no Reproducer and with no Reviewer on 20 TRAIN tasks and compares
   against the matrix's full-pipeline mid rows (run the matrix first, or the comparison has no
   baseline). `repeat-subset` reruns cheap and mid on 10 TRAIN tasks with a second seed to estimate
   noise. Neither touches the test split.

7. **Run `cascade stats` and paste the output.**

```bash
uv run cascade stats
uv run cascade export
```

   `stats` prints repos, audited tasks (train/test), arms, attempts, tokens (input, output,
   reasoning, cached), spend by arm and by bucket, and the number of tests in this repo; every
   number comes from an artifact. `export` writes `runs/export/matrix.parquet` and `ledger.parquet`.

## Decisions I need from you

1. Everything under "Before step 1": the prices and rules (and whether long-context applies as I
   implemented it), the budget numbers, the flex multiplier (only if supported), and the optional
   arms' definitions.
2. The audit and the split (see the prerequisite below; the repo-level split cannot reach 25-30 test
   tasks with click dominating the supply).
3. After step 3: how to handle any flag, and whether the matrix is affordable as is.

---

# Prerequisite: the audit (STOP C, still open)

Goal: a **candidate manifest**, then **your audit**. Target is **60 tasks audited `ok` (minimum 40)**,
with about **25 to 30 of them in the test split**. Nothing here spends money. Nothing in this phase
has touched the live GitHub API, so expect a first-run surprise or two and paste back anything odd.

Two things block the audit and only you can clear them: a **GitHub token** (I have not borrowed any
other credential) and the **date window** (it must start after your newest arm's knowledge cutoff in
`models.yaml`, which you have not filled yet; a window of 18 months or more gives comfortable margin).

## Part A. Build the candidate manifest

```bash
export GITHUB_TOKEN=<fine-grained token, public repositories read-only>
```

Edit `configs/repos.yaml`: set `date_window.start` and `date_window.end` (`YYYY-MM-DD`). Then:

```bash
uv run cascade taskgen run-all --stop-after patch
```

That mines, filters, and builds patches for the four enabled repos in parallel (token only, no Docker;
jinja is dormant and skipped). Look at the attrition table it prints. Then, with Docker running:

```bash
uv run cascade taskgen run-all
```

It builds each repo's image and validates every surviving PR three times before and after the fix
(this is the slow step; defaults are 4 repos x 2 PRs at once, lower with `--workers 2 --jobs 1` if your
machine struggles). It is safe to interrupt and rerun: finished PRs are skipped and transient errors
are retried. To see the funnel again at any time, with no network or Docker:

```bash
uv run cascade taskgen report
```

**Paste back** the table it prints (per-stage attrition, drop reasons, tasks per repo) and any line
under `REPOS THAT STOPPED EARLY`. If the total is under 40, tell me and I will propose filter changes
from the drop reasons; I will not loosen anything on my own.

## Part B. Decide the split, then freeze it (once)

**Decision for you.** Holding out whole repos is coarse. Click is about 60% of the projected supply,
so the default hash split will probably put one or two small repos in test: on the upper-bound counts
from `STATUS.md` that is roughly 13 of 60 tasks, below the 25-30 target. Your options:

1. **Accept a smaller test split** (cleanest; the result then rests on fewer test tasks and the
   confidence interval will be wide, which the README will say).
2. **Hold out click** (about 25-30 test tasks, but train shrinks to the three small repos).
3. **Add more repos** before freezing, so no single repo dominates.

See what the default would do with your *real* counts, without writing anything:

```bash
uv run python -c "from collections import Counter; from cascade.taskgen import split as s; from cascade.taskgen.manifest import read_manifest; c = Counter(t.repo for t in read_manifest('tasks/manifest.jsonl')); print(dict(c)); print('test repos:', s.choose_test_repos(c, salt=s.DEFAULT_SALT, target_test_fraction=0.4))"
```

Choosing a salt or target by looking at counts is allowed (counts are not outcomes), but do it **once,
before any result exists**, and tell me so I can log it in `DECISIONS.md`. When you are happy:

```bash
uv run cascade split freeze
```

It refuses to overwrite an existing split. Commit `tasks/split.json` right away. Then stamp the manifest
(split, features, audit) so the audit cards show each task's split:

```bash
uv run cascade tasks finalize
```

## Part C. Audit

```bash
uv run cascade audit review
```

You see one task at a time: the issue (all the agent will ever see), the gold patch and test patch
(hidden from agents), the tests that must pass, some model-free hints, and the rubric. Answer with one key:

| Key | Verdict | Meaning |
|---|---|---|
| `o` | ok | the issue gives enough for a competent engineer to write a fix that passes the hidden tests |
| `u` | underspecified | the hidden tests need behaviour, names, or messages the issue never gives |
| `l` | leaks_fix | the issue text contains the fix itself (code or a patch), so it is trivial |
| `a` | arbitrary_test | the hidden tests check incidental details unrelated to what the issue asks |
| `s` | skip | decide later (nothing is recorded; it comes back next session) |
| `q` | quit | everything so far is saved |

After a verdict you are asked for an optional one-line note (press Enter to skip). The hints
(`LEAK?`, `UNDERSPECIFIED?`, `ARBITRARY?`) are only pointers: they miss subtle problems and flag harmless
ones, and the verdict is always yours. Tasks with hundreds of fail_to_pass tests are usually a test
module that could not import before the fix; judge the issue, not the count. Tasks appear in a fixed
order that mixes repos. Useful options: `--split test` (audit the test split first, since it is the
scarcer one), `--repo <owner/name>`, `--task <id>`, `--full` (no truncation), `--redo` (revisit).

```bash
uv run cascade audit status
```

shows progress against the targets. The log is append-only and the latest line wins, so changing your
mind is just another verdict. When you have enough (or are done), stamp the verdicts onto the manifest:

```bash
uv run cascade tasks finalize
```

Commit `tasks/audit.jsonl` and `tasks/manifest.jsonl`. **Only tasks you marked `ok` enter experiments.**
If you end with fewer than 40 `ok`, or fewer than about 25 `ok` in test, tell me; the options are more
repos, a longer window, or relaxing a filter, and each is your call.

**Paste back** the output of `cascade audit status` and `cascade split show`.

## Decisions I need from you (STOP C)

1. **The token and the date window** (part A). The window depends on the knowledge cutoff of your
   arms (`models.yaml`), so those three are linked.
2. **The split choice** (part B): accept a smaller test split, hold out click, or add repos.
3. **The audit** (part C): the only step that cannot be automated, and the one that decides what
   enters the experiment.
4. The models and budget configs (step "Before step 1", B and C, above) are independent of the audit,
   but the pilot needs both the audit and the configs.

---

## What I expect to go wrong first on the pilot (so you recognize it)

This is the first contact with the real API, so some friction is likely and is the point of STOP B:

- **`error: ... model not found` / HTTP 400:** the model or effort string in `models.yaml` is not what
  the API accepts. Fix the string, not the code.
- **Attempts ending with `MaxTurnsExceeded` or `ModelBehaviorError` in the stages:** the model did not
  finish, or returned output that does not match the schema. Prompts are untuned; tell me what you see.
- **Empty or truncated outputs:** `--max-output-tokens` includes reasoning. Raise it (try 16000).
- **`BudgetExceededError ... attempt cap`:** the cap bound before the work finished. Raise
  `--cap-usd`, or accept it as a real data point about cost.
- **`LEAK AUDIT:` lines:** investigate before anything else. A hit may be a harmless coincidence
  (a reproducer reusing a hidden test's name) or a real leak; the artifacts will say which.
