# Next (STOP C): audit the tasks

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

## Decisions I need from you

1. **The token and the date window** (part A). The window depends on the knowledge cutoff of your
   arms (`models.yaml`), so those three are linked.
2. **The split choice** (part B): accept a smaller test split, hold out click, or add repos.
3. **The audit** (part C): the only step that cannot be automated, and the one that decides what
   enters the experiment.
4. Everything under "STOP B" below is still open and is **independent of this audit**: the one cheap
   arm, the budget, and the first paid run of `cascade run`. You can do it before or after.

---

# STOP B instructions (still open, unchanged)

Part 1 of the old list ("get about 5 validated tasks") is **replaced by part A above**; use
`cascade taskgen run-all` and take the first 5 tasks from the manifest. Parts 2 to 6 stand as written.
After the first paid results, run `uv run cascade triage` and paste its output too: it classifies each
failed attempt as a pipeline bug, a Verifier issue, or a genuine model failure, which is what the
phase 3 brief asked me to do first and could not, because no real run exists yet.

## Part 2. Configure exactly one cheap arm

Replace `configs/models.yaml` with a single model and a single arm. The angle-bracket values are
yours to supply from the provider's docs and pricing page; I have not filled any (hard rule 5).

```yaml
models:
  - name: <model id exactly as the provider lists it>
    provider: openai
    input_usd_per_mtok: <number>
    output_usd_per_mtok: <number>
    reasoning_usd_per_mtok: null        # null unless reasoning tokens bill differently
    knowledge_cutoff: <YYYY-MM-DD>
    reasoning_efforts: [<effort value, e.g. low>]
arms:
  - {model: <same model id>, reasoning_effort: <same effort>}
```

If the model has no effort setting, use `reasoning_efforts: []` and drop `reasoning_effort` from the
arm. The arm id you will pass to `--arm` is `<model id>@<effort>` (or `<model id>@default`).

## Part 3. Set a small budget

Edit `configs/budget.yaml`. These are **my suggested starting points, a judgment call and not a
measurement** (real cost per attempt is unknown, see `STATUS.md` risk 2). Lower them if you like:

```yaml
global_cap_usd: 5.00          # the whole experiment can never exceed this
per_task_cap_usd: 1.00
per_attempt_cap_usd: 1.00
per_attempt_max_tokens: 1000000   # input+output+reasoning, summed over an attempt's calls
kill_switch_path: KILL_SWITCH
```

Caps must satisfy per_attempt <= per_task <= global. The per-call check is deliberately pessimistic
(it reserves `--max-output-tokens` at your output price before every call), so a cap set too tight
refuses calls that would have fit. To stop everything at any moment: `touch KILL_SWITCH`.

```bash
uv run cascade config-status    # models.yaml and budget.yaml should show no lines
```

(`repos.yaml` lines may remain until part 1's date window is set.)

## Part 4. Rehearse for $0 with the mock provider

This uses real Docker, your real task, and your real prices, but no model: the mock fixes nothing,
so expect `true_success=False`. It proves the plumbing on a real task before you spend anything.

```bash
TASK=$(uv run python -c "import json; print(json.loads(open('tasks/manifest.jsonl').readline())['task_id'])")
ARM='<model id>@<effort>'
uv run cascade run --task "$TASK" --arm "$ARM" --max-output-tokens 8000
uv run cascade ledger-summary
```

It finishes in seconds with `calls 6`, `true_success=False`, and a tiny synthetic cost: it proves your
config, Docker, the clone, workspace export, ledger and matrix on a real task. It does not run click's
tests (an empty patch short-circuits that), so the first paid run is also the first real test run.

## Part 5. The paid run: ONE task first, then the rest

```bash
export OPENAI_API_KEY=<your key>
export CASCADE_ALLOW_SPEND=1

# one task, per-attempt cap of $0.50 (the flag can only lower budget.yaml's cap)
uv run cascade run --task "$TASK" --arm "$ARM" --provider openai --cap-usd 0.50 --max-output-tokens 8000
```

Look at the output before continuing. It prints `true_success`, `verifier_claimed`, cost, calls,
and the artifacts directory. Then compare `ledger total spent so far` with your provider's usage
dashboard: they should agree closely. If they do not, stop and tell me.

If that looks sane, run the rest (up to 5 tasks in total):

```bash
for T in $(uv run python -c "import json; [print(json.loads(l)['task_id']) for l in list(open('tasks/manifest.jsonl'))[:5]]"); do
  uv run cascade run --task "$T" --arm "$ARM" --provider openai --cap-usd 0.50 --max-output-tokens 8000
done
```

Rerunning is safe: a task that already has a result is skipped (`--force` reruns it).

## Part 6. Paste back

1. The output of:

```bash
uv run cascade ledger-summary
```

2. Any line starting with `error:`, `attempt ended early:`, `grading:`, or `LEAK AUDIT:` from the runs.
3. For each attempt that failed in a way you do not understand, the contents of its
   `result.json` (the path is printed as `artifacts:`). The full local trace is next to it in
   `trace.jsonl`; do not paste that unless I ask, since it holds every prompt.

## What I expect to go wrong first (so you recognize it)

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

## Decisions I need from you (STOP B)

1. **The cheap arm:** which model and effort (part 2).
2. **Budget numbers** (part 3), if my suggestions are not right for you.
3. **Part 1 still open:** token choice and date window. Your four extra repos are configured (jinja is
   dormant, so skip it). Over 12 months they bracket about 52 candidates; for Tier 1's 40+ a window of
   18 months or more gives margin, if your model's knowledge cutoff allows it.
4. After seeing the paid results: do you want me to proceed to the 3-arm matrix, or tune prompts first?
   I will recommend based on what the failures look like.
5. Approval for the Phase 3 scope when you are ready: per-role ablations and the policy code
   (fixed, cascade, oracle) come after the matrix exists.
