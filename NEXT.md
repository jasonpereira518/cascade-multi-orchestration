# Next (STOP B)

Goal: run `cascade run` on about 5 real tasks with **one cheap arm** under a **small cap**, then paste
back the ledger summary and any failures. Nothing in this file spends money until step 5.

You need validated tasks first. None exist yet, so part 1 comes first.

## Part 1. Get about 5 validated tasks (the open STOP A items)

Unchanged from before. In short: a GitHub token, the date window, Docker running, then the pipeline.

```bash
export GITHUB_TOKEN=<fine-grained token, public repositories read-only>
# edit configs/repos.yaml: set date_window.start and date_window.end (YYYY-MM-DD)

REPOS="pallets/click python-attrs/attrs hynek/structlog marshmallow-code/marshmallow"   # not jinja: dormant
for R in $REPOS; do uv run cascade taskgen run --repo "$R" --stop-after patch; done   # token only, no Docker
for R in $REPOS; do uv run cascade taskgen run --repo "$R"; done                      # builds images, validates
grep -c . tasks/manifest.jsonl                                                       # how many tasks you got
```

If you get fewer than 5, run on what you have. `cascade taskgen attrition --repo <owner/name>` shows
where each repo's candidates dropped. Choose the date window after you know your model's knowledge cutoff
(part 2), because the window must start after it.

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

## Decisions I need from you

1. **The cheap arm:** which model and effort (part 2).
2. **Budget numbers** (part 3), if my suggestions are not right for you.
3. **Part 1 still open:** token choice and date window. Your four extra repos are configured (jinja is
   dormant, so skip it). Over 12 months they bracket about 52 candidates; for Tier 1's 40+ a window of
   18 months or more gives margin, if your model's knowledge cutoff allows it.
4. After seeing the paid results: do you want me to proceed to the 3-arm matrix, or tune prompts first?
   I will recommend based on what the failures look like.
5. Approval for the Phase 3 scope when you are ready: per-role ablations and the policy code
   (fixed, cascade, oracle) come after the matrix exists.
