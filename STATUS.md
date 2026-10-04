# Status

**Stage:** STOP 1 (foundation) complete. Halted, waiting on the human (see `NEXT.md`).
**Updated:** 2026-10-04

## Done

- `CLAUDE.md` and `docs/SPEC.md` written verbatim from the brief.
- Scaffold: package layout per SPEC, `pyproject.toml` with exact-pinned deps and a committed
  `uv.lock`, ruff + pyright (strict on `cascade/`) config, `Makefile`
  (`check`, `test`, `smoke`, `docker-test`), GitHub Actions CI, `.env.example`, `.gitignore`.
- Config schemas (pydantic v2) and `configs/{models,budget,repos}.yaml` with TODO fields.
  `uv run cascade config-status` lists what is unfilled. Unfilled prices or caps fail closed.
- `harness/ledger.py`: SQLite call ledger, global/per-task/per-attempt dollar caps, per-attempt
  token cap, kill-switch file, duplicate-call refusal, costs computed from `models.yaml`.
- `harness/providers`: provider interface, scriptable mock provider (the default), OpenAI stub that
  refuses without `CASCADE_ALLOW_SPEND=1` at construction and on every call. `MeteredClient` routes
  every call through the ledger and refuses real providers unless the spend gate is open.
- `harness/sandbox.py`: workspace export from the base commit only (`git archive`, no `.git`),
  an always-on leak guard, and a Docker runner with network off, dropped capabilities, read-only
  root, resource limits, and an empty environment.
- Task schema (`taskgen/schema.py`) with an `AgentView` that structurally cannot carry gold data.
- Synthetic fixture (`tests/fixtures/`): a toy repo with one bug, a gold patch, a test patch.
- `make smoke`: builds a git repo from the fixture (with a later commit holding the gold and test
  patches), exports the base-commit workspace, runs the leak guard, runs the pipeline stub on the
  mock provider, and checks the single ledger row, its cost, and that the prompt held no gold data.

## Verified

| Check | Result |
|---|---|
| `make check` (ruff, ruff format, pyright strict, pytest) | exit 0; 82 passed, 1 skipped |
| `make smoke` | exit 0; 1 passed |
| `uv lock --check` | lockfile current |
| Guard mutation check (8 deliberate breakages on a scratch copy) | all 8 caught by the suite |

The mutation check was a one-off script run on a scratch copy. It is not committed and not in CI.

## Metrics

None. No experiments have run, so there are no results to report. Nothing in the repo may be cited
as a result: every number computed from `tests/fixtures/` is synthetic.

## Spend

**$0.** No real provider call has ever been made (the OpenAI adapter does not exist yet, only the
gate). Ledger totals in tests are synthetic mock costs.

## Risks and unverified items

1. **Docker was not running on this machine, so nothing touched a real container.** The docker
   command line is covered by tests on the constructed argv and the mount set, and the real-daemon
   test (`make docker-test`) was skipped. Flags such as `--read-only` with a tmpfs `/tmp`, and
   `--user` against Docker Desktop bind mounts, are unproven until that test runs. See `NEXT.md`.
2. **CI has never run.** The workflow was written against action versions read from the GitHub API,
   but there is no remote yet, so it is untested.
3. **Leak guard limits.** It catches added paths and whole-file copies of the patches, not an
   excerpt pasted into an existing file. The structural defence (only the base-commit tree is
   mounted) covers that. Detail in `DECISIONS.md` #8.
4. **Ledger overshoot under parallelism.** authorize and record are not atomic; parallel workers
   can overshoot a cap by their in-flight calls (`DECISIONS.md` #5).
5. **No grading sandbox yet.** Hidden-test execution needs its own runner (`DECISIONS.md` #9).
6. **No OpenAI adapter, Agents SDK, or MCP wiring.** Intentionally deferred; versions get pinned
   and API usage checked against the installed package when they land.
7. **Config is unfilled by design.** Until the human fills `models.yaml` and `budget.yaml`, the
   ledger refuses to start for any real arm.
