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
