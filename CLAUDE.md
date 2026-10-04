# Cascade
Turns GitHub issues into verified pull requests with a team of agents, and learns which model and reasoning effort to use per issue. Thesis: a cost-aware router can match an always-max-effort baseline's resolve rate at materially lower cost per resolved issue. A null result is acceptable. A fabricated or leaky result is not.

## Hard rules
1. Never fabricate or hand-edit results. Every reported metric is computed from logged runs. Synthetic test data is labeled synthetic and lives in a separate directory.
2. No real LLM spend unless env CASCADE_ALLOW_SPEND=1 and the budget ledger permits it. Default provider is mock.
3. No leakage. Agents never see the gold patch, test patch, hidden tests, or git history after the base commit. Agent sandboxes run with the network off.
4. Train/test split is frozen by hash and by repo. The test split is never used for fitting, tuning, or selection.
5. Never hardcode model names or prices. They live in configs/models.yaml, filled in by the human.
6. Never open PRs on third-party repos. Produce draft patches and PR text for human review, with an AI-assistance disclosure.
7. Secrets from env vars only. Commit .env.example, never .env.
8. Check any framework API against the installed version or docs before using it. Pin versions.
9. Long-running jobs are resumable and idempotent.

## Tech
Python 3.11+, uv, ruff, pyright, pytest, pydantic v2, typer, Docker, SQLite (parquet export for analysis). Orchestration: OpenAI Agents SDK with function tools. MCP (GitHub) only in the live pilot. Write local JSONL traces too. Prefer deterministic tools (ripgrep/BM25, tree-sitter, pytest parsers) to shrink prompts.

## Scope
In: Tier 1 (40+ audited tasks, 3-arm outcome matrix, fixed/cascade/oracle policies, CI-backed comparison, honest README) and Tier 2 (predictive router with leave-one-repo-out CV, role ablations, Verifier false-positive analysis, draft-only live pilot on 5-10 issues).
Out unless the human asks: per-role routing, local open-weight arm, public dataset/blog, polished dashboard.

## Protocol
At the start of every session read STATUS.md, NEXT.md, docs/SPEC.md. At every STOP update STATUS.md (done, metrics, spend, risks) and NEXT.md (exact human commands, decisions needed). Log non-obvious choices in DECISIONS.md. Commit often. Halt at each STOP.
