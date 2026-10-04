# Cascade

Turns GitHub issues into verified pull requests with a team of agents, and learns which model and
reasoning effort to use per issue. The thesis is that a cost-aware router can match an
always-max-effort baseline's resolve rate at materially lower cost per resolved issue. A null
result is acceptable; a fabricated or leaky one is not.

**Status: foundation only.** There are no experiments and no results yet. See `STATUS.md` for what
exists and `NEXT.md` for what is needed from a human before the next stage. Project rules are in
`CLAUDE.md`; the design is in `docs/SPEC.md`; non-obvious choices are in `DECISIONS.md`.

## Quickstart

```bash
uv sync
make check   # ruff, pyright, pytest
make smoke   # end-to-end on the synthetic fixture with the mock provider ($0)
uv run cascade config-status   # list config fields you still need to fill
```

The default provider is `mock`. Real spend requires `CASCADE_ALLOW_SPEND=1` and a filled
`configs/budget.yaml`. Copy `.env.example` to `.env` for local settings; never commit `.env`.
