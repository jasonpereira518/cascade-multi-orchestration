# Test fixtures

Everything under this directory is **synthetic test data** (hard rule 1). None of it is a real
task, model, price, or result, and no metric may ever be computed from it.

- `synthetic/`: mock model identifiers with made-up prices, a made-up budget, and a hand-built
  task with its gold and test patches.
- `tiny_repo/`: the base-commit tree of a toy library with one bug. The smoke test turns it into a
  git repository in a temp directory, adds a later commit containing the gold and test patches,
  and checks that none of that later content reaches the agent workspace.

The directory is excluded from pytest collection, ruff, and pyright.
