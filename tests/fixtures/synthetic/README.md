# SYNTHETIC test data

Not real. Mock model names, invented prices, an invented budget, and one invented task. Used only
by the test suite. Never use these values for experiments or reported numbers.

- `models.yaml`, `budget.yaml`: configs with all fields filled so the ledger can run offline.
- `task.json`: a task for `tiny_repo`. Its `base_commit` is a placeholder; tests replace it with
  the sha of the temp git repo they build.
- `gold.patch`, `test.patch`: the source fix and the added hidden test for `tiny_repo`. They are
  also embedded in `task.json` and are kept in sync by `tests/test_task_schema.py`.
