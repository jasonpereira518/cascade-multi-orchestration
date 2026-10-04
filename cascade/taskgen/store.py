"""SQLite state for the task pipeline. Every stage result is persisted so reruns skip finished work.

A row in ``stage_results`` is keyed by (pr_number, stage). Outcomes:
- "ok":      the PR passed this stage; ``data`` holds what later stages need.
- "dropped": the PR was rejected, with a machine-readable ``reason``. Final; never re-evaluated.
- "error":   something transient or environmental failed (docker, network). Retried on rerun.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any

OK = "ok"
DROPPED = "dropped"
ERROR = "error"
_FINAL = (OK, DROPPED)

# Stage names, in pipeline order. "mine" is implicit: a PR exists once it is in `candidates`.
FILTER = "filter"
PATCH = "patch"
ENV = "env"
VALIDATE = "validate"
STAGES = (FILTER, PATCH, ENV, VALIDATE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    pr_number INTEGER PRIMARY KEY,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stage_results (
    pr_number INTEGER NOT NULL,
    stage TEXT NOT NULL,
    outcome TEXT NOT NULL,
    reason TEXT,
    data TEXT,
    ts TEXT NOT NULL,
    PRIMARY KEY (pr_number, stage)
);
CREATE TABLE IF NOT EXISTS mine_slices (
    slice_start TEXT NOT NULL,
    slice_end TEXT NOT NULL,
    PRIMARY KEY (slice_start, slice_end)
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class StageStore:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # --- candidates ----------------------------------------------------------------------------

    def upsert_candidate(self, pr_number: int, data: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO candidates(pr_number, data) VALUES (?, ?)"
            " ON CONFLICT(pr_number) DO UPDATE SET data = excluded.data",
            (pr_number, json.dumps(data, sort_keys=True)),
        )

    def candidates(self) -> list[tuple[int, dict[str, Any]]]:
        rows = self._conn.execute("SELECT pr_number, data FROM candidates ORDER BY pr_number")
        return [(int(r["pr_number"]), json.loads(r["data"])) for r in rows.fetchall()]

    def candidate_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])

    # --- stage results -------------------------------------------------------------------------

    def put_stage(
        self,
        pr_number: int,
        stage: str,
        outcome: str,
        reason: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO stage_results(pr_number, stage, outcome, reason, data, ts)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(pr_number, stage) DO UPDATE SET outcome = excluded.outcome,"
            " reason = excluded.reason, data = excluded.data, ts = excluded.ts",
            (
                pr_number,
                stage,
                outcome,
                reason,
                None if data is None else json.dumps(data, sort_keys=True),
                dt.datetime.now(dt.UTC).isoformat(),
            ),
        )

    def get_stage(
        self, pr_number: int, stage: str
    ) -> tuple[str, str | None, dict[str, Any]] | None:
        row = self._conn.execute(
            "SELECT outcome, reason, data FROM stage_results WHERE pr_number = ? AND stage = ?",
            (pr_number, stage),
        ).fetchone()
        if row is None:
            return None
        data: dict[str, Any] = {} if row["data"] is None else json.loads(row["data"])
        return str(row["outcome"]), row["reason"], data

    def is_final(self, pr_number: int, stage: str) -> bool:
        got = self.get_stage(pr_number, stage)
        return got is not None and got[0] in _FINAL

    def passed(self, stage: str) -> list[int]:
        """PR numbers that finished ``stage`` with outcome ok, ascending."""
        rows = self._conn.execute(
            "SELECT pr_number FROM stage_results WHERE stage = ? AND outcome = ?"
            " ORDER BY pr_number",
            (stage, OK),
        )
        return [int(r["pr_number"]) for r in rows.fetchall()]

    def count(self, stage: str, outcome: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM stage_results WHERE stage = ? AND outcome = ?", (stage, outcome)
        ).fetchone()
        return int(row[0])

    def drop_reasons(self, stage: str) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT reason, COUNT(*) AS n FROM stage_results WHERE stage = ? AND outcome = ?"
            " GROUP BY reason ORDER BY n DESC, reason",
            (stage, DROPPED),
        )
        return {str(r["reason"]): int(r["n"]) for r in rows.fetchall()}

    # --- mining progress -----------------------------------------------------------------------

    def slice_done(self, start: dt.date, end: dt.date) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM mine_slices WHERE slice_start = ? AND slice_end = ?",
            (start.isoformat(), end.isoformat()),
        ).fetchone()
        return row is not None

    def mark_slice_done(self, start: dt.date, end: dt.date) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO mine_slices(slice_start, slice_end) VALUES (?, ?)",
            (start.isoformat(), end.isoformat()),
        )

    # --- key/value -----------------------------------------------------------------------------

    def get_kv(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_kv(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO kv(key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
