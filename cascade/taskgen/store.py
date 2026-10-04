"""SQLite state for the task pipeline. Every stage result is persisted so reruns skip finished work.

A row in ``stage_results`` is keyed by (pr_number, stage). Outcomes:
- "ok":      the PR passed this stage; ``data`` holds what later stages need.
- "dropped": the PR was rejected, with a machine-readable ``reason``. Final; never re-evaluated.
- "error":   something transient or environmental failed (docker, network). Retried on rerun.

One store is shared by the worker threads that validate PRs in parallel, so every statement runs
under a lock (SQLite connections are not safe for concurrent use).
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
from collections.abc import Sequence
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
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _run(self, sql: str, params: Sequence[Any] = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)

    def _all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    # --- candidates ----------------------------------------------------------------------------

    def upsert_candidate(self, pr_number: int, data: dict[str, Any]) -> None:
        self._run(
            "INSERT INTO candidates(pr_number, data) VALUES (?, ?)"
            " ON CONFLICT(pr_number) DO UPDATE SET data = excluded.data",
            (pr_number, json.dumps(data, sort_keys=True)),
        )

    def candidates(self) -> list[tuple[int, dict[str, Any]]]:
        rows = self._all("SELECT pr_number, data FROM candidates ORDER BY pr_number")
        return [(int(r["pr_number"]), json.loads(r["data"])) for r in rows]

    def candidate_count(self) -> int:
        row = self._one("SELECT COUNT(*) FROM candidates")
        return int(row[0]) if row else 0

    # --- stage results -------------------------------------------------------------------------

    def put_stage(
        self,
        pr_number: int,
        stage: str,
        outcome: str,
        reason: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self._run(
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
        row = self._one(
            "SELECT outcome, reason, data FROM stage_results WHERE pr_number = ? AND stage = ?",
            (pr_number, stage),
        )
        if row is None:
            return None
        data: dict[str, Any] = {} if row["data"] is None else json.loads(row["data"])
        return str(row["outcome"]), row["reason"], data

    def is_final(self, pr_number: int, stage: str) -> bool:
        got = self.get_stage(pr_number, stage)
        return got is not None and got[0] in _FINAL

    def passed(self, stage: str) -> list[int]:
        """PR numbers that finished ``stage`` with outcome ok, ascending."""
        rows = self._all(
            "SELECT pr_number FROM stage_results WHERE stage = ? AND outcome = ?"
            " ORDER BY pr_number",
            (stage, OK),
        )
        return [int(r["pr_number"]) for r in rows]

    def count(self, stage: str, outcome: str) -> int:
        row = self._one(
            "SELECT COUNT(*) FROM stage_results WHERE stage = ? AND outcome = ?", (stage, outcome)
        )
        return int(row[0]) if row else 0

    def drop_reasons(self, stage: str) -> dict[str, int]:
        rows = self._all(
            "SELECT reason, COUNT(*) AS n FROM stage_results WHERE stage = ? AND outcome = ?"
            " GROUP BY reason ORDER BY n DESC, reason",
            (stage, DROPPED),
        )
        return {str(r["reason"]): int(r["n"]) for r in rows}

    # --- mining progress -----------------------------------------------------------------------

    def slice_done(self, start: dt.date, end: dt.date) -> bool:
        row = self._one(
            "SELECT 1 FROM mine_slices WHERE slice_start = ? AND slice_end = ?",
            (start.isoformat(), end.isoformat()),
        )
        return row is not None

    def mark_slice_done(self, start: dt.date, end: dt.date) -> None:
        self._run(
            "INSERT OR IGNORE INTO mine_slices(slice_start, slice_end) VALUES (?, ?)",
            (start.isoformat(), end.isoformat()),
        )

    # --- key/value -----------------------------------------------------------------------------

    def get_kv(self, key: str) -> str | None:
        row = self._one("SELECT value FROM kv WHERE key = ?", (key,))
        return None if row is None else str(row["value"])

    def set_kv(self, key: str, value: str) -> None:
        self._run(
            "INSERT INTO kv(key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
