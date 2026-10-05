"""Parquet export of the outcome matrix and the ledger, for analysis.

A straight copy: one parquet row per matrix row and per logged call, every field as stored, ordered
deterministically. Nothing is derived, filled, or edited here, so the parquet files are exactly as
trustworthy as the SQLite files they come from (CLAUDE.md hard rule 1).
"""

# pyright: basic
# (pyarrow ships no type stubs; this module only copies rows into it, so strict mode adds noise)
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from cascade.harness.ledger import CallRecord
from cascade.routing.matrix import MatrixRow, MatrixStore


def _write(rows: list[dict[str, Any]], columns: list[str], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows) if rows else pa.table({c: [] for c in columns})
    pq.write_table(
        table,
        path,
    )
    return len(rows)


def export_parquet(matrix_db: Path, ledger_db: Path, out_dir: Path) -> dict[str, int]:
    """Write matrix.parquet and ledger.parquet into ``out_dir``; return their row counts."""
    counts: dict[str, int] = {}
    if matrix_db.exists():
        store = MatrixStore(matrix_db)
        try:
            rows = [r.model_dump() for r in store.rows()]
        finally:
            store.close()
    else:
        rows = []
    counts["matrix"] = _write(rows, list(MatrixRow.model_fields), out_dir / "matrix.parquet")

    calls: list[dict[str, Any]] = []
    if ledger_db.exists():
        conn = sqlite3.connect(f"file:{ledger_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute("SELECT * FROM calls ORDER BY rowid")
            calls = [CallRecord.model_validate(dict(r)).model_dump() for r in cur.fetchall()]
        finally:
            conn.close()
    counts["ledger"] = _write(calls, list(CallRecord.model_fields), out_dir / "ledger.parquet")
    return counts
