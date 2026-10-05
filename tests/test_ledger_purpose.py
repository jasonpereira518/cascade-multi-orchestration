"""The ledger's `purpose` column: default, live-pilot tagging, in-place migration, parquet export.

The `live_pilot` bucket cap itself is tested end to end in tests/test_pilot_draft.py.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pyarrow.parquet as pq  # pyright: ignore[reportMissingTypeStubs]

from cascade.harness.ledger import (
    EXPERIMENT_PURPOSE,
    LIVE_PILOT_PURPOSE,
    CallContext,
    Ledger,
)
from cascade.harness.providers.base import Usage
from cascade.routing.export import export_parquet
from tests.matrix_helpers import CHEAP, matrix_budget, matrix_models, new_ledger


def call(ledger: Ledger, call_id: str, **kw: str) -> None:
    ctx = CallContext(call_id, "run", "t1", "a1", CHEAP, "patcher", **kw)
    ledger.record(
        ctx, provider="mock", model="mock-luna", usage=Usage(input_tokens=100, output_tokens=10),
        latency_s=0.0,
    )  # fmt: skip


def test_a_call_is_an_experiment_call_unless_tagged(tmp_path: Path) -> None:
    ledger = new_ledger(tmp_path)
    call(ledger, "c1")
    call(ledger, "c2", bucket="live_pilot", purpose=LIVE_PILOT_PURPOSE)
    assert [(r.call_id, r.purpose) for r in ledger.rows()] == [
        ("c1", EXPERIMENT_PURPOSE),
        ("c2", LIVE_PILOT_PURPOSE),
    ]
    assert (EXPERIMENT_PURPOSE, LIVE_PILOT_PURPOSE) == ("experiment", "live_pilot")


def test_an_older_ledger_file_is_migrated_in_place(tmp_path: Path) -> None:
    db = tmp_path / "old.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE calls (call_id TEXT PRIMARY KEY, ts TEXT NOT NULL, run_id TEXT NOT NULL,"
        " task_id TEXT NOT NULL, attempt_id TEXT NOT NULL, arm_id TEXT NOT NULL,"
        " role TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,"
        " input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL,"
        " reasoning_tokens INTEGER NOT NULL, cost_usd REAL NOT NULL, latency_s REAL NOT NULL,"
        " status TEXT NOT NULL, error TEXT)"
    )
    conn.execute(
        "INSERT INTO calls VALUES ('old', 't', 'r', 't1', 'a1', 'arm', 'x', 'mock', 'mock-luna',"
        " 1, 1, 0, 0.5, 0.0, 'ok', NULL)"
    )
    conn.commit()
    conn.close()
    ledger = Ledger(db, budget=matrix_budget(tmp_path), models=matrix_models())
    call(ledger, "new", purpose=LIVE_PILOT_PURPOSE)
    assert {r.call_id: r.purpose for r in ledger.rows()} == {
        "old": EXPERIMENT_PURPOSE,
        "new": LIVE_PILOT_PURPOSE,
    }
    ledger.close()


def test_the_parquet_export_carries_the_purpose_column(tmp_path: Path) -> None:
    ledger = new_ledger(tmp_path)
    call(ledger, "c1")
    call(ledger, "c2", bucket="live_pilot", purpose=LIVE_PILOT_PURPOSE)
    ledger.close()
    export_parquet(tmp_path / "none.sqlite", tmp_path / "ledger.sqlite", tmp_path / "out")
    table = pq.read_table(tmp_path / "out" / "ledger.parquet")  # pyright: ignore[reportUnknownMemberType]
    assert sorted(table.column("purpose").to_pylist()) == ["experiment", "live_pilot"]
