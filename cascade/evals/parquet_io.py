"""The only place the evals touch pyarrow, kept tiny because pyarrow ships no type stubs.

Reading is column-selective and filter-pushed: ``read_rows`` hands back ONLY the rows whose task id
is in the allowed set, so a dev run that is allowed to see train tasks never materializes a test
row's outcome in memory. ``file_sha256`` hashes the raw bytes without parsing anything.
"""

# pyright: basic
from __future__ import annotations

import hashlib
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_column(path: Path, name: str) -> list[Any]:
    """One whole column (e.g. processing_tier), with no other field of any row."""
    return pq.read_table(path, columns=[name]).column(name).to_pylist()


def read_rows(path: Path, task_ids: Collection[str]) -> list[dict[str, Any]]:
    """Rows whose task_id is in ``task_ids``; every other row stays on disk."""
    ids = sorted(task_ids)
    if not ids:
        return []
    return pq.read_table(path, filters=[("task_id", "in", ids)]).to_pylist()


def row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)
