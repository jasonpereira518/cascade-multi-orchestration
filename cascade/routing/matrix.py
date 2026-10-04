"""The outcome matrix: one row per (task, arm, seed, pipeline config), both outcomes recorded.

Row fields are those in docs/SPEC.md. Everything numeric is computed from the logged ledger calls of
the attempt (rule 1: no hand-entered or estimated metrics). ``true_success`` is graded against the
hidden tests; ``verifier_claimed`` is the agent's own evidence. Keeping both per attempt is what
allows Verifier false-positive analysis and the cascade policy later.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class MatrixRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    arm_id: str
    seed: int
    true_success: bool
    verifier_claimed: bool
    cost_usd: Annotated[float, Field(ge=0)]
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    reasoning_tokens: Annotated[int, Field(ge=0)]
    wall_s: Annotated[float, Field(ge=0)]
    turns: Annotated[int, Field(ge=0)]  # LLM calls made
    patch_sha: str | None
    pipeline_config: str  # canonical JSON of the PipelineConfig
    error: str | None

    @property
    def config_id(self) -> str:
        return config_id_of(self.pipeline_config)


def config_id_of(pipeline_config: str) -> str:
    return hashlib.sha256(pipeline_config.encode()).hexdigest()[:10]


class MatrixStore:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS matrix (task_id TEXT NOT NULL, arm_id TEXT NOT NULL,"
            " seed INTEGER NOT NULL, config_id TEXT NOT NULL, row TEXT NOT NULL,"
            " PRIMARY KEY (task_id, arm_id, seed, config_id))"
        )

    def close(self) -> None:
        self._conn.close()

    def put(self, row: MatrixRow) -> None:
        self._conn.execute(
            "INSERT INTO matrix(task_id, arm_id, seed, config_id, row) VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(task_id, arm_id, seed, config_id) DO UPDATE SET row = excluded.row",
            (row.task_id, row.arm_id, row.seed, row.config_id, row.model_dump_json()),
        )

    def get(self, task_id: str, arm_id: str, seed: int, config_id: str) -> MatrixRow | None:
        found = self._conn.execute(
            "SELECT row FROM matrix"
            " WHERE task_id = ? AND arm_id = ? AND seed = ? AND config_id = ?",
            (task_id, arm_id, seed, config_id),
        ).fetchone()
        return None if found is None else MatrixRow.model_validate_json(found["row"])

    def rows(self) -> list[MatrixRow]:
        cur = self._conn.execute("SELECT row FROM matrix ORDER BY task_id, arm_id, seed, config_id")
        return [MatrixRow.model_validate_json(r["row"]) for r in cur.fetchall()]
