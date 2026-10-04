"""SQLite budget ledger: a row per LLM call, with global/task/attempt caps and a kill switch.

Contract:
- ``authorize`` runs BEFORE a call and raises if the worst-case cost or tokens of that call would
  breach any cap, if the kill-switch file exists, or if a real provider is used without the spend
  gate. It uses worst-case figures (estimated input + max_output_tokens at the highest output-side
  price), so a call that passes cannot overshoot a cap on its own.
- ``record`` runs AFTER a call and logs what actually happened (cost from models.yaml prices). It
  never silently drops or merges rows; a repeated call_id raises DuplicateCallError, because a
  repeated real call would otherwise be invisible to the caps.
- Construction needs a fully resolved budget and models config. TODO prices or caps fail closed.

Known limit: authorize/record are not one atomic reservation, so N parallel workers can overshoot
a cap by up to N in-flight worst-case calls. Run with low parallelism or leave headroom in caps.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from cascade.config import ModelsConfig, ResolvedBudget
from cascade.errors import BudgetExceededError, DuplicateCallError, KillSwitchEngagedError
from cascade.harness.providers.base import Usage
from cascade.harness.spend import require_spend_allowed

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    call_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    arm_id TEXT NOT NULL,
    role TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    reasoning_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_s REAL NOT NULL,
    status TEXT NOT NULL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_task ON calls(task_id);
CREATE INDEX IF NOT EXISTS idx_calls_attempt ON calls(task_id, attempt_id);
"""

_EPS = 1e-12


@dataclass(frozen=True)
class CallContext:
    """Identifies one LLM call. ``call_id`` is the idempotency key and must be deterministic
    (derive it from run/task/arm/attempt/role/turn) so a resumed job reproduces the same ids."""

    call_id: str
    run_id: str
    task_id: str
    attempt_id: str
    arm_id: str
    role: str


@dataclass(frozen=True)
class AttemptTotals:
    calls: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost_usd: float
    latency_s: float


class CallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str
    ts: str
    run_id: str
    task_id: str
    attempt_id: str
    arm_id: str
    role: str
    provider: str
    model: str
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    reasoning_tokens: Annotated[int, Field(ge=0)]
    cost_usd: Annotated[float, Field(ge=0)]
    latency_s: Annotated[float, Field(ge=0)]
    status: str
    error: str | None = None


class Ledger:
    def __init__(
        self, db_path: Path | str, *, budget: ResolvedBudget, models: ModelsConfig
    ) -> None:
        self._budget = budget
        self._models = models
        # The Agents SDK runs coroutines on the calling thread, but allow cross-thread use anyway.
        self._conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # --- pricing -------------------------------------------------------------------------------

    def cost_of(self, model: str, usage: Usage) -> float:
        p = self._models.prices_for(model)
        return (
            usage.input_tokens * p.input_usd_per_mtok
            + usage.output_tokens * p.output_usd_per_mtok
            + usage.reasoning_tokens * p.reasoning_usd_per_mtok
        ) / 1e6

    def worst_case_cost(self, model: str, est_input_tokens: int, max_output_tokens: int) -> float:
        p = self._models.prices_for(model)
        out_price = max(p.output_usd_per_mtok, p.reasoning_usd_per_mtok)
        return (est_input_tokens * p.input_usd_per_mtok + max_output_tokens * out_price) / 1e6

    # --- pre-call gate -------------------------------------------------------------------------

    def authorize(
        self,
        ctx: CallContext,
        *,
        model: str,
        est_input_tokens: int,
        max_output_tokens: int,
        real: bool,
    ) -> None:
        if self._budget.kill_switch_path.exists():
            raise KillSwitchEngagedError(f"kill switch present at {self._budget.kill_switch_path}")
        if real:
            require_spend_allowed(f"real call to {model}")

        worst = self.worst_case_cost(model, est_input_tokens, max_output_tokens)
        scopes = (
            ("global", self.spent_usd(), self._budget.global_cap_usd),
            ("task", self.spent_usd(task_id=ctx.task_id), self._budget.per_task_cap_usd),
            (
                "attempt",
                self.spent_usd(task_id=ctx.task_id, attempt_id=ctx.attempt_id),
                self._budget.per_attempt_cap_usd,
            ),
        )
        for scope, spent, cap in scopes:
            if spent + worst > cap + _EPS:
                raise BudgetExceededError(
                    f"{scope} cap: spent ${spent:.6f} + worst-case ${worst:.6f} > cap ${cap:.6f}"
                )

        used = self.tokens_used(task_id=ctx.task_id, attempt_id=ctx.attempt_id)
        if used + est_input_tokens + max_output_tokens > self._budget.per_attempt_max_tokens:
            raise BudgetExceededError(
                f"attempt token cap: used {used} + requested "
                f"{est_input_tokens + max_output_tokens} > {self._budget.per_attempt_max_tokens}"
            )

    # --- post-call record ----------------------------------------------------------------------

    def has_call(self, call_id: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM calls WHERE call_id = ?", (call_id,)).fetchone()
        return row is not None

    def record(
        self,
        ctx: CallContext,
        *,
        provider: str,
        model: str,
        usage: Usage,
        latency_s: float,
        status: str = "ok",
        error: str | None = None,
    ) -> CallRecord:
        rec = CallRecord(
            call_id=ctx.call_id,
            ts=dt.datetime.now(dt.UTC).isoformat(),
            run_id=ctx.run_id,
            task_id=ctx.task_id,
            attempt_id=ctx.attempt_id,
            arm_id=ctx.arm_id,
            role=ctx.role,
            provider=provider,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cost_usd=self.cost_of(model, usage),
            latency_s=latency_s,
            status=status,
            error=error,
        )
        try:
            self._conn.execute(
                "INSERT INTO calls VALUES (:call_id, :ts, :run_id, :task_id, :attempt_id, :arm_id,"
                " :role, :provider, :model, :input_tokens, :output_tokens, :reasoning_tokens,"
                " :cost_usd, :latency_s, :status, :error)",
                rec.model_dump(),
            )
        except sqlite3.IntegrityError as e:
            raise DuplicateCallError(f"call_id already recorded: {ctx.call_id}") from e
        return rec

    # --- queries -------------------------------------------------------------------------------

    def spent_usd(self, *, task_id: str | None = None, attempt_id: str | None = None) -> float:
        if task_id is None and attempt_id is not None:
            raise ValueError("attempt_id requires task_id")
        if task_id is None:
            row = self._conn.execute("SELECT COALESCE(SUM(cost_usd), 0.0) FROM calls").fetchone()
        elif attempt_id is None:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM calls WHERE task_id = ?", (task_id,)
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM calls"
                " WHERE task_id = ? AND attempt_id = ?",
                (task_id, attempt_id),
            ).fetchone()
        return float(row[0])

    def tokens_used(self, *, task_id: str, attempt_id: str) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(input_tokens + output_tokens + reasoning_tokens), 0) FROM calls"
            " WHERE task_id = ? AND attempt_id = ?",
            (task_id, attempt_id),
        ).fetchone()
        return int(row[0])

    def call_count(self, *, task_id: str, attempt_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM calls WHERE task_id = ? AND attempt_id = ?", (task_id, attempt_id)
        ).fetchone()
        return int(row[0])

    def attempt_ids(self, *, task_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT attempt_id FROM calls WHERE task_id = ? ORDER BY attempt_id",
            (task_id,),
        )
        return [str(r[0]) for r in rows.fetchall()]

    def attempt_totals(self, *, task_id: str, attempt_id: str) -> AttemptTotals:
        """Everything reported about an attempt's cost comes from here: sums over logged calls."""
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0),"
            " COALESCE(SUM(reasoning_tokens), 0), COALESCE(SUM(cost_usd), 0.0),"
            " COALESCE(SUM(latency_s), 0.0) FROM calls WHERE task_id = ? AND attempt_id = ?",
            (task_id, attempt_id),
        ).fetchone()
        return AttemptTotals(
            calls=int(row[0]),
            input_tokens=int(row[1]),
            output_tokens=int(row[2]),
            reasoning_tokens=int(row[3]),
            cost_usd=float(row[4]),
            latency_s=float(row[5]),
        )

    def rows(self) -> list[CallRecord]:
        cur = self._conn.execute("SELECT * FROM calls ORDER BY rowid")
        out: list[CallRecord] = []
        for row in cur.fetchall():
            data: dict[str, Any] = dict(row)
            out.append(CallRecord.model_validate(data))
        return out


@dataclass(frozen=True)
class LedgerSummary:
    calls: int
    errors: int
    cost_usd: float
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    by_arm: list[tuple[str, int, float]]  # (arm_id, calls, cost_usd)
    by_role: list[tuple[str, int, float]]  # (role, calls, cost_usd)
    error_rows: list[tuple[str, str, str, str]]  # (task_id, arm_id, role, error)


def summarize(db_path: Path | str) -> LedgerSummary:
    """Summarize a ledger file with no config needed. Read-only."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        total = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(status = 'error'), 0), COALESCE(SUM(cost_usd), 0.0),"
            " COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0),"
            " COALESCE(SUM(reasoning_tokens), 0) FROM calls"
        ).fetchone()
        by_arm = conn.execute(
            "SELECT arm_id, COUNT(*), SUM(cost_usd) FROM calls GROUP BY arm_id ORDER BY arm_id"
        ).fetchall()
        by_role = conn.execute(
            "SELECT role, COUNT(*), SUM(cost_usd) FROM calls GROUP BY role ORDER BY role"
        ).fetchall()
        errors = conn.execute(
            "SELECT task_id, arm_id, role, COALESCE(error, '') FROM calls"
            " WHERE status = 'error' ORDER BY rowid"
        ).fetchall()
    finally:
        conn.close()
    return LedgerSummary(
        calls=int(total[0]),
        errors=int(total[1]),
        cost_usd=float(total[2]),
        input_tokens=int(total[3]),
        output_tokens=int(total[4]),
        reasoning_tokens=int(total[5]),
        by_arm=[(str(a), int(n), float(c)) for a, n, c in by_arm],
        by_role=[(str(r), int(n), float(c)) for r, n, c in by_role],
        error_rows=[(str(t), str(a), str(r), str(e)) for t, a, r, e in errors],
    )
