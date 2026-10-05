"""`cascade stats`: the numbers a write-up needs, every one read from an artifact.

Sources: the manifest and audit log (tasks, repos, split), the ledger (tokens and spend, by arm and
by budget bucket), the matrix (attempts), models.yaml (arms), and ``pytest --collect-only`` (the
number of tests in this repository, counted by running pytest, not remembered). Nothing is cached or
typed in, so the output is safe to paste into a README or a resume.
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from cascade.config import ModelsConfig
from cascade.routing.matrix import MatrixRow
from cascade.taskgen.audit import AuditVerdict, experiment_tasks
from cascade.taskgen.schema import Task

CollectRunner = Callable[[Path], str]
_COLLECTED = re.compile(r"(\d+) tests? collected|collected (\d+) items?")


@dataclass(frozen=True)
class Stats:
    repos: list[str]
    audited_ok: int
    audited_train: int
    audited_test: int
    ok_by_repo: dict[str, int]
    arms: list[str]
    attempts: int
    ledger_calls: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    spend_total: float
    spend_by_arm: dict[str, float]
    spend_by_bucket: dict[str, float]
    tests_in_repo: int | None


def pytest_collect(cwd: Path) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=cwd, capture_output=True, text=True, check=False,
    )  # fmt: skip
    return proc.stdout


def count_tests(output: str) -> int | None:
    """Parse the total from ``pytest --collect-only -q`` output (e.g. '609 tests collected')."""
    for line in reversed(output.splitlines()):
        m = _COLLECTED.search(line)
        if m:
            return int(m.group(1) or m.group(2))
    return None


def _grouped(conn: sqlite3.Connection, column: str) -> dict[str, float]:
    if column not in ("arm_id", "bucket"):
        raise ValueError(column)
    rows = conn.execute(f"SELECT {column}, SUM(cost_usd) FROM calls GROUP BY {column}")  # noqa: S608
    return {str(k): float(v) for k, v in rows.fetchall()}


def collect_stats(
    *,
    tasks: Sequence[Task],
    verdicts: Mapping[str, AuditVerdict],
    models: ModelsConfig,
    matrix_rows: Sequence[MatrixRow],
    ledger_db: Path | None,
    repo_root: Path,
    collect: CollectRunner = pytest_collect,
) -> Stats:
    ok = experiment_tasks(tasks, verdicts)
    ok_by_repo: dict[str, int] = {}
    for t in ok:
        ok_by_repo[t.repo] = ok_by_repo.get(t.repo, 0) + 1
    totals = (0, 0, 0, 0, 0, 0)
    spend = 0.0
    by_arm: dict[str, float] = {}
    by_bucket: dict[str, float] = {}
    if ledger_db is not None and ledger_db.exists():
        conn = sqlite3.connect(f"file:{ledger_db}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),"
                " COALESCE(SUM(reasoning_tokens),0), COALESCE(SUM(cached_tokens),0),"
                " COALESCE(SUM(cache_write_tokens),0), COALESCE(SUM(cost_usd),0.0) FROM calls"
            ).fetchone()
            totals = (int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]))
            spend = float(row[6])
            by_arm, by_bucket = _grouped(conn, "arm_id"), _grouped(conn, "bucket")
        finally:
            conn.close()
    return Stats(
        repos=sorted(ok_by_repo),
        audited_ok=len(ok),
        audited_train=sum(t.split == "train" for t in ok),
        audited_test=sum(t.split == "test" for t in ok),
        ok_by_repo=dict(sorted(ok_by_repo.items())),
        arms=[a.arm_id for a in models.matrix_arms()],
        attempts=len(matrix_rows),
        ledger_calls=totals[0],
        input_tokens=totals[1],
        output_tokens=totals[2],
        reasoning_tokens=totals[3],
        cached_tokens=totals[4],
        cache_write_tokens=totals[5],
        spend_total=spend,
        spend_by_arm=dict(sorted(by_arm.items())),
        spend_by_bucket=dict(sorted(by_bucket.items())),
        tests_in_repo=count_tests(collect(repo_root)),
    )


def format_stats(s: Stats) -> str:
    lines = ["STATS (each number is read from an artifact: manifest, audit log, ledger, matrix)"]
    lines.append(f"repos with audited-ok tasks: {len(s.repos)} ({', '.join(s.repos) or 'none'})")
    lines.append(
        f"audited-ok tasks: {s.audited_ok} (train {s.audited_train}, test {s.audited_test})"
    )
    for repo, n in s.ok_by_repo.items():
        lines.append(f"  {repo}: {n}")
    lines.append(f"arms: {len(s.arms)} ({', '.join(s.arms)})")
    lines.append(
        f"total attempts (matrix rows): {s.attempts}; logged model calls: {s.ledger_calls}"
    )
    lines.append(
        f"tokens: input {s.input_tokens:,}, output {s.output_tokens:,}, "
        f"reasoning {s.reasoning_tokens:,}, cached {s.cached_tokens:,}, "
        f"cache-write {s.cache_write_tokens:,}"
    )
    lines.append(f"spend: ${s.spend_total:.4f}")
    lines.append("  by arm:")
    lines += [f"    {k}: ${v:.4f}" for k, v in s.spend_by_arm.items()] or ["    (none)"]
    lines.append("  by budget bucket:")
    lines += [f"    {k}: ${v:.4f}" for k, v in s.spend_by_bucket.items()] or ["    (none)"]
    tests = (
        "unknown (pytest collection failed)" if s.tests_in_repo is None else str(s.tests_in_repo)
    )
    lines.append(f"tests in this repository (pytest --collect-only): {tests}")
    return "\n".join(lines)
