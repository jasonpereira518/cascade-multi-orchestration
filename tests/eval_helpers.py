"""Helpers for the eval tests: write synthetic datasets, tamper with parquet, build tiny git repos.

Everything is SYNTHETIC. ``make_real_like`` only fakes the PROVENANCE evidence (a ledger of openai
calls and no SYNTHETIC marker) so the loader's real-data path can be tested; the numbers inside are
still the generator's invented ones, and no test may present them as results.
"""

# pyright: basic
from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from tests.synthetic import gen

PLAN = "docs/EVAL_PLAN.md"


def write_synthetic(out: Path, seed: int = 0) -> Path:
    return gen.write(gen.generate(seed), out)


def rewrite_matrix(data_dir: Path, edit: Callable[[list[dict[str, Any]]], None]) -> None:
    path = data_dir / "matrix.parquet"
    rows: list[dict[str, Any]] = pq.read_table(path).to_pylist()
    edit(rows)
    pq.write_table(pa.Table.from_pylist(rows), path)


def make_real_like(
    data_dir: Path, providers: list[str] | None = None, extra_cost: float = 1.0
) -> None:
    """Drop the SYNTHETIC marker and add a ledger export, so the loader treats it as real data."""
    (data_dir / "SYNTHETIC.json").unlink()
    costs = pq.read_table(data_dir / "matrix.parquet").column("cost_usd").to_pylist()
    providers = ["openai", "openai"] if providers is None else providers
    share = (sum(float(c) for c in costs) + extra_cost) / max(len(providers), 1)
    table = pa.table(
        {
            "provider": pa.array(providers, pa.string()),
            "cost_usd": pa.array([share] * len(providers), pa.float64()),
        }
    )
    pq.write_table(table, data_dir / "ledger.parquet")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
            "-c", "commit.gpgsign=false", *args,
        ],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip


def init_repo(root: Path, *, commit_plan: bool = True) -> Path:
    """A tiny git repo with a README commit, and (optionally) docs/EVAL_PLAN.md committed."""
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q", "-b", "main")
    (root / "README.md").write_text("synthetic test repo\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    if commit_plan:
        (root / PLAN).parent.mkdir(parents=True, exist_ok=True)
        (root / PLAN).write_text("# plan\n")
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", "plan")
    return root
