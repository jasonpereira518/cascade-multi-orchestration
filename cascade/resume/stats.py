"""`artifacts/resume_stats.json`: every number a resume bullet, the README, or the write-up may use.

Each value is read from an artifact a run produced (the final-evaluation outputs and lock, the
matrix and ledger parquet exports, the manifest and audit log, the pilot review log); nothing is
typed in, estimated, or defaulted. A missing input is an error that says which one. The file has no
timestamp, so regenerating it from unchanged inputs is byte-identical, and it records the SHA-256 of
every input so a later edit to any of them is visible.

Synthetic data is refused: the final evaluation of a labeled-synthetic matrix can only exist through
the test-only flag, and its numbers are not evidence, so ``allow_synthetic`` stamps the output
SYNTHETIC and everything downstream refuses to render it without the same flag.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cascade.errors import CascadeError
from cascade.eval_cli import arms_for
from cascade.evals import parquet_io
from cascade.evals.designate import (
    BASELINE_POLICY,
    FinalArtifactError,
    FinalEval,
    designated_policy,
    load_final,
)
from cascade.pilot import reviews as rev
from cascade.routing.calibrate import wilson
from cascade.routing.selection import audited_ok
from cascade.taskgen.audit import load_verdicts
from cascade.taskgen.manifest import read_manifest

SCHEMA_VERSION = 1
DEFAULT_PATH = Path("artifacts/resume_stats.json")
EXPERIMENT = "experiment"
LIVE_PILOT = "live_pilot"


class StatsError(CascadeError):
    """An input is missing or inconsistent, so no stats were written."""


@dataclass(frozen=True)
class StatsInputs:
    final_dir: Path = Path("results/final")
    lock_path: Path = Path("FINAL.lock.json")
    data_dir: Path = Path("runs/export")
    manifest: Path = Path("tasks/manifest.jsonl")
    audit_file: Path = Path("tasks/audit.jsonl")
    reviews_file: Path = Path("tasks/pilot_reviews.jsonl")
    dev_dir: Path = Path("results/dev")
    configs_dir: Path = Path("configs")


def _require(path: Path, what: str, hint: str) -> Path:
    if not path.exists():
        raise StatsError(f"missing input: {what} ({path}). {hint}")
    return path


def _interval(d: dict[str, Any] | None) -> dict[str, Any] | None:
    if d is None:
        return None
    return {k: d[k] for k in ("point", "lo", "hi", "n_undefined")}


def _policies(results: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in results["metrics"]:
        out[str(m["name"])] = {
            "n_tasks": m["n"],
            "resolve_rate": _interval(m["resolve_rate"]),
            "cost_per_task_usd": _interval(m["cost_per_task"]),
            "cost_per_resolved_usd": _interval(m["cost_per_resolved"]),
            "mean_arms_tried": m["mean_arms_tried"],
        }
    return out


def _point_cost_per_resolved(policies: dict[str, Any], name: str) -> float | None:
    entry: dict[str, Any] | None = policies.get(name)
    if entry is None:
        return None
    iv: dict[str, Any] | None = entry["cost_per_resolved_usd"]
    return None if iv is None or iv["point"] is None else float(iv["point"])


def _cpr_reduction(policies: dict[str, Any], name: str) -> float | None:
    """1 - cost_per_resolved(policy) / cost_per_resolved(always_max), from the POINT estimates."""
    base = _point_cost_per_resolved(policies, BASELINE_POLICY)
    own = _point_cost_per_resolved(policies, name)
    if base is None or own is None or base <= 0:
        return None
    return 1.0 - own / base


def _comparisons(results: dict[str, Any], policies: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in results["comparisons"]:
        name = str(c["policy"])
        out[name] = {
            "baseline": c["baseline"],
            "margin": c["margin"],
            "resolve_rate_diff": _interval(c["resolve_rate_diff"]),
            "cost_per_task_reduction": _interval(c["cost_reduction"]),
            "cost_per_resolved_reduction_point": _cpr_reduction(policies, name),
            "oracle_savings_share": _interval(c["oracle_savings_share"]),
            "non_inferior": c["non_inferior"],
            "saves": c["saves"],
            "verdict": c["verdict"],
            "verdict_clustered": c["verdict_clustered"],
        }
    return out


def _verifier(results: dict[str, Any]) -> dict[str, Any]:
    rows = results["verifier"]
    failed = sum(int(r["n"]) - int(r["passed"]) for r in rows)
    false_pos = sum(int(r["false_pos"]) for r in rows)
    pooled: dict[str, Any] | None = None
    if failed:
        lo, hi = wilson(false_pos, failed)
        pooled = {
            "n_failed_attempts": failed, "false_positives": false_pos,
            "fp_rate": {"point": false_pos / failed, "lo": lo, "hi": hi},
        }  # fmt: skip
    return {
        "per_arm": [
            {
                "arm": r["arm"],
                "n": r["n"],
                "passed": r["passed"],
                "claimed": r["claimed"],
                "false_pos": r["false_pos"],
                "false_neg": r["false_neg"],
                "fp_rate": _interval(r["fp_rate"]),
                "fn_rate": _interval(r["fn_rate"]),
            }
            for r in rows
        ],
        "pooled": pooled,
        "scope": "final-evaluation test split; fp_rate = P(Verifier claimed success | failed)",
    }


def _ledger(data_dir: Path) -> dict[str, Any]:
    path = _require(
        data_dir / "ledger.parquet", "the ledger export", "Run `cascade export` to create it."
    )
    cols = set(parquet_io.column_names(path))
    cost = [float(x) for x in parquet_io.read_column(path, "cost_usd")]
    purposes = (
        [str(x) for x in parquet_io.read_column(path, "purpose")]
        if "purpose" in cols
        else [EXPERIMENT] * len(cost)
    )
    tokens = {
        k: sum(int(x) for x in parquet_io.read_column(path, f"{k}_tokens"))
        for k in ("input", "output", "reasoning")
    }
    by_purpose: dict[str, float] = {}
    for purpose, c in zip(purposes, cost, strict=True):
        by_purpose[purpose] = by_purpose.get(purpose, 0.0) + c
    return {
        "tokens": {**tokens, "total": sum(tokens.values())},
        "spend_usd": {
            "experiment": by_purpose.get(EXPERIMENT, 0.0),
            "live_pilot": by_purpose.get(LIVE_PILOT, 0.0),
            "total": sum(cost),
        },
        "n_calls": len(cost),
        "purpose_column_present": "purpose" in cols,
    }


def _dev_noise(inputs: StatsInputs, source: str, matrix_hash: str) -> list[dict[str, Any]] | None:
    """The repeat-subset noise estimate from the matching DEV analysis (train only), if it ran."""
    path = inputs.dev_dir / f"{source}-{matrix_hash[:8]}" / "results.json"
    if not path.exists():
        return None
    noise = json.loads(path.read_text(encoding="utf-8")).get("noise")
    return list(noise) if noise else None


def _tasks(inputs: StatsInputs, final: FinalEval) -> dict[str, Any]:
    _require(inputs.manifest, "the task manifest", "Build it with `cascade taskgen run-all`.")
    _require(inputs.audit_file, "the audit log", "Audit tasks with `cascade audit`.")
    ok = audited_ok(read_manifest(inputs.manifest), load_verdicts(inputs.audit_file))
    by_split = Counter(str(t.split) for t in ok)
    n_test_eval = int(final.results["n_tasks"])
    n_train_eval = int(final.lock["n_train_tasks"])
    if by_split.get("test", 0) != int(final.lock["n_test_tasks"]):
        raise StatsError(
            f"the manifest has {by_split.get('test', 0)} audited test tasks but the final lock "
            f"recorded {final.lock['n_test_tasks']}: the manifest or audit changed after the "
            "final evaluation"
        )
    complete = by_split.get("train", 0) == n_train_eval and by_split.get("test", 0) == n_test_eval
    repos = sorted({t.repo for t in ok})
    return {
        "audited": {
            "train": by_split.get("train", 0),
            "test": by_split.get("test", 0),
            "total": len(ok),
        },
        "evaluated": {
            "train": n_train_eval,
            "test": n_test_eval,
            "total": n_train_eval + n_test_eval,
        },
        "all_audited_tasks_evaluated": complete,
        # a repo count is only exact when no audited task is missing from the matrix
        "repos": {"n": len(repos) if complete else None, "names": repos},
    }


def build_stats(inputs: StatsInputs, *, allow_synthetic: bool = False) -> dict[str, Any]:
    matrix = _require(
        inputs.data_dir / "matrix.parquet", "the matrix export", "Run `cascade export` first."
    )
    try:
        final = load_final(inputs.final_dir, inputs.lock_path, matrix_path=matrix)
    except FinalArtifactError as e:
        raise StatsError(str(e)) from e
    source = final.data_source
    if source != "real" and not allow_synthetic:
        raise StatsError(
            f"the final evaluation was run on {source} data, which is not evidence; resume stats "
            "are built from real data only (--allow-synthetic is for tests)"
        )
    _require(
        inputs.reviews_file, "the pilot review log",
        "Run `cascade review` for each pilot draft (an empty log means no pilot).",
    )  # fmt: skip
    results = final.results
    policies = _policies(results)
    for needed in (BASELINE_POLICY, "cascade"):
        if needed not in policies:
            raise StatsError(f"the final results have no policy {needed!r}")
    designated = designated_policy(results)
    ledger = _ledger(inputs.data_dir)
    pilot = rev.pilot_metrics(rev.latest(rev.read_reviews(inputs.reviews_file)))
    arms = arms_for(inputs.data_dir, inputs.configs_dir)
    tasks = _tasks(inputs, final)  # also requires the manifest and audit log, before hashing them
    return {
        "schema": SCHEMA_VERSION,
        "data_source": source,
        "synthetic_warning": None if source == "real" else "SYNTHETIC: not evidence",
        "inputs_sha256": {
            "final_results": parquet_io.file_sha256(inputs.final_dir / "results.json"),
            "final_lock": parquet_io.file_sha256(inputs.lock_path),
            "matrix": parquet_io.file_sha256(matrix),
            "ledger": parquet_io.file_sha256(inputs.data_dir / "ledger.parquet"),
            "manifest": parquet_io.file_sha256(inputs.manifest),
            "audit": parquet_io.file_sha256(inputs.audit_file),
            "pilot_reviews": parquet_io.file_sha256(inputs.reviews_file),
        },
        "final": {
            "commit": final.lock["commit"],
            "matrix_hash": final.lock["matrix_hash"],
            "n_test_tasks": results["n_tasks"],
            "n_test_repos": results["n_repos"],
            "margin": results["margin"],
            "margin_basis": results["margin_basis"],
            "resamples": results["resamples"],
            "seed": results["seed"],
            "tier": results["provenance"]["tier"],
        },
        "tasks": tasks,
        "arms": {"n": len(arms), "ids": arms},
        "attempts": {"n_matrix_rows": parquet_io.row_count(matrix)},
        "tokens": ledger["tokens"],
        "spend_usd": ledger["spend_usd"],
        "ledger": {
            "n_calls": ledger["n_calls"],
            "purpose_column_present": ledger["purpose_column_present"],
        },
        "policies": policies,
        "comparisons": _comparisons(results, policies),
        "designated_policy": designated,
        "claim_supported": designated != BASELINE_POLICY,
        "verifier": _verifier(results),
        "noise": _dev_noise(inputs, source, final.lock["matrix_hash"]),
        "pilot": asdict(pilot),
    }


def write_stats(stats: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(stats, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def load_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StatsError(f"missing input: {path}. Run `cascade resume-stats` first.")
    stats: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if stats.get("schema") != SCHEMA_VERSION:
        raise StatsError(f"{path} has schema {stats.get('schema')!r}; expected {SCHEMA_VERSION}")
    return stats
