"""Loading the outcome matrix for evaluation, with provenance, filtering, and the refusals.

What this module guarantees, because every later number depends on it:

- **Provenance.** Data is ``synthetic`` if the directory carries the ``SYNTHETIC.json`` marker the
  test generator writes; otherwise it is ``real`` ONLY IF a ledger export is present and every
  logged call came from the ``openai`` provider (a mock-provider matrix is not real data). Anything
  else is refused. Synthetic and real data are never combined: one data directory, one source.
  Every output carries the data source, the git commit (and whether the tree was dirty), and the
  SHA-256 of the matrix file.
- **One tier.** Every row in the matrix file must have the same ``processing_tier``; otherwise the
  load is refused (a mixed-tier matrix compares costs that are not comparable).
- **Default-pipeline rows only** feed policies: seed 0, the full pipeline (reproducer and reviewer
  on). Ablation rows (reproducer or reviewer off) and repeat-seed rows are kept apart, for the
  ablation and noise analyses only. If more than one distinct default config exists the load is
  refused as ambiguous.
- **Cap-exceeded cells are failures.** An attempt that hit a per-attempt cap counts as a failed cell
  (``true_success`` False, ``verifier_claimed`` False so a cascade escalates) with the cost it
  incurred, even if its partial patch happened to pass.
- **Splits.** Only tasks in the requested splits are read; rows of other splits are never
  materialized. A dev run requests ``train`` only.
- **Complete tasks only.** A task needs a default row for all three arms; others are excluded and
  listed.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cascade.errors import CascadeError
from cascade.evals import parquet_io
from cascade.routing.matrix import MatrixRow
from cascade.routing.policies import ARMS, Cell, TaskRecord
from cascade.taskgen.audit import load_verdicts
from cascade.taskgen.manifest import read_manifest

SYNTHETIC_MARKER = "SYNTHETIC.json"
_GIT = "git"
CAP_MARKERS = ("cap_exceeded", "attempt cap", "attempt token cap", "task cap")


class DataError(CascadeError):
    """The matrix cannot be used as evidence."""


class ProvenanceError(DataError):
    """The data's origin cannot be established."""


class MixedTierError(DataError):
    """Rows use more than one processing tier."""


def is_cap_exceeded(error: str | None) -> bool:
    return bool(error) and any(m in (error or "") for m in CAP_MARKERS)


def _needs_normalizing(row: MatrixRow) -> bool:
    return is_cap_exceeded(row.error) and (row.true_success or row.verifier_claimed)


def normalize(row: MatrixRow) -> MatrixRow:
    """A cap-exceeded attempt is a failure that keeps its incurred cost."""
    if _needs_normalizing(row):
        return row.model_copy(update={"true_success": False, "verifier_claimed": False})
    return row


def git_info(repo_root: Path) -> tuple[str, bool]:
    """(HEAD commit, working tree dirty?). ``("unknown", True)`` outside a git repository."""

    def run(*args: str) -> str:
        return subprocess.run(  # noqa: S603  (fixed argv, no shell)
            [_GIT, "-C", str(repo_root), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return run("rev-parse", "HEAD"), bool(run("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True


@dataclass(frozen=True)
class Provenance:
    data_source: str  # "real" | "synthetic"
    git_commit: str
    git_dirty: bool
    matrix_hash: str
    tier: str
    splits: tuple[str, ...]
    n_rows_loaded: int
    n_cap_normalized: int
    n_incomplete_tasks: int
    default_config: str

    def stamp(self) -> str:
        dirty = "DIRTY" if self.git_dirty else "clean"
        return (
            f"data_source={self.data_source} | commit={self.git_commit[:12]} ({dirty}) | "
            f"matrix={self.matrix_hash[:16]} | tier={self.tier} | splits={','.join(self.splits)}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "data_source": self.data_source, "git_commit": self.git_commit,
            "git_dirty": self.git_dirty, "matrix_hash": self.matrix_hash, "tier": self.tier,
            "splits": list(self.splits), "n_rows_loaded": self.n_rows_loaded,
            "n_cap_normalized": self.n_cap_normalized,
            "n_incomplete_tasks": self.n_incomplete_tasks, "default_config": self.default_config,
        }  # fmt: skip


@dataclass(frozen=True)
class Dataset:
    provenance: Provenance
    records: list[TaskRecord]  # complete tasks of the loaded splits, sorted by task id
    incomplete: list[str]
    default_rows: dict[tuple[str, str], MatrixRow]  # (task_id, arm_id) at seed 0, default config
    ablation_rows: dict[str, dict[str, MatrixRow]]  # variant -> task_id -> mid-arm row
    repeat_rows: dict[tuple[str, str], MatrixRow]  # (task_id, arm_id), seed != 0, default config
    arm_ids: dict[str, str]  # label -> arm id
    audited_by_split: dict[str, int]  # audited-ok tasks per split, from the manifest (no outcomes)


def _config(row: MatrixRow) -> dict[str, Any]:
    parsed = json.loads(row.pipeline_config)
    if not isinstance(parsed, dict):
        raise DataError(f"pipeline_config of {row.task_id} is not a JSON object")
    items: dict[Any, Any] = dict(parsed)  # pyright: ignore[reportUnknownArgumentType]
    return {str(k): v for k, v in items.items()}


def _is_full_pipeline(row: MatrixRow) -> bool:
    cfg = _config(row)
    return bool(cfg.get("reproducer")) and bool(cfg.get("reviewer"))


def _ablation_label(cfg: Mapping[str, Any], default: Mapping[str, Any]) -> str | None:
    """no_reproducer / no_reviewer if ``cfg`` is the default with exactly that role switched off."""
    rest = {k for k in cfg if k not in ("reproducer", "reviewer")}
    if any(cfg.get(k) != default.get(k) for k in rest):
        return None
    if cfg.get("reproducer") is False and cfg.get("reviewer") is True:
        return "no_reproducer"
    if cfg.get("reproducer") is True and cfg.get("reviewer") is False:
        return "no_reviewer"
    return None


def detect_source(data_dir: Path) -> str:
    """'synthetic' (marker present) or 'real' (ledger shows openai-only calls); else refuse."""
    if (data_dir / SYNTHETIC_MARKER).exists():
        return "synthetic"
    ledger = data_dir / "ledger.parquet"
    if not ledger.exists():
        raise ProvenanceError(
            f"{data_dir} has no {SYNTHETIC_MARKER} and no ledger.parquet: cannot establish that "
            "this matrix came from real runs (export it with `cascade export`)"
        )
    providers = Counter(str(p) for p in parquet_io.read_column(ledger, "provider"))
    if not providers:
        raise ProvenanceError(f"{ledger} is empty: no logged calls back this matrix")
    if set(providers) != {"openai"}:
        raise ProvenanceError(
            f"the ledger shows providers {dict(providers)}; only openai calls count as real data "
            "(a mock-provider matrix is synthetic; use tests/synthetic for development)"
        )
    return "real"


def _check_cost_against_ledger(data_dir: Path) -> None:
    matrix = sum(float(c) for c in parquet_io.read_column(data_dir / "matrix.parquet", "cost_usd"))
    ledger = sum(float(c) for c in parquet_io.read_column(data_dir / "ledger.parquet", "cost_usd"))
    if matrix > ledger * (1 + 1e-6) + 1e-9:
        raise ProvenanceError(
            f"matrix cost ${matrix:.6f} exceeds the ledger's ${ledger:.6f}: rows are not backed "
            "by logged calls"
        )


def load_dataset(
    *,
    data_dir: Path,
    manifest: Path,
    audit_file: Path,
    arms: Mapping[str, str],
    splits: Sequence[str],
    repo_root: Path,
) -> Dataset:
    matrix_path = data_dir / "matrix.parquet"
    if not matrix_path.exists():
        raise DataError(f"no matrix at {matrix_path}; run the matrix and `cascade export` first")
    if sorted(arms) != sorted(ARMS):
        raise DataError(f"arms must be labelled {ARMS}, got {sorted(arms)}")
    source = detect_source(data_dir)
    if source == "real":
        _check_cost_against_ledger(data_dir)

    tiers = sorted({str(t) for t in parquet_io.read_column(matrix_path, "processing_tier")})
    if not tiers:
        raise DataError(f"{matrix_path} has no rows")
    if len(tiers) > 1:
        raise MixedTierError(
            f"the matrix mixes processing tiers {tiers}; costs on different tiers are not "
            "comparable, so nothing will be evaluated"
        )

    verdicts = load_verdicts(audit_file)
    audited = [t for t in read_manifest(manifest) if t.task_id in verdicts]
    tasks = [t for t in audited if verdicts[t.task_id].verdict == "ok"]
    unsplit = [t.task_id for t in tasks if t.split is None]
    if unsplit:
        raise DataError(f"{len(unsplit)} audited-ok task(s) have no split; run `tasks finalize`")
    audited_by_split = dict(Counter(str(t.split) for t in tasks))
    wanted = {t.task_id: t for t in tasks if t.split in splits}
    if not wanted:
        raise DataError(f"no audited-ok tasks in splits {list(splits)}")

    rows = [MatrixRow.model_validate(r) for r in parquet_io.read_rows(matrix_path, set(wanted))]
    cap_normalized = sum(1 for r in rows if _needs_normalizing(r))
    rows = [normalize(r) for r in rows]

    full = [r for r in rows if r.seed == 0 and _is_full_pipeline(r)]
    configs = sorted({r.pipeline_config for r in full})
    if not configs:
        raise DataError("no default-pipeline rows (seed 0, reproducer and reviewer on)")
    if len(configs) > 1:
        raise DataError(
            f"{len(configs)} distinct default pipeline configs; the default is ambiguous"
        )
    default_json = configs[0]
    default_cfg = _config(full[0])

    default_rows = {(r.task_id, r.arm_id): r for r in full}
    unknown_arms = {a for (_t, a) in default_rows if a not in set(arms.values())}
    if unknown_arms:
        raise DataError(f"default rows use arms not in the arm map: {sorted(unknown_arms)}")
    ablation: dict[str, dict[str, MatrixRow]] = {}
    repeat: dict[tuple[str, str], MatrixRow] = {}
    for r in rows:
        if r.pipeline_config == default_json and r.seed != 0:
            repeat[(r.task_id, r.arm_id)] = r
        elif r.seed == 0 and r.pipeline_config != default_json and r.arm_id == arms["mid"]:
            label = _ablation_label(_config(r), default_cfg)
            if label is not None:
                ablation.setdefault(label, {})[r.task_id] = r

    records: list[TaskRecord] = []
    incomplete: list[str] = []
    for task_id in sorted(wanted):
        cells: list[Cell] = []
        for label in ARMS:
            r = default_rows.get((task_id, arms[label]))
            if r is None:
                break
            cells.append(Cell(r.true_success, r.verifier_claimed, r.cost_usd, r.wall_s))
        if len(cells) == len(ARMS):
            records.append(TaskRecord(wanted[task_id], (cells[0], cells[1], cells[2])))
        else:
            incomplete.append(task_id)
    if not records:
        raise DataError("no task has a default row for every arm")

    commit, dirty = git_info(repo_root)
    prov = Provenance(
        data_source=source, git_commit=commit, git_dirty=dirty,
        matrix_hash=parquet_io.file_sha256(matrix_path), tier=tiers[0], splits=tuple(splits),
        n_rows_loaded=len(rows), n_cap_normalized=cap_normalized,
        n_incomplete_tasks=len(incomplete), default_config=default_json,
    )  # fmt: skip
    return Dataset(
        prov, records, incomplete, default_rows, ablation, repeat, dict(arms), audited_by_split
    )
