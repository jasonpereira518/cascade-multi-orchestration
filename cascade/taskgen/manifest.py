"""Reading and rewriting tasks/manifest.jsonl.

One place for every write, so three things stay consistent:
  - the task pipeline replaces only its own repo's lines, and keeps the ``split`` and ``audit`` that
    a human or the frozen split already put on a task, so rerunning mining never undoes either;
  - ``finalize_manifest`` stamps the frozen split, the audit verdicts, and recomputed model-free
    features onto every task, idempotently;
  - writes are atomic (temp file, then rename) and ordered by task id, so diffs are small.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cascade.errors import CascadeError
from cascade.taskgen import audit as audit_mod
from cascade.taskgen import split as split_mod
from cascade.taskgen.cutoffs import clearance
from cascade.taskgen.features import compute_features
from cascade.taskgen.schema import Audit, MinedWindow, Task

_PRESERVED = ("split", "audit")


def read_manifest(path: Path) -> list[Task]:
    if not path.exists():
        return []
    return [
        Task.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _dump(row: dict[str, Any]) -> str:
    return json.dumps(row, separators=(",", ":"), ensure_ascii=False)


def _write_rows(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(_dump(rows[k]) + "\n" for k in sorted(rows)), encoding="utf-8")
    tmp.replace(path)


def write_tasks(path: Path, tasks: Sequence[Task]) -> None:
    _write_rows(path, {t.task_id: json.loads(t.model_dump_json()) for t in tasks})


class WindowMixError(CascadeError):
    """The manifest holds this repo's tasks from a different date window."""


def merge_repo_tasks(
    path: Path,
    new_tasks: Sequence[Task],
    repo_name: str,
    *,
    window: MinedWindow | None = None,
    replace_window: bool = False,
) -> None:
    """Replace this repo's lines; keep every other repo's lines and any existing split/audit.

    With ``window`` given, refuse to replace tasks that were mined under a different recorded
    window unless ``replace_window``: replacing would silently discard audited tasks.
    """
    rows: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["task_id"]] = row
    if window is not None and not replace_window:
        mine = window.model_dump(mode="json")

        def label(w: dict[str, Any]) -> str:
            return f"{w['start']}..{w['end']} ({w['kind']})"

        other = {
            label(r["mined_window"])
            for r in rows.values()
            if r["repo"] == repo_name and r.get("mined_window") and r["mined_window"] != mine
        }
        if other:
            raise WindowMixError(
                f"{path} has {repo_name} tasks mined under {', '.join(sorted(other))}, not "
                f"{window.start}..{window.end} ({window.kind}). Replacing them would discard that "
                "work (audits included); pass --replace-window to do it anyway."
            )
    kept = {k: r for k, r in rows.items() if r["repo"] != repo_name}
    for task in new_tasks:
        row = json.loads(task.model_dump_json())
        old = rows.get(task.task_id)
        if old is not None:
            for key in _PRESERVED:
                if key in old:
                    row[key] = old[key]
        kept[task.task_id] = row
    _write_rows(path, kept)


@dataclass(frozen=True)
class FinalizeResult:
    tasks: int
    by_split: dict[str, int]
    by_audit: dict[str, int]
    new_repos: list[str]


def finalize_manifest(
    manifest_path: Path,
    split_path: Path,
    audit_path: Path,
    *,
    assign_new: bool = True,
    cutoffs: Mapping[str, dt.date | None] | None = None,
) -> FinalizeResult:
    """Stamp split, audit status, recomputed features, and (if ``cutoffs``) which arms' knowledge
    cutoffs each task clears onto every task. Idempotent."""
    tasks = read_manifest(manifest_path)
    split = split_mod.read_split(split_path)
    new_repos: list[str] = []
    if assign_new:
        for repo in sorted({t.repo for t in tasks} - set(split.assignments)):
            split = split_mod.assign_new_repo(split, repo)
            new_repos.append(repo)
        if new_repos:
            split_mod.write_split(split_path, split)
    tasks = split_mod.apply_split(tasks, split)

    verdicts = audit_mod.load_verdicts(audit_path)
    finalized: list[Task] = []
    for t in tasks:
        v = verdicts.get(t.task_id)
        features = compute_features(
            issue_title=t.issue_title, issue_body=t.issue_body, gold_patch=t.gold_patch,
            test_patch=t.test_patch, fail_to_pass=t.fail_to_pass, pass_to_pass=t.pass_to_pass,
            test_runtime_s=t.features.test_runtime_s,
        )  # fmt: skip
        audit = Audit(status=v.verdict, notes=v.notes) if v else Audit(status="pending", notes="")
        update: dict[str, Any] = {"features": features, "audit": audit}
        if cutoffs is not None:
            update["cutoff_clearance"] = clearance(t.merged_at, cutoffs)
        finalized.append(t.model_copy(update=update))
    write_tasks(manifest_path, finalized)

    by_split: dict[str, int] = {}
    by_audit: dict[str, int] = {}
    for t in finalized:
        by_split[t.split or "unassigned"] = by_split.get(t.split or "unassigned", 0) + 1
        by_audit[t.audit.status] = by_audit.get(t.audit.status, 0) + 1
    return FinalizeResult(len(finalized), by_split, by_audit, new_repos)
