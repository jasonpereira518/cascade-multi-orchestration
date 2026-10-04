"""Run the task pipeline over many repos in parallel, and report the funnel.

Two levels of parallelism, both safe because every unit of work writes only its own rows:
  - ``workers``: repos at a time. Each repo has its own state directory, clone, and image.
  - ``jobs``: PRs at a time within a repo (env check and validation), via run_pipeline.
Docker load is roughly workers x jobs, so the defaults are modest.

Resumable and idempotent exactly as for a single repo: finished PRs are skipped, transient errors
are retried. One repo failing (bad config, rate limit, Docker down) does not stop the others; its
error is reported and the rest proceed.

Shared files are written only here, after the parallel part, by one thread: the manifest is merged
and STATUS.md refreshed sequentially, so concurrent workers never race on them. A repo that failed
midway still contributes the tasks it had already validated, because those rows are final.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from cascade.config import ResolvedRepo
from cascade.errors import CascadeError
from cascade.taskgen import attrition
from cascade.taskgen.manifest import merge_repo_tasks
from cascade.taskgen.pipeline import (
    PipelineDeps,
    PipelineSummary,
    build_tasks,
    refresh_status,
    run_pipeline,
)
from cascade.taskgen.store import ENV, FILTER, PATCH, VALIDATE, StageStore

DEFAULT_WORKERS = 4
DEFAULT_JOBS = 2


@dataclass(frozen=True)
class RepoOutcome:
    repo: str
    summary: PipelineSummary | None
    error: str | None  # set if the repo's run raised; its finished rows are still used


def state_path(runs_dir: Path, repo: ResolvedRepo) -> Path:
    return runs_dir / repo.slug / "state.sqlite"


def run_all(
    repos: Sequence[ResolvedRepo],
    *,
    runs_dir: Path,
    manifest_path: Path | None,
    status_md: Path | None,
    stop_after: str = VALIDATE,
    workers: int = DEFAULT_WORKERS,
    jobs: int = DEFAULT_JOBS,
    deps_for: Callable[[ResolvedRepo], PipelineDeps | None] = lambda _repo: None,
) -> list[RepoOutcome]:
    def one(repo: ResolvedRepo) -> RepoOutcome:
        try:
            summary = run_pipeline(
                repo, runs_dir=runs_dir, manifest_path=None, status_md=None,
                stop_after=stop_after, deps=deps_for(repo), jobs=jobs,
            )  # fmt: skip
        except CascadeError as e:
            return RepoOutcome(repo.name, None, f"{type(e).__name__}: {e}")
        return RepoOutcome(repo.name, summary, None)

    if not repos:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(repos)))) as pool:
        outcomes = list(pool.map(one, repos))

    # Sequential, single-writer section: manifest merge and STATUS.md refresh.
    for repo in repos:
        path = state_path(runs_dir, repo)
        if not path.exists():
            continue
        store = StageStore(path)
        try:
            if manifest_path is not None and stop_after == VALIDATE:
                merge_repo_tasks(manifest_path, build_tasks(store, repo), repo.name)
            if status_md is not None:
                refresh_status(store, repo, status_md)
        finally:
            store.close()
    return outcomes


# --- reporting -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RepoFunnel:
    repo: str
    candidates: int
    after_filter: int
    after_patch: int
    env_ok: int
    validated: int
    errors: int
    pending: int
    reasons: dict[str, dict[str, int]]  # stage -> reason -> count


def repo_funnel(repo: ResolvedRepo, runs_dir: Path) -> RepoFunnel | None:
    path = state_path(runs_dir, repo)
    if not path.exists():
        return None
    store = StageStore(path)
    try:
        table = attrition.compute(store)
    finally:
        store.close()
    rows = {r.stage: r for r in table.rows}
    return RepoFunnel(
        repo=repo.name,
        candidates=table.candidates,
        after_filter=rows[FILTER].passed,
        after_patch=rows[PATCH].passed,
        env_ok=rows[ENV].passed,
        validated=rows[VALIDATE].passed,
        errors=sum(r.errors for r in table.rows),
        pending=sum(r.pending for r in table.rows),
        reasons={r.stage: r.reasons for r in table.rows if r.reasons},
    )


def format_report(
    funnels: Sequence[RepoFunnel],
    *,
    missing: Sequence[str] = (),
    outcomes: Sequence[RepoOutcome] = (),
    split_counts: dict[str, dict[str, int]] | None = None,
) -> str:
    lines = [
        "PER-STAGE ATTRITION (candidates -> filter -> patch -> env -> validated)",
        f"{'repo':32s} {'cand':>5s} {'filter':>6s} {'patch':>5s} {'env':>5s} "
        f"{'valid':>5s} {'err':>4s} {'pend':>5s}",
    ]
    totals = [0] * 7
    for f in funnels:
        vals = [
            f.candidates,
            f.after_filter,
            f.after_patch,
            f.env_ok,
            f.validated,
            f.errors,
            f.pending,
        ]
        totals = [a + b for a, b in zip(totals, vals, strict=True)]
        lines.append(
            f"{f.repo:32s} "
            + " ".join(f"{v:{w}d}" for v, w in zip(vals, (5, 6, 5, 5, 5, 4, 5), strict=True))
        )
    lines.append(
        f"{'TOTAL':32s} "
        + " ".join(f"{v:{w}d}" for v, w in zip(totals, (5, 6, 5, 5, 5, 4, 5), strict=True))
    )
    reasons: dict[str, dict[str, int]] = {}
    for f in funnels:
        for stage, by_reason in f.reasons.items():
            merged = reasons.setdefault(stage, {})
            for reason, n in by_reason.items():
                merged[reason] = merged.get(reason, 0) + n
    if reasons:
        lines.append("")
        lines.append("DROP REASONS (all repos)")
        for stage in (FILTER, PATCH, ENV, VALIDATE):
            if stage in reasons:
                top = sorted(reasons[stage].items(), key=lambda kv: (-kv[1], kv[0]))
                lines.append(f"  {stage}: " + ", ".join(f"{r} ({n})" for r, n in top))
    lines.append("")
    lines.append("TASKS PER REPO (validated by the pipeline)")
    for f in funnels:
        extra = ""
        if split_counts and f.repo in split_counts:
            extra = (
                "  [" + ", ".join(f"{k} {v}" for k, v in sorted(split_counts[f.repo].items())) + "]"
            )
        lines.append(f"  {f.repo:32s} {f.validated}{extra}")
    for repo in missing:
        lines.append(f"  {repo:32s} (no state yet)")
    failed = [o for o in outcomes if o.error]
    if failed:
        lines.append("")
        lines.append("REPOS THAT STOPPED EARLY (finished work is kept; rerun to resume)")
        lines += [f"  {o.repo}: {o.error}" for o in failed]
    return "\n".join(lines)
