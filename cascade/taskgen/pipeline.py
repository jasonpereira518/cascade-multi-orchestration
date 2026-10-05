"""Orchestrates mine -> filter -> patch -> env -> validate for one repo, resumably.

Every stage reads and writes the SQLite StageStore, so rerunning skips every PR whose stage result
is final (ok or dropped) and retries only "error" rows. State lives in a directory keyed by the
date window (``<repo>/w-<start>_<end>/``), so mining under two windows can never share or mix rows;
the clone is shared because git data does not depend on the window. The state is also tied to a
fingerprint of the repo config; changing limits or commands against an old state is refused.

Every run appends to ``runs.jsonl`` in the state directory (window, kind, stages, timeouts), reports
live progress through ``deps.progress``, and gives each PR a wall-clock budget per stage
(``deps.pr_timeout_s``): a PR that exceeds it is skipped, logged, and retried on the next run.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import shutil
import threading
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cascade.config import ResolvedRepo
from cascade.errors import CascadeError
from cascade.harness.grading import GradingRunner
from cascade.harness.sandbox import prepare_workspace
from cascade.taskgen import attrition
from cascade.taskgen.build_env import (
    DockerRun,
    DockerTestExecutor,
    DockerUnavailableError,
    build_image,
    default_docker_run,
    docker_available,
    image_exists,
)
from cascade.taskgen.cutoffs import clearance
from cascade.taskgen.features import compute_features
from cascade.taskgen.filter import FilterContext, filter_metadata
from cascade.taskgen.github import GitHubClient, PRCandidate
from cascade.taskgen.gitrepo import (
    GitError,
    PatchApplyError,
    apply_patch,
    derive_patches,
    ensure_clone,
)
from cascade.taskgen.manifest import merge_repo_tasks
from cascade.taskgen.mine import check_repo, mine, month_slices
from cascade.taskgen.results import ERROR as OUTCOME_ERROR
from cascade.taskgen.schema import Audit, MinedWindow, Task
from cascade.taskgen.store import DROPPED, ENV, ERROR, FILTER, OK, PATCH, VALIDATE, StageStore
from cascade.taskgen.validate import PATCH_APPLY_FAILED, TestExecutor, validate_task

STAGE_ORDER = ("mine", FILTER, PATCH, ENV, VALIDATE)
DEFAULT_PR_TIMEOUT_S = 900.0  # 3 validation runs at the 120 s test cap, plus setup, with headroom
PR_TIMEOUT = "pr_timeout"
ENV_COLLECT_FAILED = "env_collect_failed"
ENV_NO_TESTS = "env_no_tests"

_TEST_FILE_IN_PATCH = re.compile(r"^\+\+\+ b/(\S+)$", re.MULTILINE)


class StateMismatchError(CascadeError):
    """The state directory was built with a different repo configuration."""


@dataclass
class PipelineDeps:
    """Everything the pipeline touches outside the filesystem, injectable for tests."""

    client_factory: Callable[[], GitHubClient] = GitHubClient
    docker_run: DockerRun = default_docker_run
    clone_url: Callable[[str], str] = lambda repo: f"https://github.com/{repo}.git"
    executor_factory: Callable[[str, Path], TestExecutor] | None = None
    today: dt.date | None = None
    # Wall-clock budget per PR per stage (patch, env, validate); None disables it.
    pr_timeout_s: float | None = DEFAULT_PR_TIMEOUT_S
    # progress(repo, stage, done, total, validated_so_far); called from worker threads.
    progress: Callable[[str, str, int, int, int], None] | None = None
    # arm id -> knowledge cutoff, for tagging tasks (see taskgen.cutoffs). Empty: no tags.
    cutoffs: Mapping[str, dt.date | None] = field(default_factory=lambda: {})
    replace_window: bool = False  # allow replacing manifest tasks mined under another window


@dataclass
class PipelineSummary:
    stage_reached: str
    manifest_tasks: int = 0
    errors: list[str] = field(default_factory=lambda: [])


def config_fingerprint(repo: ResolvedRepo) -> str:
    return hashlib.sha256(repo.model_dump_json().encode()).hexdigest()[:16]


def _check_fingerprint(store: StageStore, repo: ResolvedRepo) -> None:
    fp = config_fingerprint(repo)
    existing = store.get_kv("config_fingerprint")
    if existing is None:
        store.set_kv("config_fingerprint", fp)
    elif existing != fp:
        raise StateMismatchError(
            f"{store.path} was built with a different config for {repo.name}. Use a fresh "
            "--runs-dir (or delete this state) instead of mixing two configurations."
        )


def _candidate(data: dict[str, Any]) -> PRCandidate:
    return PRCandidate.model_validate(data)


def state_dir(runs_dir: Path, repo: ResolvedRepo) -> Path:
    return runs_dir / repo.slug / f"w-{repo.window_key}"


def mined_window(repo: ResolvedRepo) -> MinedWindow:
    return MinedWindow(start=repo.window_start, end=repo.window_end, kind=repo.window_kind)


class RunLog:
    """Append-only JSONL record of what each run did (window, stages, skipped PRs)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def event(self, **fields: Any) -> None:
        line = {"ts": dt.datetime.now(dt.UTC).isoformat(), **fields}
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, sort_keys=True, default=str) + "\n")


@dataclass
class Tracker:
    """Reports progress for one stage of one repo; ``done`` counts PRs finished or skipped."""

    deps: PipelineDeps
    store: StageStore
    repo: str
    stage: str
    total: int
    done: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def tick(self) -> None:
        with self._lock:
            self.done += 1
            done = self.done
        if self.deps.progress is not None:
            self.deps.progress(
                self.repo, self.stage, done, self.total, self.store.count(VALIDATE, OK)
            )


def _with_timeout(
    fn: Callable[[int], str | None],
    number: int,
    *,
    timeout_s: float | None,
    on_timeout: Callable[[int], str | None],
) -> str | None:
    """Run ``fn(number)``; if it exceeds ``timeout_s`` give up on it and call ``on_timeout``.

    A Python thread cannot be killed, so the abandoned call is left to finish in a daemon thread.
    Its late result (if any) is still correct data for that PR; what matters is that the run
    moves on instead of hanging.
    """
    if timeout_s is None:
        return fn(number)
    box: list[str | None] = []
    failure: list[BaseException] = []
    abandoned = threading.Event()

    def target() -> None:
        try:
            box.append(fn(number))
        except BaseException as e:
            if not abandoned.is_set():  # nobody is waiting any more (the store may be closed)
                failure.append(e)

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        abandoned.set()
        return on_timeout(number)
    if failure:
        raise failure[0]
    return box[0]


def _skip_on_timeout(
    store: StageStore, log: RunLog | None, stage: str, timeout_s: float | None
) -> Callable[[int], str | None]:
    def skip(number: int) -> str | None:
        store.put_stage(number, stage, ERROR, PR_TIMEOUT, {"timeout_s": timeout_s})
        if log is not None:
            log.event(event=PR_TIMEOUT, stage=stage, pr=number, timeout_s=timeout_s)
        return f"PR {number} {stage}: skipped after {timeout_s:g}s (retried on rerun)"

    return skip


def run_filter(store: StageStore, repo: ResolvedRepo, default_branch: str) -> None:
    ctx = FilterContext(
        window_start=repo.window_start,
        window_end=repo.window_end,
        default_branch=default_branch,
        repo=repo.name,
        limits=repo.limits,
    )
    for number, data in store.candidates():
        if store.is_final(number, FILTER):
            continue
        reason = filter_metadata(_candidate(data), ctx)
        store.put_stage(number, FILTER, DROPPED if reason else OK, reason)


def run_patch(
    store: StageStore,
    repo: ResolvedRepo,
    clone: Path,
    url: str,
    deps: PipelineDeps | None = None,
    log: RunLog | None = None,
) -> list[str]:
    """Derive gold and test patches per PR. Returns retryable messages (timeouts)."""
    deps = deps or PipelineDeps()
    todo = [n for n in store.passed(FILTER) if not store.is_final(n, PATCH)]
    if not todo:
        return []
    ensure_clone(url, clone)
    cands = dict(store.candidates())
    track = Tracker(deps, store, repo.name, PATCH, len(todo))
    skipped: list[str] = []

    def one(number: int) -> str | None:
        _patch_one(store, repo, clone, cands, number)
        return None

    for number in todo:
        msg = _with_timeout(
            one, number, timeout_s=deps.pr_timeout_s,
            on_timeout=_skip_on_timeout(store, log, PATCH, deps.pr_timeout_s),
        )  # fmt: skip
        if msg:
            skipped.append(msg)
        track.tick()
    return skipped


def _patch_one(
    store: StageStore,
    repo: ResolvedRepo,
    clone: Path,
    cands: dict[int, dict[str, Any]],
    number: int,
) -> None:
    try:
        outcome = derive_patches(clone, _candidate(cands[number]), repo.limits)
    except GitError as e:
        store.put_stage(number, PATCH, ERROR, "git_error", {"message": str(e)})
        return
    if outcome.data is None:
        store.put_stage(number, PATCH, DROPPED, outcome.reason)
    else:
        d = outcome.data
        store.put_stage(
            number,
            PATCH,
            OK,
            data={
                "base_commit": d.base_commit,
                "gold_patch": d.gold_patch,
                "test_patch": d.test_patch,
                "src_files": d.src_files,
                "src_lines": d.src_lines,
            },
        )


def _workspace_factory(
    clone: Path, base_commit: str, test_patch: str, gold_patch: str, work: Path, label: str
) -> Callable[[bool], Path]:
    def make(with_gold: bool) -> Path:
        ws = work / f"{label}-{uuid.uuid4().hex[:8]}"
        prepare_workspace(clone, base_commit, ws)
        apply_patch(ws, test_patch)
        if with_gold:
            apply_patch(ws, gold_patch)
        return ws

    return make


def _stage_data(store: StageStore, number: int, stage: str) -> dict[str, Any]:
    got = store.get_stage(number, stage)
    if got is None:
        raise CascadeError(f"PR {number} has no {stage} result in the state store")
    return got[2]


def _test_files(test_patch: str) -> list[str]:
    return sorted(set(_TEST_FILE_IN_PATCH.findall(test_patch)))


def _ensure_image(store: StageStore, repo: ResolvedRepo, clone: Path, deps: PipelineDeps) -> str:
    """Build the repo image once. Reused on reruns while the image still exists locally."""
    if not docker_available(deps.docker_run):
        raise DockerUnavailableError(
            "the docker daemon is not reachable; start Docker and rerun (finished stages are kept)"
        )
    tag = store.get_kv("image_tag")
    if tag and image_exists(tag, deps.docker_run):
        return tag
    cands = dict(store.candidates())
    passing = store.passed(PATCH)
    newest = max(passing, key=lambda n: _candidate(cands[n]).merged_at)
    ref_commit = str(_stage_data(store, newest, PATCH)["base_commit"])
    tag = build_image(repo, clone, ref_commit, run_fn=deps.docker_run)
    store.set_kv("image_tag", tag)
    store.set_kv("image_ref_commit", ref_commit)
    return tag


def run_env_and_validate(
    store: StageStore,
    repo: ResolvedRepo,
    clone: Path,
    work: Path,
    deps: PipelineDeps,
    *,
    do_validate: bool,
    jobs: int = 1,
    log: RunLog | None = None,
) -> list[str]:
    errors: list[str] = []
    pending_env = [n for n in store.passed(PATCH) if not store.is_final(n, ENV)]
    pending_val = do_validate and any(not store.is_final(n, VALIDATE) for n in store.passed(ENV))
    if not pending_env and not pending_val:
        return errors  # nothing to do: no docker needed, no image build
    image = _ensure_image(store, repo, clone, deps)
    if deps.executor_factory is not None:
        executor = deps.executor_factory(image, work)
    else:
        work.mkdir(parents=True, exist_ok=True)
        executor = DockerTestExecutor(
            image, GradingRunner(sandbox_root=work, run_fn=deps.docker_run), repo.test_cmd
        )
    cap = float(repo.limits.max_test_runtime_s)

    def stage(name: str, numbers: list[int], work_fn: Callable[[int], str | None]) -> list[str]:
        track = Tracker(deps, store, repo.name, name, len(numbers))

        def guarded(number: int) -> str | None:
            try:
                return _with_timeout(
                    work_fn, number, timeout_s=deps.pr_timeout_s,
                    on_timeout=_skip_on_timeout(store, log, name, deps.pr_timeout_s),
                )  # fmt: skip
            finally:
                track.tick()

        return _run_parallel(numbers, guarded, jobs)

    errors += stage(ENV, pending_env, lambda n: _env_check(store, clone, work, executor, cap, n))
    if not do_validate:
        return errors
    to_validate = [n for n in store.passed(ENV) if not store.is_final(n, VALIDATE)]
    errors += stage(
        VALIDATE, to_validate, lambda n: _validate_one(store, clone, work, executor, cap, n)
    )
    return errors


def _run_parallel(numbers: list[int], fn: Callable[[int], str | None], jobs: int) -> list[str]:
    """Run ``fn`` over PR numbers, ``jobs`` at a time. Returns the retryable error messages.

    Each PR's work is independent and writes its own stage row, so a crash loses nothing that is
    finished. Unexpected exceptions propagate (a bug must be visible, not logged and forgotten).
    """
    if jobs <= 1:
        results = [fn(n) for n in numbers]
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = [f.result() for f in as_completed([pool.submit(fn, n) for n in numbers])]
    return [r for r in results if r]


def _env_check(
    store: StageStore, clone: Path, work: Path, executor: TestExecutor, cap: float, number: int
) -> str | None:
    pd = _stage_data(store, number, PATCH)
    try:
        factory = _workspace_factory(
            clone, pd["base_commit"], pd["test_patch"], pd["gold_patch"], work, f"env-{number}"
        )
        ws = factory(True)
        try:
            run = executor.run(
                workspace=ws, pytest_args=_test_files(pd["test_patch"]), timeout_s=cap
            )
        finally:
            shutil.rmtree(ws, ignore_errors=True)
    except PatchApplyError as e:
        store.put_stage(number, ENV, DROPPED, PATCH_APPLY_FAILED, {"message": str(e)})
        return None
    except CascadeError as e:
        store.put_stage(number, ENV, ERROR, "env_exception", {"message": str(e)})
        return f"PR {number} env: {e}"
    collected_ok = not any(
        "::" not in nodeid and outcome == OUTCOME_ERROR for nodeid, outcome in run.outcomes.items()
    )
    ran = any("::" in nodeid for nodeid in run.outcomes)
    if run.exit_code not in (0, 1) or run.timed_out or not collected_ok:
        store.put_stage(
            number, ENV, DROPPED, ENV_COLLECT_FAILED,
            {"exit_code": run.exit_code, "tail": run.stdout_tail},
        )  # fmt: skip
    elif not ran:
        store.put_stage(number, ENV, DROPPED, ENV_NO_TESTS, {"tail": run.stdout_tail})
    else:
        store.put_stage(number, ENV, OK)
    return None


def _validate_one(
    store: StageStore, clone: Path, work: Path, executor: TestExecutor, cap: float, number: int
) -> str | None:
    pd = _stage_data(store, number, PATCH)
    factory = _workspace_factory(
        clone, pd["base_commit"], pd["test_patch"], pd["gold_patch"], work, f"val-{number}"
    )
    try:
        result = validate_task(
            executor, factory, runs=3, runtime_cap_s=cap,
            cleanup=lambda p: shutil.rmtree(p, ignore_errors=True),
        )  # fmt: skip
    except PatchApplyError as e:
        store.put_stage(number, VALIDATE, DROPPED, PATCH_APPLY_FAILED, {"message": str(e)})
        return None
    except CascadeError as e:
        store.put_stage(number, VALIDATE, ERROR, "validate_exception", {"message": str(e)})
        return f"PR {number} validate: {e}"
    detail = {
        "flaky": result.flaky,
        "detail": result.detail,
        "test_runtime_s": result.test_runtime_s,
    }
    if result.validated:
        store.put_stage(
            number, VALIDATE, OK,
            data={
                **detail,
                "fail_to_pass": result.fail_to_pass,
                "pass_to_pass": result.pass_to_pass,
            },
        )  # fmt: skip
    else:
        store.put_stage(number, VALIDATE, DROPPED, result.reason, detail)
    return None


def build_tasks(
    store: StageStore, repo: ResolvedRepo, cutoffs: Mapping[str, dt.date | None] | None = None
) -> list[Task]:
    image = store.get_kv("image_tag") or ""
    cands = dict(store.candidates())
    window = mined_window(repo)
    tasks: list[Task] = []
    for number in store.passed(VALIDATE):
        cand = _candidate(cands[number])
        pd = _stage_data(store, number, PATCH)
        vd = _stage_data(store, number, VALIDATE)
        issue = cand.closing_issues[0]
        tasks.append(
            Task(
                task_id=f"{repo.slug}-{number}",
                repo=repo.name,
                base_commit=pd["base_commit"],
                issue_title=issue.title,
                issue_body=issue.body,
                gold_patch=pd["gold_patch"],
                test_patch=pd["test_patch"],
                fail_to_pass=vd["fail_to_pass"],
                pass_to_pass=vd["pass_to_pass"],
                image=image,
                features=compute_features(
                    issue_title=issue.title,
                    issue_body=issue.body,
                    gold_patch=pd["gold_patch"],
                    test_patch=pd["test_patch"],
                    fail_to_pass=vd["fail_to_pass"],
                    pass_to_pass=vd["pass_to_pass"],
                    test_runtime_s=vd["test_runtime_s"],
                ),
                audit=Audit(status="pending"),
                mined_window=window,
                merged_at=cand.merged_at,
                cutoff_clearance=clearance(cand.merged_at, cutoffs or {}),
            )
        )
    return tasks


def write_manifest(
    path: Path,
    tasks: list[Task],
    repo_name: str,
    *,
    window: MinedWindow | None = None,
    replace_window: bool = False,
) -> None:
    """Rewrite this repo's lines in the manifest; other repos, splits, and audits are kept."""
    merge_repo_tasks(path, tasks, repo_name, window=window, replace_window=replace_window)


def refresh_status(store: StageStore, repo: ResolvedRepo, status_md: Path) -> None:
    table = attrition.compute(store)
    block = attrition.render_markdown(
        table,
        repo=repo.name,
        window=(repo.window_start, repo.window_end),
        generated_at=dt.datetime.now(dt.UTC),
        state_path=store.path,
    )
    attrition.update_status_md(status_md, repo.name, block)


def run_pipeline(
    repo: ResolvedRepo,
    *,
    runs_dir: Path,
    manifest_path: Path | None,
    status_md: Path | None,
    stop_after: str = VALIDATE,
    deps: PipelineDeps | None = None,
    jobs: int = 1,
) -> PipelineSummary:
    if stop_after not in STAGE_ORDER:
        raise ValueError(f"stop_after must be one of {STAGE_ORDER}")
    deps = deps or PipelineDeps()
    root = runs_dir / repo.slug
    state = state_dir(runs_dir, repo)
    store = StageStore(state / "state.sqlite")
    log = RunLog(state / "runs.jsonl")
    log.event(
        event="run_start", repo=repo.name, window=[repo.window_start, repo.window_end],
        window_kind=repo.window_kind, stop_after=stop_after, jobs=jobs,
        pr_timeout_s=deps.pr_timeout_s,
    )  # fmt: skip
    summary = PipelineSummary(stage_reached="none")
    reach = STAGE_ORDER.index(stop_after)
    try:
        _check_fingerprint(store, repo)
        meta = check_repo(deps.client_factory(), repo)
        store.set_kv("default_branch", meta.default_branch)
        slices = len(month_slices(repo.window_start, repo.window_end))

        def on_slice(done: int, seen: int) -> None:
            if deps.progress is not None:
                deps.progress(repo.name, "mine", done, slices, store.count(VALIDATE, OK))

        mine(deps.client_factory(), store, repo, today=deps.today, on_slice=on_slice)
        summary.stage_reached = "mine"
        if reach >= 1:
            run_filter(store, repo, meta.default_branch)
            summary.stage_reached = FILTER
        clone = root / "clone.git"  # shared across windows: git data does not depend on them
        if reach >= 2:
            summary.errors += run_patch(store, repo, clone, deps.clone_url(repo.name), deps, log)
            summary.stage_reached = PATCH
        if reach >= 3:
            errs = run_env_and_validate(
                store, repo, clone, state / "work", deps, do_validate=reach >= 4, jobs=jobs, log=log
            )
            summary.errors += errs
            summary.stage_reached = VALIDATE if reach >= 4 else ENV
        if reach >= 4 and manifest_path is not None:
            tasks = build_tasks(store, repo, deps.cutoffs)
            write_manifest(
                manifest_path, tasks, repo.name, window=mined_window(repo),
                replace_window=deps.replace_window,
            )  # fmt: skip
            summary.manifest_tasks = len(tasks)
    finally:
        log.event(
            event="run_end", stage_reached=summary.stage_reached, retryable=len(summary.errors),
            validated=store.count(VALIDATE, OK),
        )  # fmt: skip
        if status_md is not None:
            refresh_status(store, repo, status_md)
        store.close()
    return summary
