"""Orchestrates mine -> filter -> patch -> env -> validate for one repo, resumably.

Every stage reads and writes the SQLite StageStore, so rerunning skips every PR whose stage result
is final (ok or dropped) and retries only "error" rows. The state is tied to a fingerprint of the
repo config; changing the window, limits, or commands against an old state is refused rather than
mixing results from two configurations.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import shutil
import uuid
from collections.abc import Callable
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
from cascade.taskgen.filter import FilterContext, filter_metadata
from cascade.taskgen.github import GitHubClient, PRCandidate
from cascade.taskgen.gitrepo import (
    GitError,
    PatchApplyError,
    apply_patch,
    derive_patches,
    ensure_clone,
)
from cascade.taskgen.mine import check_repo, mine
from cascade.taskgen.results import ERROR as OUTCOME_ERROR
from cascade.taskgen.schema import Audit, Features, Task
from cascade.taskgen.store import DROPPED, ENV, ERROR, FILTER, OK, PATCH, VALIDATE, StageStore
from cascade.taskgen.validate import PATCH_APPLY_FAILED, TestExecutor, validate_task

STAGE_ORDER = ("mine", FILTER, PATCH, ENV, VALIDATE)
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


def run_patch(store: StageStore, repo: ResolvedRepo, clone: Path, url: str) -> None:
    todo = [n for n in store.passed(FILTER) if not store.is_final(n, PATCH)]
    if not todo:
        return
    ensure_clone(url, clone)
    cands = dict(store.candidates())
    for number in todo:
        try:
            outcome = derive_patches(clone, _candidate(cands[number]), repo.limits)
        except GitError as e:
            store.put_stage(number, PATCH, ERROR, "git_error", {"message": str(e)})
            continue
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

    for number in pending_env:
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
            continue
        except CascadeError as e:
            store.put_stage(number, ENV, ERROR, "env_exception", {"message": str(e)})
            errors.append(f"PR {number} env: {e}")
            continue
        collected_ok = not any(
            "::" not in nodeid and outcome == OUTCOME_ERROR
            for nodeid, outcome in run.outcomes.items()
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

    if not do_validate:
        return errors
    for number in [n for n in store.passed(ENV) if not store.is_final(n, VALIDATE)]:
        pd = _stage_data(store, number, PATCH)
        factory = _workspace_factory(
            clone, pd["base_commit"], pd["test_patch"], pd["gold_patch"], work, f"val-{number}"
        )
        try:
            result = validate_task(
                executor,
                factory,
                runs=3,
                runtime_cap_s=cap,
                cleanup=lambda p: shutil.rmtree(p, ignore_errors=True),
            )
        except PatchApplyError as e:
            store.put_stage(number, VALIDATE, DROPPED, PATCH_APPLY_FAILED, {"message": str(e)})
            continue
        except CascadeError as e:
            store.put_stage(number, VALIDATE, ERROR, "validate_exception", {"message": str(e)})
            errors.append(f"PR {number} validate: {e}")
            continue
        detail = {
            "flaky": result.flaky,
            "detail": result.detail,
            "test_runtime_s": result.test_runtime_s,
        }
        if result.validated:
            store.put_stage(
                number,
                VALIDATE,
                OK,
                data={
                    **detail,
                    "fail_to_pass": result.fail_to_pass,
                    "pass_to_pass": result.pass_to_pass,
                },
            )
        else:
            store.put_stage(number, VALIDATE, DROPPED, result.reason, detail)
    return errors


def build_tasks(store: StageStore, repo: ResolvedRepo) -> list[Task]:
    image = store.get_kv("image_tag") or ""
    cands = dict(store.candidates())
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
                features=Features(
                    src_files=pd["src_files"],
                    src_lines=pd["src_lines"],
                    issue_chars=len(issue.title) + len(issue.body),
                    test_runtime_s=vd["test_runtime_s"],
                ),
                audit=Audit(status="pending"),
            )
        )
    return tasks


def write_manifest(path: Path, tasks: list[Task], repo_name: str) -> None:
    """Rewrite this repo's lines in the manifest, keeping other repos' lines. Atomic."""
    kept: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["repo"] != repo_name:
                    kept[row["task_id"]] = line
    for task in tasks:
        kept[task.task_id] = task.model_dump_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(kept[k] + "\n" for k in sorted(kept)), encoding="utf-8")
    tmp.replace(path)


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
) -> PipelineSummary:
    if stop_after not in STAGE_ORDER:
        raise ValueError(f"stop_after must be one of {STAGE_ORDER}")
    deps = deps or PipelineDeps()
    root = runs_dir / repo.slug
    store = StageStore(root / "state.sqlite")
    summary = PipelineSummary(stage_reached="none")
    reach = STAGE_ORDER.index(stop_after)
    try:
        _check_fingerprint(store, repo)
        meta = check_repo(deps.client_factory(), repo)
        store.set_kv("default_branch", meta.default_branch)
        mine(deps.client_factory(), store, repo, today=deps.today)
        summary.stage_reached = "mine"
        if reach >= 1:
            run_filter(store, repo, meta.default_branch)
            summary.stage_reached = FILTER
        clone = root / "clone.git"
        if reach >= 2:
            run_patch(store, repo, clone, deps.clone_url(repo.name))
            summary.stage_reached = PATCH
        if reach >= 3:
            errs = run_env_and_validate(
                store, repo, clone, root / "work", deps, do_validate=reach >= 4
            )
            summary.errors += errs
            summary.stage_reached = VALIDATE if reach >= 4 else ENV
        if reach >= 4 and manifest_path is not None:
            tasks = build_tasks(store, repo)
            write_manifest(manifest_path, tasks, repo.name)
            summary.manifest_tasks = len(tasks)
    finally:
        if status_md is not None:
            refresh_status(store, repo, status_md)
        store.close()
    return summary
