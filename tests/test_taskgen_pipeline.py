"""End to end: fake GitHub + real local git upstream + fake docker + fake test container.

Everything here is synthetic. It proves the orchestration (stage order, resumability, attrition
accounting, manifest output), not that Docker or GitHub behave as assumed.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from cascade.config import ResolvedRepo, SelectionLimits
from cascade.errors import CascadeError
from cascade.taskgen.build_env import DockerUnavailableError
from cascade.taskgen.github import GitHubClient
from cascade.taskgen.mine import ConfigMismatchError
from cascade.taskgen.pipeline import (
    PipelineDeps,
    StateMismatchError,
    build_tasks,
    run_pipeline,
    write_manifest,
)
from cascade.taskgen.results import FAILED, PASSED, RunResult
from cascade.taskgen.schema import Task
from cascade.taskgen.store import StageStore
from tests.taskgen_helpers import (
    REPO,
    Upstream,
    build_upstream,
    changed,
    default_files,
    make_pr,
    repo_response,
    search_response,
    to_node,
)

SEARCH_Q = "repo:synthetic/tinylib is:pr is:merged merged:2025-01-01..2025-01-31"


def resolved(**over: Any) -> ResolvedRepo:
    base: dict[str, Any] = {
        "name": REPO,
        "license": "MIT",
        "python_version": "3.12.13",
        "install_cmd": "uv pip install --group tests -e .",
        "test_cmd": "python -m pytest",
        "window_start": dt.date(2025, 1, 1),
        "window_end": dt.date(2025, 1, 31),
        "limits": SelectionLimits(),
    }
    return ResolvedRepo(**{**base, **over})


class FakeGitHub:
    def __init__(self, nodes: list[dict[str, Any]], repo: dict[str, Any]) -> None:
        self.nodes, self.repo, self.search_calls, self.repo_calls = nodes, repo, 0, 0

    def __call__(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        if "repository(" in query:
            self.repo_calls += 1
            return self.repo
        self.search_calls += 1
        if variables["q"] == SEARCH_Q and variables["after"] is None:
            return search_response(self.nodes)
        return search_response([])


class FakeDocker:
    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.built: set[str] = set()
        self.calls: list[list[str]] = []

    def __call__(
        self, argv: Sequence[str], timeout: float | None
    ) -> subprocess.CompletedProcess[str]:
        a = list(argv)
        self.calls.append(a)
        rc = 0
        if a[1] == "info":
            rc = 0 if self.available else 1
        elif a[1] == "image":
            rc = 0 if a[-1] in self.built else 1
        elif a[1] == "build":
            self.built.add(a[a.index("--tag") + 1])
        return subprocess.CompletedProcess(a, rc, stdout="", stderr="")

    def count(self, verb: str) -> int:
        return sum(1 for c in self.calls if c[1] == verb)


class FakeExecutor:
    """Decides outcomes from the workspace contents, so it only works if patches were applied."""

    def __init__(self, *, flaky: bool = False, fail_first: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.flaky, self.fail_first = flaky, fail_first

    def run(self, *, workspace: Path, pytest_args: Sequence[str], timeout_s: float) -> RunResult:
        self.calls.append(list(pytest_args))
        if self.fail_first and len(self.calls) == 1:
            raise CascadeError("docker hiccup")
        assert "test_above_range" in (workspace / "tests/test_clamp.py").read_text()  # test patch
        gold = "return high" in (workspace / "src/tinylib/clamp.py").read_text()
        t = "tests/test_clamp.py::"
        outcomes = {
            t + "test_inside_range": PASSED,
            t + "test_below_range": PASSED,
            t + "test_above_range": PASSED if gold else FAILED,
        }
        if self.flaky:
            outcomes[t + "test_wobble"] = PASSED if len(self.calls) % 2 else FAILED
        return RunResult(0 if gold else 1, 2.0, False, outcomes, "")


@pytest.fixture
def up(tmp_path: Path) -> Upstream:
    return build_upstream(tmp_path)


def nodes_for(up: Upstream) -> list[dict[str, Any]]:
    good = make_pr(merge_commit=up.merge_sha)
    bot = make_pr(
        number=102, merge_commit=up.merge_sha, author_login="dependabot", author_type="Bot"
    )
    no_commit = make_pr(number=103, merge_commit="9" * 40)
    mismatch = make_pr(
        number=104, merge_commit=up.merge_sha, files=[*default_files(), changed("src/tinylib/x.py")]
    )
    return [to_node(p) for p in (good, bot, no_commit, mismatch)]


class Env:
    def __init__(self, tmp_path: Path, up: Upstream, **exec_kw: bool) -> None:
        self.tmp, self.up = tmp_path, up
        self.gh = FakeGitHub(nodes_for(up), repo_response("MIT", "main"))
        self.docker = FakeDocker()
        self.executor = FakeExecutor(**exec_kw)
        self.runs = tmp_path / "runs"
        self.manifest = tmp_path / "tasks" / "manifest.jsonl"
        self.status = tmp_path / "STATUS.md"

    def deps(self) -> PipelineDeps:
        return PipelineDeps(
            client_factory=lambda: GitHubClient(transport=self.gh),
            docker_run=self.docker,
            clone_url=lambda _repo: str(self.up.path),
            executor_factory=lambda _image, _work: self.executor,
            today=dt.date(2025, 6, 1),
        )

    def run(self, repo: ResolvedRepo | None = None, **kw: Any):
        return run_pipeline(
            repo or resolved(), runs_dir=self.runs, manifest_path=self.manifest,
            status_md=self.status, deps=self.deps(), **kw,
        )  # fmt: skip


@pytest.fixture
def env(tmp_path: Path, up: Upstream) -> Env:
    return Env(tmp_path, up)


def test_full_run_produces_one_validated_task_and_an_honest_attrition_table(env: Env) -> None:
    summary = env.run()
    assert (
        summary.stage_reached == "validate" and summary.manifest_tasks == 1 and summary.errors == []
    )

    [line] = env.manifest.read_text().splitlines()
    task = Task.model_validate_json(line)
    assert task.task_id == "synthetic__tinylib-101" and task.repo == REPO
    assert task.base_commit == env.up.base_sha
    assert task.fail_to_pass == ["tests/test_clamp.py::test_above_range"]
    assert task.pass_to_pass == [
        "tests/test_clamp.py::test_below_range",
        "tests/test_clamp.py::test_inside_range",
    ]
    assert task.issue_title.startswith("clamp()") and "expected 10" in task.issue_body
    assert "return high" in task.gold_patch and "test_above_range" in task.test_patch
    assert task.image.startswith("cascade-env/synthetic__tinylib:")
    assert task.features.src_files == 1 and task.features.src_lines == 2
    assert task.features.test_runtime_s == 2.0 and task.audit.status == "pending"
    assert task.split is None  # splitting is a later, frozen step

    status = env.status.read_text()
    assert "4 candidates -> 1 after filters -> 1 env builds -> 1 validated" in status
    assert "bot_author (1)" in status
    assert "merge_commit_missing (1)" in status and "diff_mismatch (1)" in status
    assert "Window 2025-01-01 to 2025-01-31" in status


def test_run_counts_exactly_one_env_check_and_six_validation_runs(env: Env) -> None:
    env.run()
    assert len(env.executor.calls) == 7  # 1 env smoke run + 3 pre + 3 post
    assert env.executor.calls[0] == ["tests/test_clamp.py"]  # env check runs the patched test file
    assert env.docker.count("build") == 1


def test_rerun_skips_everything_that_is_already_done(env: Env) -> None:
    env.run()
    first_manifest = env.manifest.read_text()
    searches, runs, builds = env.gh.search_calls, len(env.executor.calls), env.docker.count("build")

    summary = env.run()
    assert env.gh.search_calls == searches  # finished month slice is not refetched
    assert len(env.executor.calls) == runs  # no test run repeated
    assert env.docker.count("build") == builds  # image reused
    assert env.manifest.read_text() == first_manifest and summary.manifest_tasks == 1
    assert (
        "4 candidates -> 1 after filters -> 1 env builds -> 1 validated" in env.status.read_text()
    )


def test_stop_after_patch_needs_no_docker_and_writes_no_manifest(env: Env) -> None:
    env.docker.available = False
    summary = env.run(stop_after="patch")
    assert summary.stage_reached == "patch" and not env.manifest.exists()
    assert env.docker.calls == [] and env.executor.calls == []
    assert (
        "4 candidates -> 1 after filters -> 0 env builds -> 0 validated" in env.status.read_text()
    )


def test_docker_unavailable_keeps_finished_stages_and_shows_pending(env: Env) -> None:
    env.docker.available = False
    with pytest.raises(DockerUnavailableError, match="docker daemon"):
        env.run()
    status = env.status.read_text()
    assert "4 candidates -> 1 after filters -> 0 env builds -> 0 validated" in status
    assert "**Projection:** none" in status  # incomplete, so no extrapolation

    env.docker.available = True  # start Docker and rerun: resumes where it stopped
    searches = env.gh.search_calls
    summary = env.run()
    assert summary.manifest_tasks == 1 and env.gh.search_calls == searches


def test_transient_executor_failure_is_an_error_row_that_a_rerun_retries(
    tmp_path: Path, up: Upstream
) -> None:
    env = Env(tmp_path, up, fail_first=True)
    first = env.run()
    assert first.manifest_tasks == 0 and any("docker hiccup" in e for e in first.errors)
    assert "Error (retry)" in env.status.read_text()

    second = env.run()  # the executor works now
    assert second.errors == [] and second.manifest_tasks == 1


def test_flaky_tests_are_excluded_and_recorded(tmp_path: Path, up: Upstream) -> None:
    env = Env(tmp_path, up, flaky=True)
    env.run()
    task = Task.model_validate_json(env.manifest.read_text().splitlines()[0])
    assert all("wobble" not in t for t in task.fail_to_pass + task.pass_to_pass)
    store = StageStore(env.runs / "synthetic__tinylib" / "state.sqlite")
    stage = store.get_stage(101, "validate")
    assert stage is not None and stage[2]["flaky"] == ["tests/test_clamp.py::test_wobble"]
    store.close()


def test_changing_the_config_against_old_state_is_refused(env: Env) -> None:
    env.run()
    with pytest.raises(StateMismatchError, match="different config"):
        env.run(resolved(limits=SelectionLimits(max_source_files=4)))


def test_license_mismatch_stops_before_any_mining(env: Env) -> None:
    env.gh.repo = repo_response("Apache-2.0", "main")
    with pytest.raises(ConfigMismatchError):
        env.run()
    assert env.gh.search_calls == 0


def test_invalid_stop_after_is_rejected(env: Env) -> None:
    with pytest.raises(ValueError, match="stop_after"):
        env.run(stop_after="deploy")


# --- manifest --------------------------------------------------------------------------------


def task_json(task_id: str, repo: str) -> str:
    return json.dumps({"task_id": task_id, "repo": repo})


def test_write_manifest_replaces_only_this_repos_lines(tmp_path: Path, env: Env) -> None:
    env.run()
    store = StageStore(env.runs / "synthetic__tinylib" / "state.sqlite")
    tasks = build_tasks(store, resolved())
    store.close()

    path = tmp_path / "m.jsonl"
    path.write_text(
        task_json("other__repo-1", "other/repo")
        + "\n"
        + task_json("synthetic__tinylib-999", REPO)
        + "\n"
    )
    write_manifest(path, tasks, REPO)
    rows = [json.loads(ln) for ln in path.read_text().splitlines()]
    assert [r["task_id"] for r in rows] == ["other__repo-1", "synthetic__tinylib-101"]  # sorted
    assert not path.with_suffix(".jsonl.tmp").exists()  # atomic: no temp file left behind
