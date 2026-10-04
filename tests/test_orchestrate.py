"""Multi-repo orchestration: parallel, resumable, isolated failures, single-writer outputs.

Fake GitHub, real local git upstreams, fake docker and test executor. All SYNTHETIC; this shows the
orchestration works, not that GitHub or Docker behave as assumed.
"""

from __future__ import annotations

import datetime as dt
import threading
from pathlib import Path

import pytest

from cascade.config import ResolvedRepo
from cascade.errors import CascadeError
from cascade.taskgen.github import GitHubClient
from cascade.taskgen.manifest import read_manifest
from cascade.taskgen.orchestrate import (
    RepoOutcome,
    format_report,
    repo_funnel,
    run_all,
    state_path,
)
from cascade.taskgen.pipeline import PipelineDeps
from tests.phase3_helpers import MultiRepoGitHub, build_upstreams, resolved_repo
from tests.test_taskgen_pipeline import FakeDocker, FakeExecutor

REPOS = ["synthetic/alpha", "synthetic/beta", "synthetic/gamma"]


class Fixture:
    def __init__(self, tmp_path: Path, repos: list[str]) -> None:
        self.tmp = tmp_path
        self.repos = [resolved_repo(r) for r in repos]
        self.ups = build_upstreams(tmp_path, repos)
        self.gh = MultiRepoGitHub({r: u.merge_sha for r, u in self.ups.items()})
        self.docker = FakeDocker()
        self.executor = FakeExecutor()
        self.runs = tmp_path / "runs"
        self.manifest = tmp_path / "tasks" / "manifest.jsonl"
        self.status = tmp_path / "STATUS.md"
        self.lock = threading.Lock()
        self.clone_calls: list[str] = []
        self.broken: set[str] = set()

    def clone_url(self, repo: str) -> str:
        with self.lock:
            self.clone_calls.append(repo)
        if repo in self.broken:
            raise CascadeError(f"cannot clone {repo}")
        return str(self.ups[repo].path)

    def deps_for(self, _repo: ResolvedRepo) -> PipelineDeps:
        return PipelineDeps(
            client_factory=lambda: GitHubClient(transport=self.gh),
            docker_run=self.docker,
            clone_url=self.clone_url,
            executor_factory=lambda _image, _work: self.executor,
            today=dt.date(2025, 6, 1),
        )

    def run(
        self,
        repos: list[ResolvedRepo] | None = None,
        *,
        workers: int = 4,
        jobs: int = 2,
        stop_after: str = "validate",
    ) -> list[RepoOutcome]:
        return run_all(
            repos if repos is not None else self.repos,
            runs_dir=self.runs,
            manifest_path=self.manifest,
            status_md=self.status,
            stop_after=stop_after,
            workers=workers,
            jobs=jobs,
            deps_for=self.deps_for,
        )


@pytest.fixture
def fx(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path, REPOS)


def test_every_repo_is_processed_and_the_manifest_has_each_repos_task(fx: Fixture) -> None:
    outcomes = fx.run(workers=3, jobs=2)
    assert [o.repo for o in outcomes] == REPOS and all(o.error is None for o in outcomes)
    assert all(o.summary is not None and o.summary.stage_reached == "validate" for o in outcomes)
    tasks = read_manifest(fx.manifest)
    assert sorted(t.task_id for t in tasks) == [
        "synthetic__alpha-101", "synthetic__beta-101", "synthetic__gamma-101",
    ]  # fmt: skip
    assert all(t.features.f2p_count == 1 and t.features.src_added >= 1 for t in tasks)


def test_each_repo_gets_its_own_state_and_status_block(fx: Fixture) -> None:
    fx.run()
    for repo in fx.repos:
        assert state_path(fx.runs, repo).exists()
    status = fx.status.read_text()
    assert all(r in status for r in REPOS)


def test_results_do_not_depend_on_the_degree_of_parallelism(tmp_path: Path) -> None:
    serial = Fixture(tmp_path / "s", REPOS)
    serial.run(workers=1, jobs=1)
    parallel = Fixture(tmp_path / "p", REPOS)
    parallel.run(workers=3, jobs=3)

    def strip(path: Path) -> list[tuple[object, ...]]:
        # shas and patch index lines differ between two separately built git upstreams
        return [
            (t.task_id, t.repo, t.issue_title, t.fail_to_pass, t.pass_to_pass, t.features)
            for t in read_manifest(path)
        ]

    assert strip(serial.manifest) == strip(parallel.manifest)


def test_rerun_does_no_repeated_work(fx: Fixture) -> None:
    fx.run()
    first = fx.manifest.read_text()
    searches, runs, builds = fx.gh.search_calls, len(fx.executor.calls), fx.docker.count("build")
    outcomes = fx.run()
    assert all(o.error is None for o in outcomes)
    assert fx.gh.search_calls == searches and len(fx.executor.calls) == runs
    assert fx.docker.count("build") == builds
    assert fx.manifest.read_text() == first


def test_one_failing_repo_does_not_stop_the_others_and_is_reported(fx: Fixture) -> None:
    fx.broken.add("synthetic/beta")
    outcomes = {o.repo: o for o in fx.run(workers=3)}
    assert outcomes["synthetic/beta"].error and "cannot clone" in outcomes["synthetic/beta"].error
    assert outcomes["synthetic/beta"].summary is None
    assert outcomes["synthetic/alpha"].error is None and outcomes["synthetic/gamma"].error is None
    assert sorted(t.repo for t in read_manifest(fx.manifest)) == [
        "synthetic/alpha", "synthetic/gamma",
    ]  # fmt: skip

    fx.broken.clear()  # fixed; rerun resumes and completes beta
    again = fx.run()
    assert all(o.error is None for o in again)
    assert len(read_manifest(fx.manifest)) == 3


def test_rerunning_one_repo_leaves_the_others_manifest_lines_alone(fx: Fixture) -> None:
    fx.run()
    others = {t.task_id: t for t in read_manifest(fx.manifest) if t.repo != "synthetic/beta"}
    fx.run([fx.repos[1]])
    now = {t.task_id: t for t in read_manifest(fx.manifest)}
    assert len(now) == 3 and all(now[k] == v for k, v in others.items())


def test_stop_after_patch_writes_no_manifest(fx: Fixture) -> None:
    fx.docker.available = False  # patch stage needs no docker
    outcomes = fx.run(stop_after="patch")
    assert all(o.error is None for o in outcomes) and not fx.manifest.exists()
    assert fx.docker.calls == []


def test_no_repos_is_a_noop(fx: Fixture) -> None:
    assert fx.run([]) == [] and not fx.manifest.exists()


def test_workers_are_capped_by_the_number_of_repos(fx: Fixture) -> None:
    assert [o.repo for o in fx.run([fx.repos[0]], workers=16)] == [REPOS[0]]


# --- reporting ---------------------------------------------------------------------------


def test_funnel_is_none_until_a_repo_has_state(fx: Fixture) -> None:
    assert repo_funnel(fx.repos[0], fx.runs) is None
    fx.run()
    f = repo_funnel(fx.repos[0], fx.runs)
    assert f is not None
    assert (f.candidates, f.after_filter, f.after_patch, f.env_ok, f.validated) == (1, 1, 1, 1, 1)
    assert f.errors == 0 and f.pending == 0 and f.reasons == {}


def test_report_has_per_stage_attrition_totals_and_per_repo_counts(fx: Fixture) -> None:
    fx.run()
    funnels = [f for r in fx.repos if (f := repo_funnel(r, fx.runs)) is not None]
    text = format_report(funnels, split_counts={"synthetic/alpha": {"train": 1}})
    assert "PER-STAGE ATTRITION" in text and "TASKS PER REPO" in text
    total = next(line for line in text.splitlines() if line.startswith("TOTAL"))
    assert total.split()[1:] == ["3", "3", "3", "3", "3", "0", "0"]
    assert "synthetic/alpha" in text and "[train 1]" in text
    assert "DROP REASONS" not in text  # nothing was dropped


def test_report_lists_missing_repos_and_early_stops() -> None:
    text = format_report(
        [], missing=["o/never"], outcomes=[RepoOutcome("o/bad", None, "CascadeError: boom")]
    )
    assert "o/never" in text and "(no state yet)" in text
    assert "STOPPED EARLY" in text and "o/bad: CascadeError: boom" in text
