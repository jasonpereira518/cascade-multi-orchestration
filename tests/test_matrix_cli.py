"""End-to-end dry run of the experiment commands through the real Typer app, on the mock provider.

pilot -> size -> calibrate -> matrix -> ablate -> repeat-subset -> stats -> export, plus the guards
around real-provider runs. Only Docker is stubbed (daemon probe and executor factory, as in
test_cli_run.py); configs, manifest, audit log, clones, ledger, matrix, grading, and CLI output are
real. Everything is SYNTHETIC: six tasks over two repo names, each backed by a copy of the
synthetic fixture repo, mock models, invented prices. It proves the plumbing, not any result.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq  # pyright: ignore[reportMissingTypeStubs]
import pytest
import typer
from typer.testing import CliRunner, Result

from cascade import matrix_cli
from cascade.agents import runner as agent_runner
from cascade.cli import app
from cascade.harness.sandbox import ContainerRunner
from cascade.routing.matrix import MatrixRow, MatrixStore
from cascade.taskgen.audit import append_verdict, make_verdict
from cascade.taskgen.manifest import write_tasks
from cascade.taskgen.schema import Task
from tests.agent_helpers import NativeExecutor
from tests.conftest import ROOT, SYNTHETIC, FixtureRepo
from tests.matrix_helpers import MATRIX_BUDGET, MATRIX_MODELS

runner = CliRunner()
NOW = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
REPOS = ("synthetic/tinylib", "synthetic/tinylib2")
TRAIN = ["t0", "t1", "t2", "t3"]
TEST = ["t4", "t5"]


@dataclass
class Env:
    root: Path
    configs: Path
    manifest: Path
    audit: Path
    runs: Path
    taskgen: Path
    kill: Path
    tasks: dict[str, Task]

    def opts(self, *names: str) -> list[str]:
        table = {
            "configs": ["--configs-dir", str(self.configs)],
            "manifest": ["--manifest", str(self.manifest)],
            "audit": ["--audit-file", str(self.audit)],
            "runs": ["--runs-dir", str(self.runs)],
            "taskgen": ["--taskgen-runs-dir", str(self.taskgen)],
        }
        return [arg for n in names for arg in table[n]]

    def run_opts(self) -> list[str]:
        return self.opts("configs", "manifest", "audit", "runs", "taskgen")

    def invoke(self, args: list[str]) -> Result:
        return runner.invoke(app, args)

    @property
    def ledger_db(self) -> Path:
        return self.runs / "ledger.sqlite"

    def sql(self, query: str, *params: object) -> list[tuple[object, ...]]:
        conn = sqlite3.connect(self.ledger_db)
        try:
            return [tuple(r) for r in conn.execute(query, params).fetchall()]
        finally:
            conn.close()

    def count_calls(self) -> int:
        return int(str(self.sql("SELECT COUNT(*) FROM calls")[0][0]))

    def matrix_rows(self, db: Path | None = None) -> list[MatrixRow]:
        store = MatrixStore(db or self.runs / "matrix.sqlite")
        try:
            return store.rows()
        finally:
            store.close()


def build(tmp_path: Path, fixture_repo: FixtureRepo, monkeypatch: pytest.MonkeyPatch) -> Env:
    configs = tmp_path / "configs"
    configs.mkdir()
    shutil.copy(MATRIX_MODELS, configs / "models.yaml")
    kill = tmp_path / "KILL_SWITCH"
    text = MATRIX_BUDGET.read_text()
    budget = text.replace("kill_switch_path: KILL_SWITCH", f"kill_switch_path: {kill}")
    (configs / "budget.yaml").write_text(budget)
    (configs / "repos.yaml").write_text(
        "date_window: {start: 2025-01-01, end: 2025-12-31}\nrepos:\n"
        + "".join(
            f"  - name: {r}\n    license: MIT\n    python_version: '3.12'\n"
            "    install_cmd: none\n    test_cmd: python -m pytest\n"
            for r in REPOS
        )
    )
    taskgen = tmp_path / "runs" / "taskgen"
    for r in REPOS:
        shutil.copytree(fixture_repo.path, taskgen / r.replace("/", "__") / "clone.git")

    tasks: dict[str, Task] = {}
    for i, task_id in enumerate([*TRAIN, *TEST]):
        base = fixture_repo.task
        features = base.features.model_copy(update={"src_lines": 5 + i * 10})
        tasks[task_id] = base.model_copy(
            update={
                "task_id": task_id,
                "repo": REPOS[i % 2],
                "split": "train" if task_id in TRAIN else "test",
                "features": features,
            }
        )
    manifest = tmp_path / "tasks" / "manifest.jsonl"
    write_tasks(manifest, list(tasks.values()))
    audit = tmp_path / "tasks" / "audit.jsonl"
    for task_id in tasks:
        append_verdict(audit, make_verdict(task_id, "ok", now=NOW))

    monkeypatch.setattr(matrix_cli, "docker_available", lambda: True)
    monkeypatch.setattr(
        agent_runner, "docker_executor_factory", lambda task, test_cmd: lambda r: _native(r)
    )
    return Env(tmp_path, configs, manifest, audit, tmp_path / "runs", taskgen, kill, tasks)


def _native(r: ContainerRunner) -> NativeExecutor:
    return NativeExecutor(r)


@pytest.fixture
def env(tmp_path: Path, fixture_repo: FixtureRepo, monkeypatch: pytest.MonkeyPatch) -> Env:
    return build(tmp_path, fixture_repo, monkeypatch)


def out(result: Result) -> str:
    return result.output


# --- the end-to-end dry run ---------------------------------------------------------------------


def test_dry_run_pilot_size_calibrate_matrix_ablate_repeat_stats_export(env: Env) -> None:
    # 1. pilot: train tasks only, every arm, standard tier, bucket "pilot"
    r = env.invoke(["pilot", "--n", "2", *env.run_opts()])
    assert r.exit_code == 0, out(r)
    assert "pilot: 2 train task(s), 3 arm(s), standard tier, provider mock" in out(r)
    assert out(r).count("PILOT RESULTS") == 1 and out(r).count("all arms |") == 2
    pilot_tasks = {t for (t,) in env.sql("SELECT DISTINCT task_id FROM calls WHERE bucket='pilot'")}
    assert len(pilot_tasks) == 2 and pilot_tasks <= set(TRAIN)  # never a test task
    assert {x.processing_tier for x in env.matrix_rows()} == {"standard"}
    assert len(env.matrix_rows()) == 6
    pilot_status = json.loads((env.runs / "pilot_status.json").read_text())
    assert pilot_status["state"] == "complete"

    # 2. size: options printed, nothing chosen, no forced decision at this budget
    r = env.invoke(["size", *env.opts("configs", "manifest", "audit", "runs")])
    assert r.exit_code == 0, out(r)
    for needle in ("SIZING", "OPTIONS", "full 3-arm matrix", "flex", "drop the mid arm"):
        assert needle in out(r), needle
    assert "subset of tasks" in out(r) and "audited-ok tasks: 6" in out(r)
    assert "DECISION REQUIRED" not in out(r)

    # 3. calibrate: the all-mock pilot fixes nothing, so both flags fire (and nothing is switched)
    r = env.invoke(["calibrate", *env.opts("configs", "runs")])
    assert r.exit_code == 0, out(r)
    assert "CALIBRATION on 2 pilot task(s)" in out(r)
    assert "CHEAP ARM NEAR ZERO" in out(r) and "MAX ARM OUTSIDE 40-70%" in out(r)
    assert "Nothing is switched automatically" in out(r)

    # 4. matrix: pilot cells are reused, the rest run; train and test tasks are all covered
    r = env.invoke(["matrix", "run", *env.run_opts()])
    assert r.exit_code == 0, out(r)
    assert "complete: 18/18 cells done (12 this run)" in out(r)
    status = json.loads((env.runs / "matrix_status.json").read_text())
    assert status["state"] == "complete" and status["planned"] == 18 and status["executed"] == 12
    assert status["done_by_arm"] == {
        "mock-luna@high": 6, "mock-luna@low": 6, "mock-sol@medium": 6,
    }  # fmt: skip
    assert {x.task_id for x in env.matrix_rows()} == set(env.tasks)
    again = env.invoke(["matrix", "run", *env.run_opts()])
    assert "(0 this run)" in out(again) and again.exit_code == 0  # idempotent
    shown = env.invoke(["matrix", "status", *env.opts("runs")])
    assert json.loads(out(shown))["state"] == "complete"

    # 5. ablate: no reproducer / no reviewer on train tasks only, paired with the matrix baseline
    r = env.invoke(["ablate", "--n", "3", *env.run_opts()])
    assert r.exit_code == 0, out(r)
    assert "ABLATIONS (mid arm)" in out(r)
    assert "no_reproducer vs full" in out(r) and "no_reviewer vs full" in out(r)
    assert "(0 paired tasks" not in out(r)
    sql = "SELECT DISTINCT task_id FROM calls WHERE bucket='ablate'"
    ablate_tasks = {t for (t,) in env.sql(sql)}
    assert len(ablate_tasks) == 3 and ablate_tasks <= set(TRAIN)
    configs = {
        x.pipeline_config.replace(" ", "") for x in env.matrix_rows() if x.task_id in ablate_tasks
    }
    assert len(configs) == 3  # the full pipeline plus the two ablations, recorded per row
    assert any('"reproducer":false' in c for c in configs)
    assert any('"reviewer":false' in c for c in configs)

    # 6. repeat-subset: cheap and mid only, second seed, train only
    r = env.invoke(["repeat-subset", "--n", "2", *env.run_opts()])
    assert r.exit_code == 0, out(r)
    assert "REPEAT (noise estimate)" in out(r) and "outcome agreement" in out(r)
    repeat = [x for x in env.matrix_rows() if x.seed == 1]
    assert len(repeat) == 4 and {x.arm_id for x in repeat} == {"mock-luna@low", "mock-luna@high"}
    assert {x.task_id for x in repeat} <= set(TRAIN)

    # 7. stats: every number from an artifact
    probe = env.root / "probe"
    probe.mkdir()
    (probe / "test_probe.py").write_text("def test_one(): pass\n")
    stats_args = [*env.opts("configs", "manifest", "audit", "runs"), "--repo-root", str(probe)]
    r = env.invoke(["stats", *stats_args])
    assert r.exit_code == 0, out(r)
    rows = env.matrix_rows()
    assert f"total attempts (matrix rows): {len(rows)}" in out(r) and len(rows) == 18 + 6 + 4
    assert "audited-ok tasks: 6 (train 4, test 2)" in out(r)
    assert "tests in this repository (pytest --collect-only): 1" in out(r)
    for bucket in ("pilot", "matrix", "ablate", "repeat"):
        assert f"{bucket}: $" in out(r), bucket

    # 8. export: parquet is a straight copy
    r = env.invoke(["export", *env.opts("runs")])
    assert r.exit_code == 0, out(r)
    exported = env.runs / "export"
    assert pq.read_table(exported / "matrix.parquet").num_rows == len(rows)
    assert pq.read_table(exported / "ledger.parquet").num_rows == env.count_calls()
    costs = [x["cost_usd"] for x in pq.read_table(exported / "matrix.parquet").to_pylist()]
    assert costs == [x.cost_usd for x in rows]

    # rule 1 and 2: every matrix cost is a sum of logged calls, and all of it was mock money
    ledger_total = float(str(env.sql("SELECT SUM(cost_usd) FROM calls")[0][0]))
    assert sum(x.cost_usd for x in rows) == pytest.approx(ledger_total)
    assert {p for (p,) in env.sql("SELECT DISTINCT provider FROM calls")} == {"mock"}
    # the test split was reached only by the matrix, never by pilot/ablate/repeat
    for bucket in ("pilot", "ablate", "repeat"):
        used = {t for (t,) in env.sql("SELECT DISTINCT task_id FROM calls WHERE bucket=?", bucket)}
        assert used <= set(TRAIN), bucket


# --- sizing: the forced decision ----------------------------------------------------------------


def test_size_forces_a_decision_when_fewer_than_40_tasks_are_affordable(env: Env) -> None:
    env.invoke(["pilot", "--n", "2", *env.run_opts()])
    tight = (env.configs / "budget.yaml").read_text().replace("matrix: 20.0", "matrix: 0.05")
    (env.configs / "budget.yaml").write_text(tight)
    r = env.invoke(["size", *env.opts("configs", "manifest", "audit", "runs")])
    assert r.exit_code == 0, out(r)
    assert "DECISION REQUIRED" in out(r) and "No option has been chosen" in out(r)


# --- guards -------------------------------------------------------------------------------------


def test_commands_that_need_a_pilot_say_so_when_there_is_none(env: Env) -> None:
    size = ["size", *env.opts("configs", "manifest", "audit", "runs")]
    calibrate = ["calibrate", *env.opts("configs", "runs")]
    for args in (size, calibrate):
        r = env.invoke(args)
        assert r.exit_code == 2 and "run `cascade pilot` first" in out(r)
    assert env.invoke(["matrix", "status", *env.opts("runs")]).exit_code == 2


def test_no_audited_ok_tasks_means_nothing_runs(env: Env) -> None:
    env.audit.write_text("")
    for args in (["pilot", *env.run_opts()], ["matrix", "run", *env.run_opts()]):
        r = env.invoke(args)
        assert r.exit_code == 2, out(r)
    pre = env.invoke(["preflight", *env.opts("manifest", "audit", "runs", "taskgen")])
    assert pre.exit_code == 2 and "no audited-ok tasks" in out(pre)


def test_the_matrix_refuses_audited_ok_tasks_that_have_no_split(env: Env) -> None:
    write_tasks(env.manifest, [t.model_copy(update={"split": None}) for t in env.tasks.values()])
    r = env.invoke(["matrix", "run", *env.run_opts()])
    assert r.exit_code == 2 and "no split" in out(r)
    p = env.invoke(["pilot", *env.run_opts()])
    assert p.exit_code == 2 and "no split" in out(p)


def test_the_kill_switch_file_stops_the_matrix_and_removing_it_resumes(env: Env) -> None:
    env.kill.touch()
    r = env.invoke(["matrix", "run", *env.run_opts()])
    assert r.exit_code == 0 and "reason: kill_switch" in out(r) and "(0 this run)" in out(r)
    assert json.loads((env.runs / "matrix_status.json").read_text())["state"] == "stopped"
    env.kill.unlink()
    r = env.invoke(["matrix", "run", *env.run_opts()])
    assert r.exit_code == 0 and "complete: 18/18 cells done (18 this run)" in out(r)


def test_max_cells_stops_cleanly_and_the_next_invocation_continues(env: Env) -> None:
    r = env.invoke(["matrix", "run", "--max-cells", "4", *env.run_opts()])
    assert "stopped: 4/18 cells done (4 this run), reason: cell_limit" in out(r)
    r = env.invoke(["matrix", "run", *env.run_opts()])
    assert "complete: 18/18 cells done (14 this run)" in out(r)


def test_repeat_needs_a_second_seed_and_ablate_needs_labelled_arms(env: Env) -> None:
    assert env.invoke(["repeat-subset", "--seed", "0", *env.run_opts()]).exit_code == 2
    shutil.copy(SYNTHETIC / "models.yaml", env.configs / "models.yaml")  # arms without labels
    shutil.copy(SYNTHETIC / "budget.yaml", env.configs / "budget.yaml")
    r = env.invoke(["ablate", *env.run_opts()])
    assert r.exit_code == 2 and "no arm labelled 'mid'" in out(r)


# --- tiers --------------------------------------------------------------------------------------


def test_a_flex_matrix_runs_on_a_fresh_store_and_is_priced_at_the_flex_multiple(env: Env) -> None:
    db = env.root / "flex.sqlite"
    args = ["matrix", "run", "--tier", "flex", "--max-cells", "3", "--matrix-db", str(db)]
    r = env.invoke([*args, *env.run_opts()])
    assert r.exit_code == 0, out(r)
    assert {t for (t,) in env.sql("SELECT DISTINCT processing_tier FROM calls")} == {"flex"}
    status = json.loads((env.runs / "matrix_status.json").read_text())
    assert status["tier"] == "flex" and status["executed"] == 3
    assert {x.processing_tier for x in env.matrix_rows(db)} == {"flex"}


def test_a_flex_matrix_cannot_share_a_store_with_standard_rows(env: Env) -> None:
    env.invoke(["pilot", "--n", "2", *env.run_opts()])
    r = env.invoke(["matrix", "run", "--tier", "flex", *env.run_opts()])
    assert r.exit_code == 2 and "one matrix, one tier" in out(r)


def test_the_store_wide_tier_check_catches_cells_that_do_not_overlap_existing_rows(
    env: Env,
) -> None:
    env.invoke(["pilot", "--n", "2", *env.run_opts()])  # standard rows, seed 0
    # the repeat cells use seed 1, so none of them exists yet: only a store-wide check can refuse
    r = env.invoke(["repeat-subset", "--n", "2", "--tier", "flex", *env.run_opts()])
    assert r.exit_code == 2 and "already has rows on tier ['standard']" in out(r)
    assert not any(x.seed == 1 for x in env.matrix_rows())


def test_an_unknown_tier_is_refused(env: Env) -> None:
    r = env.invoke(["matrix", "run", "--tier", "priority", *env.run_opts()])
    assert r.exit_code == 2 and "unknown tier" in out(r)


# --- preflight and the real-provider guards -----------------------------------------------------


def test_preflight_passes_clean_tasks_and_records_them(env: Env) -> None:
    r = env.invoke(["preflight", *env.opts("manifest", "audit", "runs", "taskgen")])
    assert r.exit_code == 0, out(r)
    assert "6 audited task(s) checked, 0 failed" in out(r)
    record = json.loads((env.runs / "preflight.json").read_text())
    assert sorted(record["passed"]) == sorted(env.tasks) and record["failed"] == {}


def test_preflight_names_the_leaky_task_and_exits_nonzero(env: Env) -> None:
    t0 = env.tasks["t0"]
    leaky = t0.model_copy(update={"issue_body": t0.gold_patch})
    write_tasks(env.manifest, [leaky, *[t for k, t in env.tasks.items() if k != "t0"]])
    r = env.invoke(["preflight", *env.opts("manifest", "audit", "runs", "taskgen")])
    assert r.exit_code == 1 and "6 audited task(s) checked, 1 failed" in out(r) and "- t0" in out(r)
    record = json.loads((env.runs / "preflight.json").read_text())
    assert "t0" in record["failed"] and "t0" not in record["passed"]


def test_a_real_provider_run_is_refused_without_the_spend_gate(env: Env) -> None:
    r = env.invoke(["pilot", "--provider", "openai", *env.run_opts()])
    assert r.exit_code == 2 and "CASCADE_ALLOW_SPEND" in out(r)
    assert not env.ledger_db.exists() or env.count_calls() == 0


def test_a_real_provider_run_is_refused_without_a_passing_preflight(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CASCADE_ALLOW_SPEND", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")  # constructing the client is offline
    r = env.invoke(["pilot", "--provider", "openai", *env.run_opts()])
    assert r.exit_code == 2 and "no preflight record" in out(r)
    assert env.count_calls() == 0  # refused before any call

    env.runs.mkdir(exist_ok=True)
    (env.runs / "preflight.json").write_text(json.dumps({"passed": ["t0"], "failed": {}}))
    r = env.invoke(["pilot", "--provider", "openai", "--n", "4", *env.run_opts()])
    assert r.exit_code == 2 and "no passing preflight" in out(r)
    assert env.count_calls() == 0


def test_require_preflight_accepts_exactly_the_covered_tasks(env: Env, tmp_path: Path) -> None:
    runs = tmp_path / "r"
    runs.mkdir()
    (runs / "preflight.json").write_text(json.dumps({"passed": ["t0", "t1"], "failed": {}}))
    covered = [env.tasks["t0"], env.tasks["t1"]]
    matrix_cli.require_preflight(runs, covered, "openai")  # no exception
    matrix_cli.require_preflight(runs, list(env.tasks.values()), "mock")  # mock never needs it
    with pytest.raises(typer.Exit):
        matrix_cli.require_preflight(runs, [env.tasks["t5"]], "openai")


def test_config_status_now_lists_the_buckets_and_per_arm_caps_to_fill() -> None:
    r = runner.invoke(app, ["config-status", "--configs-dir", str(ROOT / "configs")])
    assert r.exit_code == 1
    for field in (
        "buckets[pilot]", "buckets[matrix]", "per_arm_attempt_cap_usd[cheap]",
        "per_arm_attempt_cap_usd[max]",
    ):  # fmt: skip
        assert field in out(r), field


def test_export_writes_under_the_given_runs_dir_and_never_into_the_working_directory(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = env.root / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    r = env.invoke(["export", *env.opts("runs")])
    assert r.exit_code == 0, out(r)
    assert (env.runs / "export" / "matrix.parquet").exists()
    assert list(elsewhere.iterdir()) == []  # nothing leaked into the current directory
    custom = env.root / "custom-out"
    env.invoke(["export", *env.opts("runs"), "--out-dir", str(custom)])
    assert (custom / "ledger.parquet").exists()
