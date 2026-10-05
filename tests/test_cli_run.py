"""`cascade run` and `cascade ledger-summary` through the real Typer app.

Only two things are stubbed: Docker (the executor factory and the daemon probe) and, in some tests,
the model source. Configs, manifest, clone, ledger, matrix, grading, and CLI output are all real.
Data is the synthetic fixture task; no live task pipeline run exists yet.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cascade import cli
from cascade.agents import runner as agent_runner
from cascade.harness.providers.mock import MockModelSource
from cascade.harness.sandbox import ContainerRunner
from tests.agent_helpers import NativeExecutor, good_plans
from tests.conftest import ROOT, SYNTHETIC, FixtureRepo

runner = CliRunner()
ARM = "mock-small@low"


@dataclass
class CliEnv:
    root: Path
    configs: Path
    manifest: Path
    runs: Path
    taskgen_runs: Path
    kill_switch: Path
    task_id: str

    def args(self, *extra: str, arm: str = ARM, task: str | None = None) -> list[str]:
        return [
            "run", "--task", task or self.task_id, "--arm", arm,
            "--configs-dir", str(self.configs), "--manifest", str(self.manifest),
            "--runs-dir", str(self.runs), "--taskgen-runs-dir", str(self.taskgen_runs),
            "--max-output-tokens", "512", *extra,
        ]  # fmt: skip

    def invoke(self, *extra: str, **kw: str) -> object:
        return runner.invoke(cli.app, self.args(*extra, **kw))


@pytest.fixture
def cli_env(tmp_path: Path, fixture_repo: FixtureRepo, monkeypatch: pytest.MonkeyPatch) -> CliEnv:
    configs = tmp_path / "configs"
    configs.mkdir()
    shutil.copy(SYNTHETIC / "models.yaml", configs / "models.yaml")
    kill = tmp_path / "KILL_SWITCH"
    budget = (
        (SYNTHETIC / "budget.yaml")
        .read_text()
        .replace("kill_switch_path: KILL_SWITCH", f"kill_switch_path: {kill}")
    )
    (configs / "budget.yaml").write_text(budget)
    (configs / "repos.yaml").write_text(
        "date_window: {start: 2025-01-01, end: 2025-12-31}\n"
        "repos:\n  - name: synthetic/tinylib\n    license: MIT\n    python_version: '3.12'\n"
        "    install_cmd: none\n    test_cmd: python -m pytest\n"
    )
    manifest = tmp_path / "tasks" / "manifest.jsonl"
    manifest.parent.mkdir()
    manifest.write_text(fixture_repo.task.model_dump_json() + "\n")
    taskgen_runs = tmp_path / "runs" / "taskgen"
    clone = taskgen_runs / "synthetic__tinylib" / "clone.git"
    shutil.copytree(fixture_repo.path, clone)  # a clone the exporter can `git archive` from

    monkeypatch.setattr(cli, "docker_available", lambda: True)
    monkeypatch.setattr(
        agent_runner, "docker_executor_factory",
        lambda task, test_cmd: (lambda r: _native(r)),
    )  # fmt: skip
    return CliEnv(
        tmp_path,
        configs,
        manifest,
        tmp_path / "runs",
        taskgen_runs,
        kill,
        fixture_repo.task.task_id,
    )


def _native(runner_: ContainerRunner) -> NativeExecutor:
    return NativeExecutor(runner_)


def scripted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_make_source", lambda provider: MockModelSource(good_plans()))


def out(result: object) -> str:
    return str(result.output)  # type: ignore[attr-defined]


def code(result: object) -> int:
    return int(result.exit_code)  # type: ignore[attr-defined]


def test_a_successful_scripted_run_prints_both_outcomes_cost_and_ledger_total(
    cli_env: CliEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted(monkeypatch)
    result = cli_env.invoke()
    text = out(result)
    assert code(result) == 0, text
    assert f"task {cli_env.task_id} | arm {ARM} | seed 0" in text
    assert "true_success=True  verifier_claimed=True" in text
    assert "calls 10" in text and "ledger total spent so far: $" in text and "artifacts:" in text
    assert "LEAK AUDIT" not in text


def test_the_default_mock_runs_end_to_end_and_fails_honestly(cli_env: CliEnv) -> None:
    result = cli_env.invoke()  # no scripted plans: roles answer minimally and fix nothing
    text = out(result)
    assert code(result) == 0, text
    assert "true_success=False  verifier_claimed=False" in text and "grading: empty_patch" in text


def test_rerunning_reuses_the_row_and_force_replaces_it(
    cli_env: CliEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted(monkeypatch)
    cli_env.invoke()
    again = out(cli_env.invoke())
    assert "existing row reused; use --force to rerun" in again
    forced = out(cli_env.invoke("--force"))
    assert "existing row reused" not in forced and "true_success=True" in forced


def test_toggles_seed_and_repairs_are_passed_through(
    cli_env: CliEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted(monkeypatch)
    text = out(
        cli_env.invoke("--no-reproducer", "--no-reviewer", "--seed", "3", "--max-repairs", "0")
    )
    assert "seed 3" in text and "calls 6" in text and "verifier_claimed=True" in text


def test_the_cap_option_can_only_lower_the_attempt_cap(
    cli_env: CliEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted(monkeypatch)
    text = out(cli_env.invoke("--cap-usd", "0.000001"))
    assert "attempt ended early" in text and "BudgetExceededError" in text and "attempt cap" in text
    assert "true_success=False" in text


def test_the_kill_switch_is_respected_and_recorded(
    cli_env: CliEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted(monkeypatch)
    cli_env.kill_switch.touch()
    result = cli_env.invoke()
    assert (
        code(result) == 0 and "KillSwitchEngagedError" in out(result) and "calls 0" in out(result)
    )


# --- refusals ------------------------------------------------------------------------------


def test_unknown_arm_lists_the_arms_that_exist(cli_env: CliEnv) -> None:
    result = cli_env.invoke(arm="nope@high")
    assert code(result) == 2 and "unknown arm 'nope@high'" in out(result) and ARM in out(result)


def test_unknown_task_and_missing_manifest(cli_env: CliEnv) -> None:
    assert (
        "is not in" in out(cli_env.invoke(task="no-such-task"))
        and code(cli_env.invoke(task="x")) == 2
    )
    cli_env.manifest.unlink()
    assert "no manifest" in out(cli_env.invoke()) and code(cli_env.invoke()) == 2


def test_unfilled_shipped_configs_are_refused_with_the_fields_named() -> None:
    result = runner.invoke(
        cli.app,
        [
            "run",
            "--task",
            "t",
            "--arm",
            "gpt-6-luna@low",
            "--configs-dir",
            str(ROOT / "configs"),
        ],
    )
    assert code(result) == 2 and "unfilled TODO fields" in out(result)


def test_a_missing_clone_and_unavailable_docker_are_explained(
    cli_env: CliEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "docker_available", lambda: False)
    assert "docker daemon is not reachable" in out(cli_env.invoke()) and code(cli_env.invoke()) == 2
    monkeypatch.setattr(cli, "docker_available", lambda: True)
    shutil.rmtree(cli_env.taskgen_runs / "synthetic__tinylib")
    assert "no clone at" in out(cli_env.invoke())


def test_a_repo_without_a_filled_config_entry_is_refused(cli_env: CliEnv) -> None:
    (cli_env.configs / "repos.yaml").write_text(
        "date_window: {start: 2025-01-01, end: 2025-12-31}\nrepos:\n  - name: other/repo\n"
        "    license: MIT\n    python_version: '3'\n    install_cmd: x\n    test_cmd: y\n"
    )
    assert "no filled entry for synthetic/tinylib" in out(cli_env.invoke())


def test_openai_needs_the_spend_gate_the_key_and_an_openai_model(
    cli_env: CliEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    # The synthetic models are provider "mock", so this is refused before any source is built.
    r = cli_env.invoke("--provider", "openai")
    assert code(r) == 2 and "not an openai model in models.yaml" in out(r)
    # Make the model an openai one: now the spend gate is what stops it.
    models = (
        (cli_env.configs / "models.yaml").read_text().replace("provider: mock", "provider: openai")
    )
    (cli_env.configs / "models.yaml").write_text(models)
    r = cli_env.invoke("--provider", "openai")
    assert code(r) == 2 and "CASCADE_ALLOW_SPEND=1" in out(r)
    assert not (cli_env.runs / "ledger.sqlite").exists() or "calls: 0" in out(
        runner.invoke(cli.app, ["ledger-summary", "--runs-dir", str(cli_env.runs)])
    )  # nothing was sent or logged
    monkeypatch.setenv("CASCADE_ALLOW_SPEND", "1")
    monkeypatch.delenv("OPENAI_API_KEY")
    assert "OPENAI_API_KEY is not set" in out(cli_env.invoke("--provider", "openai"))


def test_an_unknown_provider_is_refused(cli_env: CliEnv) -> None:
    assert "unknown provider" in out(cli_env.invoke("--provider", "carrier-pigeon"))


# --- ledger-summary ------------------------------------------------------------------------


def test_ledger_summary_reports_calls_roles_and_outcomes(
    cli_env: CliEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted(monkeypatch)
    cli_env.invoke()
    result = runner.invoke(cli.app, ["ledger-summary", "--runs-dir", str(cli_env.runs)])
    text = out(result)
    assert code(result) == 0, text
    assert "LEDGER SUMMARY" in text and "calls: 10 (errors: 0)" in text and "total cost: $" in text
    assert f"  {ARM}: 10 calls" in text
    for role in ("triager", "reproducer", "localizer", "patcher", "reviewer"):
        assert f"  {role}:" in text
    assert "OUTCOMES (one row per attempt)" in text
    assert f"{cli_env.task_id} | {ARM} | 0 | True | True |" in text


def test_ledger_summary_without_a_ledger_is_a_clear_error(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["ledger-summary", "--runs-dir", str(tmp_path)])
    assert code(result) == 2 and "no ledger at" in out(result)
