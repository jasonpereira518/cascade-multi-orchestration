from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from cascade.errors import LeakError, SandboxError
from cascade.harness.grading import GradingRunner
from cascade.harness.sandbox import LeakGuard, SandboxRunner, SandboxSpec
from cascade.taskgen.schema import Task


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "work"
    r.mkdir()
    return r


def dirty_workspace(root: Path, task: Task) -> Path:
    """A grading workspace: it legitimately holds git metadata and the gold patch."""
    ws = root / "ws"
    ws.mkdir()
    (ws / ".git").mkdir()
    (ws / "gold.patch").write_text(task.gold_patch)
    return ws


def flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


def test_grading_runner_mounts_workspaces_the_agent_runner_refuses(root: Path, task: Task) -> None:
    ws = dirty_workspace(root, task)
    spec = SandboxSpec(image="img", workspace=ws)
    with pytest.raises(LeakError):
        SandboxRunner(LeakGuard.from_task(task), sandbox_root=root).build_command(
            spec, ["t"], name="n"
        )
    cmd = GradingRunner(sandbox_root=root).build_command(spec, ["t"], name="n")
    assert flag(cmd, "--mount").startswith("type=bind,source=")


def test_grading_and_agent_commands_share_identical_isolation_flags(root: Path, task: Task) -> None:
    ws = root / "clean"
    ws.mkdir()
    spec = SandboxSpec(image="img", workspace=ws, env={"FOO": "1"})
    agent = SandboxRunner(LeakGuard.from_task(task), sandbox_root=root)
    grading = GradingRunner(sandbox_root=root)
    assert agent.build_command(spec, ["pytest"], name="n") == grading.build_command(
        spec, ["pytest"], name="n"
    )


def test_grading_runner_keeps_network_off_and_hardening(root: Path, task: Task) -> None:
    cmd = GradingRunner(sandbox_root=root).build_command(
        SandboxSpec(image="img", workspace=dirty_workspace(root, task)), ["t"], name="n"
    )
    assert flag(cmd, "--network") == "none"
    assert flag(cmd, "--cap-drop") == "ALL" and "--read-only" in cmd
    assert "--privileged" not in cmd and "--env-file" not in cmd


def test_grading_runner_still_enforces_root_and_env_rules(
    root: Path, tmp_path: Path, task: Task
) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    runner = GradingRunner(sandbox_root=root)
    with pytest.raises(SandboxError, match="inside sandbox root"):
        runner.validate(SandboxSpec(image="i", workspace=outside))
    ws = root / "ok"
    ws.mkdir()
    with pytest.raises(SandboxError, match="refusing to pass env var"):
        runner.validate(SandboxSpec(image="i", workspace=ws, env={"GITHUB_TOKEN": "x"}))


def test_grading_runner_run_uses_the_injected_docker(root: Path, task: Task) -> None:
    seen: list[list[str]] = []

    def fake(argv: Sequence[str], timeout: float | None) -> subprocess.CompletedProcess[str]:
        seen.append(list(argv))
        return subprocess.CompletedProcess(list(argv), 0, stdout="ok", stderr="")

    runner = GradingRunner(sandbox_root=root, run_fn=fake)
    result = runner.run(SandboxSpec(image="img", workspace=dirty_workspace(root, task)), ["t"])
    assert result.exit_code == 0 and result.stdout == "ok" and seen[0][:2] == ["docker", "run"]
