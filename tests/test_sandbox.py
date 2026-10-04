from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from cascade.errors import LeakError, SandboxError
from cascade.harness.sandbox import (
    LeakGuard,
    SandboxRunner,
    SandboxSpec,
    paths_added_by_patch,
    prepare_workspace,
)
from cascade.taskgen.schema import Task
from tests.conftest import SYNTHETIC, FixtureRepo


@pytest.fixture
def sandbox_root(tmp_path: Path) -> Path:
    root = tmp_path / "sandboxes"
    root.mkdir()
    return root


@pytest.fixture
def workspace(fixture_repo: FixtureRepo, sandbox_root: Path) -> Path:
    return prepare_workspace(fixture_repo.path, fixture_repo.base_sha, sandbox_root / "ws")


def runner_for(task: Task, root: Path, host_paths: Sequence[Path] = ()) -> SandboxRunner:
    return SandboxRunner(LeakGuard.from_task(task, host_paths=host_paths), sandbox_root=root)


def flag_value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def all_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


# --- patch parsing -----------------------------------------------------------------------------


def test_paths_added_by_patch_finds_new_files_only() -> None:
    patch = (
        "diff --git a/old.py b/old.py\n--- a/old.py\n+++ b/old.py\n@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/tests/test_new.py b/tests/test_new.py\nnew file mode 100644\n"
        "--- /dev/null\n+++ b/tests/test_new.py\n@@ -0,0 +1 @@\n+def test(): pass\n"
    )
    assert paths_added_by_patch(patch) == {"tests/test_new.py"}


# --- workspace preparation ---------------------------------------------------------------------


def test_workspace_is_base_commit_without_history_or_patches(
    workspace: Path, fixture_repo: FixtureRepo
) -> None:
    assert not (workspace / ".git").exists()
    clamp = (workspace / "tinylib" / "clamp.py").read_text()
    tests = (workspace / "tests" / "test_clamp.py").read_text()
    # The fixture repo's HEAD has the gold fix and the hidden test; the workspace must not.
    assert "return high" not in clamp
    assert "test_above_range" not in tests
    assert fixture_repo.head_sha != fixture_repo.base_sha


def test_workspace_contains_no_file_equal_to_gold_or_test_patch(
    workspace: Path, task: Task
) -> None:
    forbidden = {task.gold_patch.encode(), task.test_patch.encode()}
    for path in all_files(workspace):
        assert path.read_bytes() not in forbidden
        assert path.suffix not in {".patch", ".diff"}


def test_prepare_workspace_rejects_bad_sha_and_dirty_destination(
    fixture_repo: FixtureRepo, sandbox_root: Path
) -> None:
    with pytest.raises(SandboxError, match="40-char"):
        prepare_workspace(fixture_repo.path, "--output=/tmp/x", sandbox_root / "a")
    with pytest.raises(SandboxError, match="git archive failed"):
        prepare_workspace(fixture_repo.path, "0" * 40, sandbox_root / "b")
    dirty = sandbox_root / "c"
    dirty.mkdir()
    (dirty / "stale.txt").write_text("x")
    with pytest.raises(SandboxError, match="not empty"):
        prepare_workspace(fixture_repo.path, fixture_repo.base_sha, dirty)


# --- leak guard --------------------------------------------------------------------------------


def test_guard_passes_clean_workspace(workspace: Path, task: Task) -> None:
    assert LeakGuard.from_task(task).scan(workspace) == []


def test_guard_flags_git_directory_and_gitfile(workspace: Path, task: Task) -> None:
    guard = LeakGuard.from_task(task)
    (workspace / ".git").mkdir()
    assert any("git metadata" in v for v in guard.scan(workspace))
    (workspace / ".git").rmdir()
    (workspace / ".git").write_text("gitdir: ../elsewhere")
    assert any("git metadata file" in v for v in guard.scan(workspace))


@pytest.mark.parametrize("variant", ["exact", "no_trailing_newline"])
@pytest.mark.parametrize("which", ["gold", "test"])
def test_guard_flags_renamed_copy_of_patch(
    workspace: Path, task: Task, which: str, variant: str
) -> None:
    text = task.gold_patch if which == "gold" else task.test_patch
    if variant == "no_trailing_newline":
        text = text.rstrip("\n")
    (workspace / "notes.txt").write_text(text)
    violations = LeakGuard.from_task(task).scan(workspace)
    assert violations == ["file identical to a gold/test patch: notes.txt"]


def test_guard_flags_paths_the_patches_add(workspace: Path, task: Task) -> None:
    added_patch = "--- /dev/null\n+++ b/tests/test_hidden.py\n@@ -0,0 +1 @@\n+x\n"
    task = task.model_copy(update={"test_patch": added_patch})
    (workspace / "tests" / "test_hidden.py").write_text("anything")
    assert LeakGuard.from_task(task).scan(workspace) == [
        "patch-added path present: tests/test_hidden.py"
    ]


# --- runner: validation and command ------------------------------------------------------------


def test_runner_refuses_workspace_with_leaked_patch(
    workspace: Path, task: Task, sandbox_root: Path
) -> None:
    (workspace / "gold.patch").write_text(task.gold_patch)
    with pytest.raises(LeakError):
        runner_for(task, sandbox_root).build_command(
            SandboxSpec(image="img", workspace=workspace), ["true"], name="n"
        )


def test_command_has_network_off_limits_and_no_host_leakage(
    workspace: Path, task: Task, sandbox_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    spec = SandboxSpec(
        image="img:1", workspace=workspace, cpus=1.5, memory_mb=2048, pids_limit=64,
        env={"PYTHONHASHSEED": "0"},
    )  # fmt: skip
    cmd = runner_for(task, sandbox_root).build_command(spec, ["pytest", "-x"], name="n1")

    assert flag_value(cmd, "--network") == "none"
    assert flag_value(cmd, "--cpus") == "1.5"
    assert flag_value(cmd, "--memory") == "2048m"
    assert flag_value(cmd, "--memory-swap") == "2048m"
    assert flag_value(cmd, "--pids-limit") == "64"
    assert flag_value(cmd, "--cap-drop") == "ALL"
    assert flag_value(cmd, "--security-opt") == "no-new-privileges"
    assert "--read-only" in cmd and "--rm" in cmd
    assert flag_value(cmd, "--user") == f"{os.getuid()}:{os.getgid()}"
    assert cmd[-3:] == ["img:1", "pytest", "-x"]

    for banned in ("--privileged", "--env-file", "-v", "--volume", "--network=host", "-e"):
        assert banned not in cmd
    envs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--env"]
    assert sorted(envs) == ["HOME=/tmp", "PYTHONHASHSEED=0"]
    assert not any("sk-test" in part for part in cmd)


def test_only_the_workspace_is_mounted_and_its_contents_are_clean(
    workspace: Path, task: Task, sandbox_root: Path
) -> None:
    cmd = runner_for(task, sandbox_root, host_paths=[SYNTHETIC]).build_command(
        SandboxSpec(image="img", workspace=workspace), ["true"], name="n"
    )
    mounts = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--mount"]
    assert len(mounts) == 1
    opts = dict(kv.split("=", 1) for kv in mounts[0].split(","))
    source = Path(opts["source"])
    assert opts["type"] == "bind" and opts["target"] == "/workspace"
    assert source == workspace.resolve()
    assert source.is_relative_to(sandbox_root.resolve())

    forbidden = {task.gold_patch.encode(), task.test_patch.encode()}
    for path in all_files(source):
        assert ".git" not in path.relative_to(source).parts
        assert path.read_bytes() not in forbidden
        assert path.name not in {"gold.patch", "test.patch", "task.json"}


def test_runner_rejects_workspace_outside_sandbox_root(
    workspace: Path, task: Task, tmp_path: Path
) -> None:
    other_root = tmp_path / "elsewhere"
    other_root.mkdir()
    with pytest.raises(SandboxError, match="inside sandbox root"):
        runner_for(task, other_root).validate(SandboxSpec(image="i", workspace=workspace))
    root = runner_for(task, workspace).validate  # the root itself is not a valid workspace
    with pytest.raises(SandboxError, match="inside sandbox root"):
        root(SandboxSpec(image="i", workspace=workspace))


def test_runner_rejects_mount_overlapping_forbidden_host_paths(
    task: Task, sandbox_root: Path
) -> None:
    store = sandbox_root / "task-store"
    store.mkdir()
    (store / "gold.patch").write_text(task.gold_patch)
    # Workspace IS the patch store.
    with pytest.raises(LeakError, match="forbidden host path"):
        runner_for(task, sandbox_root, [store]).validate(SandboxSpec(image="i", workspace=store))
    # Workspace is an ancestor of the patch store.
    ws = sandbox_root / "ws"
    nested = ws / "nested-store"
    nested.mkdir(parents=True)
    with pytest.raises(LeakError, match="forbidden host path"):
        runner_for(task, sandbox_root, [nested]).validate(SandboxSpec(image="i", workspace=ws))
    # Workspace sits inside a forbidden directory.
    inside = store / "ws"
    inside.mkdir()
    with pytest.raises(LeakError, match="forbidden host path"):
        runner_for(task, sandbox_root, [store]).validate(SandboxSpec(image="i", workspace=inside))


@pytest.mark.parametrize("name", ["OPENAI_API_KEY", "GITHUB_TOKEN", "MY_SECRET", "db_password"])
def test_secret_looking_env_vars_are_refused(
    workspace: Path, task: Task, sandbox_root: Path, name: str
) -> None:
    spec = SandboxSpec(image="i", workspace=workspace, env={name: "v"})
    with pytest.raises(SandboxError, match="refusing to pass env var"):
        runner_for(task, sandbox_root).validate(spec)


def test_workspace_path_with_comma_is_refused(task: Task, sandbox_root: Path) -> None:
    ws = sandbox_root / "a,b"
    ws.mkdir()
    with pytest.raises(SandboxError, match="commas"):
        runner_for(task, sandbox_root).validate(SandboxSpec(image="i", workspace=ws))


# --- runner: execution (docker faked) ----------------------------------------------------------


def test_run_returns_result_from_docker_process(
    workspace: Path, task: Task, sandbox_root: Path
) -> None:
    seen: list[list[str]] = []

    def fake(argv: Sequence[str], timeout: float | None) -> subprocess.CompletedProcess[str]:
        seen.append(list(argv))
        return subprocess.CompletedProcess(list(argv), 3, stdout="out", stderr="err")

    runner = SandboxRunner(LeakGuard.from_task(task), sandbox_root=sandbox_root, run_fn=fake)
    result = runner.run(SandboxSpec(image="img", workspace=workspace), ["pytest"])
    assert (result.exit_code, result.stdout, result.stderr, result.timed_out) == (
        3,
        "out",
        "err",
        False,
    )
    assert len(seen) == 1 and seen[0][:2] == ["docker", "run"]


def test_run_kills_container_on_timeout(workspace: Path, task: Task, sandbox_root: Path) -> None:
    seen: list[list[str]] = []

    def fake(argv: Sequence[str], timeout: float | None) -> subprocess.CompletedProcess[str]:
        seen.append(list(argv))
        if argv[1] == "run":
            raise subprocess.TimeoutExpired(list(argv), timeout or 0, output=b"partial")
        return subprocess.CompletedProcess(list(argv), 0, stdout="", stderr="")

    runner = SandboxRunner(LeakGuard.from_task(task), sandbox_root=sandbox_root, run_fn=fake)
    result = runner.run(SandboxSpec(image="img", workspace=workspace, timeout_s=1), ["sleep", "9"])
    assert result.timed_out and result.exit_code is None and result.stdout == "partial"
    name = flag_value(seen[0], "--name")
    assert seen[1] == ["docker", "rm", "-f", name]


# --- real Docker (skipped unless a daemon and a local image are available) ---------------------


def _docker_image() -> str | None:
    image = os.environ.get("CASCADE_TEST_DOCKER_IMAGE")
    if not image:
        return None
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=20)
        subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True, check=True, timeout=20
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return image


@pytest.mark.docker
@pytest.mark.skipif(
    _docker_image() is None,
    reason="needs a running Docker daemon and CASCADE_TEST_DOCKER_IMAGE set to a local image",
)
def test_real_container_has_no_network_no_git_and_read_only_root(
    workspace: Path, task: Task, sandbox_root: Path
) -> None:
    image = _docker_image()
    assert image is not None
    script = (
        "tail -n +3 /proc/net/dev | cut -d: -f1 | tr -d ' ' | tr '\\n' ','; echo;"
        "ls -A /workspace; touch /workspace/writable && echo workspace-rw;"
        "touch /etc/should-fail 2>/dev/null || echo root-ro"
    )
    result = runner_for(task, sandbox_root).run(
        SandboxSpec(image=image, workspace=workspace), ["sh", "-c", script]
    )
    lines = result.stdout.splitlines()
    assert lines[0] == "lo,"  # only loopback: no external interface
    assert ".git" not in result.stdout.split()
    assert "workspace-rw" in lines and "root-ro" in lines
