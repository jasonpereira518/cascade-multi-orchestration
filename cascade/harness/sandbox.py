"""Docker sandbox for agent code execution (hard rule 3).

Three layers keep gold/test patches, hidden tests, and later git history away from agents:

1. ``prepare_workspace`` exports the tree at the base commit with ``git archive``. The result has
   no ``.git`` at all, so no history (before or after the base) is present.
2. ``LeakGuard.scan`` inspects the workspace before every run and refuses on a ``.git`` entry, on
   any file the gold/test patch adds, or on any file that is byte-identical to a gold/test patch.
   It runs inside ``SandboxRunner`` unconditionally; there is no flag to skip it.
3. ``SandboxRunner`` mounts exactly one directory (the workspace, under ``sandbox_root``) and
   refuses any workspace that contains, or sits inside, a registered forbidden host path (for
   example the directory holding stored patches). The container runs with the network off, all
   capabilities dropped, a read-only root filesystem, resource limits, and no host environment.

Limits, stated plainly: the scan catches whole-file copies and added paths, not arbitrary excerpts
of a patch pasted into an existing file. Because the workspace is built from the base commit and
nothing else is mounted, such content has no route in. Grading runs (which legitimately apply the
test patch) must use a separate code path with its own tests; this runner is agent-side only.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import subprocess
import tarfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from cascade.errors import LeakError, SandboxError
from cascade.taskgen.schema import Task

WORKSPACE_TARGET = "/workspace"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|OPENAI|ANTHROPIC|GITHUB", re.I)


def paths_added_by_patch(patch: str) -> set[str]:
    """Paths of files a unified diff creates (``--- /dev/null`` followed by ``+++ b/<path>``)."""
    lines = patch.splitlines()
    added: set[str] = set()
    for i, line in enumerate(lines[:-1]):
        nxt = lines[i + 1]
        if line.startswith("--- /dev/null") and nxt.startswith("+++ b/"):
            added.add(nxt[len("+++ b/") :].split("\t")[0])
    return added


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class LeakGuard:
    # (size, sha256) of files that must never appear in an agent-visible tree.
    forbidden_blobs: frozenset[tuple[int, str]] = frozenset()
    forbidden_rel_paths: frozenset[str] = frozenset()
    # Host locations (e.g. where patches are stored) that no mount may contain or sit inside.
    forbidden_host_paths: tuple[Path, ...] = ()

    @classmethod
    def from_task(cls, task: Task, *, host_paths: Sequence[Path] = ()) -> LeakGuard:
        blobs: set[tuple[int, str]] = set()
        for text in (task.gold_patch, task.test_patch):
            for variant in (text, text.rstrip("\n"), text.rstrip("\n") + "\n"):
                raw = variant.encode("utf-8")
                blobs.add((len(raw), _digest(raw)))
        rel = paths_added_by_patch(task.gold_patch) | paths_added_by_patch(task.test_patch)
        return cls(
            forbidden_blobs=frozenset(blobs),
            forbidden_rel_paths=frozenset(rel),
            forbidden_host_paths=tuple(host_paths),
        )

    def scan(self, root: Path) -> list[str]:
        """Return human-readable violations found under ``root`` (empty means clean)."""
        sizes = {size for size, _ in self.forbidden_blobs}
        violations: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            base = Path(dirpath)
            rel_dir = base.relative_to(root)
            if ".git" in dirnames:
                violations.append(f"git metadata at {(rel_dir / '.git').as_posix()}")
                dirnames.remove(".git")  # report once; do not walk history
            if ".git" in filenames:
                violations.append(f"git metadata file at {(rel_dir / '.git').as_posix()}")
            for fname in filenames:
                rel = (rel_dir / fname).as_posix()
                if rel in self.forbidden_rel_paths:
                    violations.append(f"patch-added path present: {rel}")
                    continue
                full = base / fname
                if full.is_symlink():
                    continue
                size = full.stat().st_size
                if size in sizes and (size, _digest(full.read_bytes())) in self.forbidden_blobs:
                    violations.append(f"file identical to a gold/test patch: {rel}")
        return violations


def prepare_workspace(repo: Path, base_commit: str, dest: Path, *, git_bin: str = "git") -> Path:
    """Export the tree at ``base_commit`` into ``dest`` (no ``.git``, no history)."""
    if not _SHA40.fullmatch(base_commit):
        raise SandboxError("base_commit must be a full 40-char lowercase hex sha")
    if dest.exists() and any(dest.iterdir()):
        raise SandboxError(f"workspace destination is not empty: {dest}")
    proc = subprocess.run(  # noqa: S603  (argv list, validated sha, no shell)
        [git_bin, "-C", str(repo), "archive", "--format=tar", base_commit],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SandboxError(f"git archive failed: {proc.stderr.decode(errors='replace').strip()}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
        tar.extractall(dest, filter="data")
    return dest


@dataclass(frozen=True)
class SandboxSpec:
    image: str
    workspace: Path
    cpus: float = 2.0
    memory_mb: int = 4096
    pids_limit: int = 512
    timeout_s: float = 300.0
    env: Mapping[str, str] = field(default_factory=lambda: {})
    # Defaults to the invoking user so files written to the workspace stay owned by them.
    user: str | None = None


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float


class RunFn(Protocol):
    def __call__(
        self, argv: Sequence[str], timeout: float | None
    ) -> subprocess.CompletedProcess[str]: ...


def _default_run(argv: Sequence[str], timeout: float | None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603  (argv list, no shell)
        list(argv), capture_output=True, text=True, timeout=timeout, check=False
    )


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


class SandboxRunner:
    def __init__(
        self,
        guard: LeakGuard,
        *,
        sandbox_root: Path,
        docker_bin: str = "docker",
        run_fn: RunFn = _default_run,
    ) -> None:
        self._guard = guard
        self._root = sandbox_root.resolve()
        self._docker = docker_bin
        self._run = run_fn

    def validate(self, spec: SandboxSpec) -> Path:
        """Return the resolved workspace, or raise if mounting it could leak or escape."""
        ws = spec.workspace.resolve()
        if not ws.is_dir():
            raise SandboxError(f"workspace is not a directory: {ws}")
        if ws == self._root or not ws.is_relative_to(self._root):
            raise SandboxError(f"workspace must be inside sandbox root {self._root}: {ws}")
        if "," in str(ws):
            raise SandboxError("workspace path must not contain commas (docker --mount syntax)")
        for forbidden in self._guard.forbidden_host_paths:
            fp = forbidden.resolve()
            if fp == ws or fp.is_relative_to(ws) or ws.is_relative_to(fp):
                raise LeakError([f"mount {ws} overlaps forbidden host path {fp}"])
        for name in spec.env:
            if not _ENV_NAME.fullmatch(name) or _SECRET_ENV.search(name):
                raise SandboxError(f"refusing to pass env var {name!r} into the sandbox")
        violations = self._guard.scan(ws)
        if violations:
            raise LeakError(violations)
        return ws

    def build_command(self, spec: SandboxSpec, argv: Sequence[str], *, name: str) -> list[str]:
        ws = self.validate(spec)
        user = spec.user if spec.user is not None else f"{os.getuid()}:{os.getgid()}"
        env = {"HOME": "/tmp", **spec.env}  # noqa: S108  (path inside the container tmpfs)
        cmd = [
            self._docker, "run", "--rm",
            "--name", name,
            "--network", "none",
            "--cpus", str(spec.cpus),
            "--memory", f"{spec.memory_mb}m",
            "--memory-swap", f"{spec.memory_mb}m",
            "--pids-limit", str(spec.pids_limit),
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--read-only",
            "--tmpfs", "/tmp:rw,exec,nosuid,size=1g",  # noqa: S108
            "--user", user,
            "--mount", f"type=bind,source={ws},target={WORKSPACE_TARGET}",
            "--workdir", WORKSPACE_TARGET,
        ]  # fmt: skip
        for key, value in env.items():
            cmd += ["--env", f"{key}={value}"]
        cmd.append(spec.image)
        cmd += list(argv)
        return cmd

    def run(self, spec: SandboxSpec, argv: Sequence[str]) -> SandboxResult:
        name = f"cascade-{uuid.uuid4().hex[:12]}"
        cmd = self.build_command(spec, argv, name=name)
        start = time.monotonic()
        try:
            proc = self._run(cmd, spec.timeout_s)
        except subprocess.TimeoutExpired as exc:
            self._run([self._docker, "rm", "-f", name], 30.0)
            return SandboxResult(
                exit_code=None,
                stdout=_text(exc.stdout),
                stderr=_text(exc.stderr),
                timed_out=True,
                duration_s=time.monotonic() - start,
            )
        return SandboxResult(
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            timed_out=False,
            duration_s=time.monotonic() - start,
        )
