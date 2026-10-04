"""Git operations for task generation: clone/fetch, derive gold/test patches, apply patches.

Patch derivation: for a PR merged by commit M, the base commit is M's first parent P and the PR's
change is `git diff P M`. That is exactly what landed, independent of how stale the PR branch was.
It is cross-checked against GitHub's per-file list and added/deleted line counts for the PR; any
mismatch (a rebase-merge that left M holding only the last commit, a rename, ...) drops the PR
rather than guessing.

The gold patch is every non-test file in that diff (docs and changelog hunks included, as in
SWE-bench-style datasets). The test patch is every test file. Agents never see either.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cascade.config import SelectionLimits
from cascade.errors import CascadeError
from cascade.taskgen.filter import check_size_limits, classify_path, summarize_files
from cascade.taskgen.github import ChangedFile, PRCandidate

MERGE_COMMIT_MISSING = "merge_commit_missing"
UNSUPPORTED_MERGE_SHAPE = "unsupported_merge_shape"
BINARY_CHANGE = "binary_change"
DIFF_MISMATCH = "diff_mismatch"
DIFF_COUNTS_MISMATCH = "diff_counts_mismatch"
EMPTY_PATCH = "empty_patch"


class GitError(CascadeError):
    """A git command failed."""


class PatchApplyError(GitError):
    """A patch did not apply cleanly."""


def run_git(
    repo: Path | None,
    *args: str,
    input_text: str | None = None,
    env_extra: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> str:
    cmd = ["git", "-c", "core.quotepath=false"]
    if repo is not None:
        cmd += ["-C", str(repo)]
    cmd += list(args)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", **(env_extra or {})}
    proc = subprocess.run(  # noqa: S603  (argv list, no shell)
        cmd, input=input_text, capture_output=True, text=True, check=False, env=env, cwd=cwd
    )
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args[:3])} failed: {proc.stderr.strip()[:500]}")
    return proc.stdout


def ensure_clone(url: str, dest: Path) -> None:
    """Create a bare clone, or bring an existing one up to date. Idempotent."""
    if (dest / "HEAD").exists():
        run_git(dest, "fetch", "--quiet", "origin", "+refs/heads/*:refs/heads/*", "--prune")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_git(None, "clone", "--quiet", "--bare", url, str(dest))


@dataclass(frozen=True)
class PatchData:
    base_commit: str
    gold_patch: str
    test_patch: str
    src_files: int
    src_lines: int


@dataclass(frozen=True)
class PatchOutcome:
    data: PatchData | None = None
    reason: str | None = None


def _parents(repo: Path, oid: str) -> list[str] | None:
    try:
        line = run_git(repo, "rev-list", "--parents", "-n", "1", oid).strip()
    except GitError:
        return None
    return line.split()[1:] if line else None


def _diff(repo: Path, base: str, merge: str, paths: Sequence[str]) -> str:
    specs = [f":(literal){p}" for p in paths]
    return run_git(
        repo, "diff", "--no-color", "--no-ext-diff", "--no-renames", base, merge, "--", *specs
    )


def derive_patches(repo: Path, candidate: PRCandidate, limits: SelectionLimits) -> PatchOutcome:
    merge = candidate.merge_commit
    if not merge:
        return PatchOutcome(reason=MERGE_COMMIT_MISSING)
    parents = _parents(repo, merge)
    if parents is None:
        return PatchOutcome(reason=MERGE_COMMIT_MISSING)
    if not 1 <= len(parents) <= 2:
        return PatchOutcome(reason=UNSUPPORTED_MERGE_SHAPE)
    base = parents[0]

    numstat = run_git(repo, "diff", "--numstat", "--no-renames", base, merge)
    changed: dict[str, tuple[int, int]] = {}
    for line in numstat.splitlines():
        added, deleted, path = line.split("\t", 2)
        if added == "-" or deleted == "-":
            return PatchOutcome(reason=BINARY_CHANGE)
        changed[path] = (int(added), int(deleted))
    github = {f.path: (f.additions, f.deletions) for f in candidate.files}
    if set(changed) != set(github):
        return PatchOutcome(reason=DIFF_MISMATCH)
    # Same files is not enough: a rebase-merged multi-commit PR whose last commit happens to touch
    # the same files would otherwise pass with a partial diff. Per-file line counts must agree too.
    if changed != github:
        return PatchOutcome(reason=DIFF_COUNTS_MISMATCH)

    # Re-check the size rules on the real diff, which is authoritative over GitHub's counts.
    real_files = [
        ChangedFile(path=p, additions=a, deletions=d, change_type="MODIFIED")
        for p, (a, d) in changed.items()
    ]
    summary = summarize_files(real_files)
    reason = check_size_limits(summary, limits)
    if reason is not None:
        return PatchOutcome(reason=reason)

    test_paths = sorted(p for p in changed if classify_path(p) == "test")
    other_paths = sorted(p for p in changed if classify_path(p) != "test")
    test_patch = _diff(repo, base, merge, test_paths)
    gold_patch = _diff(repo, base, merge, other_paths)
    if not test_patch.strip() or not gold_patch.strip():
        return PatchOutcome(reason=EMPTY_PATCH)

    full_base = run_git(repo, "rev-parse", base).strip()
    return PatchOutcome(
        data=PatchData(
            base_commit=full_base,
            gold_patch=gold_patch,
            test_patch=test_patch,
            src_files=summary.src_files,
            src_lines=summary.src_lines,
        )
    )


def apply_patch(workspace: Path, patch: str) -> None:
    """Apply a unified diff to a plain directory.

    GIT_CEILING_DIRECTORIES stops git from discovering an enclosing repository (workspaces live
    under runs/, which sits inside this project's own work tree), which would make `git apply`
    resolve paths against the wrong root and silently skip files.
    """
    try:
        run_git(
            None,
            "apply",
            "--whitespace=nowarn",
            "-",
            input_text=patch,
            env_extra={"GIT_CEILING_DIRECTORIES": str(workspace.resolve().parent)},
            cwd=workspace,
        )
    except GitError as e:
        raise PatchApplyError(str(e)) from e
