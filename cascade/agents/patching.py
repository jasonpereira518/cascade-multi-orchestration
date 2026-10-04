"""Turn an agent's edits into a patch, deterministically.

The agent edits ``work``; ``base`` is an untouched export of the base commit. The patch is
`git diff --no-index --no-prefix a b` run on copies named ``a`` and ``b``, which yields exactly
``a/<path>`` and ``b/<path>`` headers, so it applies with `git apply` to a fresh base export.
(Without --no-prefix git would emit ``a/a/<path>``, which does not apply.)

Build products (``__pycache__``, ``.pyc``, pytest caches, the plugin's report file) are excluded so
test runs never pollute a patch.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from cascade.errors import CascadeError
from cascade.taskgen.filter import classify_path
from cascade.taskgen.gitrepo import apply_patch

IGNORED_DIRS = frozenset(
    {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".venv"}
)
IGNORED_FILES = frozenset({".cascade_results.jsonl"})
_GIT = shutil.which("git") or "git"
_DIFF_HEADER = re.compile(r"^diff --git a/(.+) b/(.+)$", re.MULTILINE)


class PatchError(CascadeError):
    """The edits could not be turned into a safe text patch."""


def is_source_path(path: str) -> bool:
    """True for files the graded patch may contain: everything that is not a test file.

    Tests (including the reproducer's) are excluded from the graded patch; hidden tests are applied
    on top of it, and edits to tests would conflict with them.
    """
    return classify_path(path) != "test"


def is_ignored(rel: Path) -> bool:
    return (
        any(part in IGNORED_DIRS for part in rel.parts)
        or rel.name in IGNORED_FILES
        or rel.suffix == ".pyc"
    )


def _copy_filtered(src: Path, dst: Path, include: Callable[[str], bool]) -> None:
    for path in sorted(src.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(src)
        if is_ignored(rel) or not include(rel.as_posix()):
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def make_patch(base: Path, work: Path, *, include: Callable[[str], bool] = lambda _p: True) -> str:
    """Unified diff turning ``base`` into ``work`` over the files ``include`` accepts."""
    with tempfile.TemporaryDirectory(prefix="cascade-diff-") as tmp:
        root = Path(tmp)
        (root / "a").mkdir()
        (root / "b").mkdir()
        _copy_filtered(base, root / "a", include)
        _copy_filtered(work, root / "b", include)
        proc_out = _diff_no_index(root)
    if "\nBinary files " in "\n" + proc_out or "GIT binary patch" in proc_out:
        raise PatchError("the edits include a binary file change, which is not supported")
    return _normalize_headers(proc_out)


def _normalize_headers(patch: str) -> str:
    """Rewrite every `diff --git` header to the canonical `a/<p> b/<p>`.

    With --no-prefix git writes `b/p b/p` for an added file and `a/p a/p` for a deleted one. Both
    still apply, but would be skipped by tools that expect the canonical form. Renames are disabled,
    so both halves always name the same path.
    """
    out: list[str] = []
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            rest = line[len("diff --git ") :].rstrip("\n")
            path_len = (len(rest) - 5) // 2  # rest is "<x>/<p> <y>/<p>" with equal p
            path = rest[2 : 2 + path_len]
            line = f"diff --git a/{path} b/{path}\n"
        out.append(line)
    return "".join(out)


def _diff_no_index(root: Path) -> str:
    proc = subprocess.run(  # noqa: S603  (fixed argv, no shell)
        [
            _GIT, "-c", "core.quotepath=false", "diff", "--no-index", "--no-color",
            "--no-ext-diff", "--no-renames", "--no-prefix", "--", "a", "b",
        ],
        cwd=root, capture_output=True, text=True, check=False,
    )  # fmt: skip
    if proc.returncode not in (0, 1):  # 1 means "differences found"
        raise PatchError(f"git diff failed: {proc.stderr.strip()[:300]}")
    return proc.stdout


def paths_in_patch(patch: str) -> list[str]:
    seen: list[str] = []
    for _a, b in _DIFF_HEADER.findall(patch):
        if b not in seen:
            seen.append(b)
    return seen


def patch_sha(patch: str) -> str | None:
    return hashlib.sha256(patch.encode()).hexdigest() if patch.strip() else None


def apply_to(workspace: Path, patch: str) -> None:
    """Apply with the same machinery the task pipeline uses (git apply, no enclosing repo)."""
    apply_patch(workspace, patch)
