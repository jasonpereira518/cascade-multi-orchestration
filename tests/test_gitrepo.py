from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cascade.config import SelectionLimits
from cascade.harness.sandbox import prepare_workspace
from cascade.taskgen import gitrepo as g
from cascade.taskgen.gitrepo import PatchApplyError, apply_patch, derive_patches, ensure_clone
from tests.conftest import git
from tests.taskgen_helpers import (
    BASE_FILES,
    Upstream,
    build_upstream,
    changed,
    commit_files,
    default_files,
    make_pr,
)

LIMITS = SelectionLimits()


@pytest.fixture
def up(tmp_path: Path) -> Upstream:
    return build_upstream(tmp_path)


def pr_for(up: Upstream, **over: object):
    return make_pr(**{"merge_commit": up.merge_sha, **over})


def test_derives_base_gold_and_test_patches(up: Upstream) -> None:
    out = derive_patches(up.path, pr_for(up), LIMITS)
    assert out.reason is None and out.data is not None
    d = out.data
    assert d.base_commit == up.base_sha  # the merge commit's first parent, full sha
    assert (d.src_files, d.src_lines) == (1, 2)
    # Gold = everything that is not a test file (source + changelog); tests are separate.
    assert "src/tinylib/clamp.py" in d.gold_patch and "CHANGES.md" in d.gold_patch
    assert "return high" in d.gold_patch and "tests/" not in d.gold_patch
    assert "tests/test_clamp.py" in d.test_patch and "test_above_range" in d.test_patch
    assert "src/" not in d.test_patch and "CHANGES.md" not in d.test_patch


def test_patches_reconstruct_the_merged_tree_when_applied_to_the_base(
    up: Upstream, tmp_path: Path
) -> None:
    d = derive_patches(up.path, pr_for(up), LIMITS).data
    assert d is not None
    ws = prepare_workspace(up.path, d.base_commit, tmp_path / "ws")
    apply_patch(ws, d.test_patch)
    apply_patch(ws, d.gold_patch)
    for rel in ("src/tinylib/clamp.py", "tests/test_clamp.py", "CHANGES.md"):
        assert (ws / rel).read_text() == git(up.path, "show", f"{up.merge_sha}:{rel}") + "\n"


def test_unknown_or_missing_merge_commit(up: Upstream) -> None:
    assert (
        derive_patches(up.path, pr_for(up, merge_commit="9" * 40), LIMITS).reason
        == g.MERGE_COMMIT_MISSING
    )
    assert (
        derive_patches(up.path, make_pr(merge_commit=None), LIMITS).reason == g.MERGE_COMMIT_MISSING
    )


def test_file_list_that_disagrees_with_the_diff_is_dropped(up: Upstream) -> None:
    extra = [*default_files(), changed("src/tinylib/extra.py")]
    assert derive_patches(up.path, pr_for(up, files=extra), LIMITS).reason == g.DIFF_MISMATCH
    fewer = default_files()[:2]  # e.g. a rebase-merge where GitHub lists more files than M holds
    assert derive_patches(up.path, pr_for(up, files=fewer), LIMITS).reason == g.DIFF_MISMATCH


def test_same_files_but_different_line_counts_is_a_partial_diff_and_is_dropped(
    up: Upstream,
) -> None:
    """A rebase-merged multi-commit PR can leave M touching the same files as the whole PR.

    The file sets agree, but GitHub's per-file additions/deletions describe the full PR, so a
    last-commit-only diff shows up as a count mismatch.
    """
    whole_pr = [changed("src/tinylib/clamp.py", 3, 1), *default_files()[1:]]
    out = derive_patches(up.path, pr_for(up, files=whole_pr), LIMITS)
    assert out.reason == g.DIFF_COUNTS_MISMATCH and out.data is None


def test_renames_are_dropped_rather_than_guessed(up: Upstream) -> None:
    git(up.path, "mv", "src/tinylib/clamp.py", "src/tinylib/limits.py")
    git(up.path, "commit", "-q", "-m", "rename")
    sha = git(up.path, "rev-parse", "HEAD")
    # GitHub lists only the new path for a rename; the raw diff shows delete + add.
    pr = make_pr(
        merge_commit=sha, files=[changed("src/tinylib/limits.py"), changed("tests/test_clamp.py")]
    )
    assert derive_patches(up.path, pr, LIMITS).reason == g.DIFF_MISMATCH


def test_binary_changes_are_dropped(up: Upstream) -> None:
    (up.path / "src" / "tinylib").mkdir(exist_ok=True)
    (up.path / "src/tinylib/blob.bin").write_bytes(b"\x00\x01\x02\xff")
    git(up.path, "add", "-A")
    git(up.path, "commit", "-q", "-m", "binary")
    sha = git(up.path, "rev-parse", "HEAD")
    pr = make_pr(merge_commit=sha, files=[changed("src/tinylib/blob.bin")])
    assert derive_patches(up.path, pr, LIMITS).reason == g.BINARY_CHANGE


def test_root_commit_has_no_base(up: Upstream) -> None:
    pr = make_pr(merge_commit=up.base_sha, files=[changed(p) for p in BASE_FILES])
    assert derive_patches(up.path, pr, LIMITS).reason == g.UNSUPPORTED_MERGE_SHAPE


def test_true_merge_commits_use_the_first_parent_as_base(up: Upstream) -> None:
    git(up.path, "checkout", "-q", "-b", "feature", up.merge_sha)
    f1 = BASE_FILES["src/tinylib/__init__.py"] + "# touched on the feature branch\n"
    commit_files(up.path, {"src/tinylib/__init__.py": f1}, "feature work")
    commit_files(up.path, {"tests/test_feature.py": "def test_f():\n    pass\n"}, "feature test")
    git(up.path, "checkout", "-q", "main")
    git(up.path, "merge", "--no-ff", "-q", "-m", "Merge feature", "feature")
    merge = git(up.path, "rev-parse", "HEAD")
    pr = make_pr(
        merge_commit=merge,
        files=[changed("src/tinylib/__init__.py"), changed("tests/test_feature.py", 2, 0)],
    )
    out = derive_patches(up.path, pr, LIMITS)
    assert out.data is not None and out.data.base_commit == up.merge_sha  # main's tip before merge


def test_size_rules_are_rechecked_on_the_real_diff(up: Upstream) -> None:
    many: dict[str, str | None] = {f"src/tinylib/m{i}.py": "x = 1\n" for i in range(6)}
    many["tests/test_m.py"] = "y = 1\n"
    sha = commit_files(up.path, many, "six source files")
    files = [changed(p) for p in many]
    assert (
        derive_patches(up.path, make_pr(merge_commit=sha, files=files), LIMITS).reason
        == "too_many_source_files"
    )


# --- apply_patch -----------------------------------------------------------------------------


def test_apply_patch_works_inside_an_enclosing_git_work_tree(up: Upstream, tmp_path: Path) -> None:
    """Workspaces live under runs/, which sits inside the project's own repo."""
    outer = tmp_path / "outer"
    outer.mkdir()
    git(outer, "init", "-q", "-b", "main")
    (outer / "keep.txt").write_text("outer repo content\n")
    git(outer, "add", "-A")
    git(outer, "commit", "-q", "-m", "outer")

    d = derive_patches(up.path, pr_for(up), LIMITS).data
    assert d is not None
    ws = prepare_workspace(up.path, d.base_commit, outer / "runs" / "ws")
    apply_patch(ws, d.gold_patch)

    assert "return high" in (ws / "src/tinylib/clamp.py").read_text()
    assert git(outer, "status", "--porcelain") == "?? runs/"  # only the new dir; nothing modified


def test_apply_patch_failure_raises_a_typed_error(up: Upstream, tmp_path: Path) -> None:
    d = derive_patches(up.path, pr_for(up), LIMITS).data
    assert d is not None
    ws = prepare_workspace(up.path, d.base_commit, tmp_path / "ws")
    apply_patch(ws, d.gold_patch)
    with pytest.raises(PatchApplyError):
        apply_patch(ws, d.gold_patch)  # already applied: hunks no longer match


# --- clone -----------------------------------------------------------------------------------


def test_ensure_clone_creates_then_fetches_new_commits(up: Upstream, tmp_path: Path) -> None:
    dest = tmp_path / "clones" / "x.git"
    ensure_clone(str(up.path), dest)
    assert git(dest, "rev-parse", "main") == up.merge_sha

    newer = commit_files(up.path, {"NOTES.md": "later\n"}, "later commit")
    ensure_clone(str(up.path), dest)  # idempotent: updates in place
    assert git(dest, "rev-parse", "main") == newer
    ws = prepare_workspace(dest, up.base_sha, tmp_path / "ws")  # clone is usable for exports
    assert (ws / "src/tinylib/clamp.py").exists()


def test_git_errors_are_typed(tmp_path: Path) -> None:
    with pytest.raises(g.GitError):
        g.run_git(tmp_path / "does-not-exist", "status")
    assert subprocess.run(["git", "--version"], capture_output=True, check=False).returncode == 0
