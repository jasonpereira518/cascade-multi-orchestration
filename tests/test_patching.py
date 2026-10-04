from __future__ import annotations

import filecmp
import shutil
from collections.abc import Mapping
from pathlib import Path

import pytest

from cascade.agents.patching import (
    PatchError,
    apply_to,
    is_ignored,
    is_source_path,
    make_patch,
    patch_sha,
    paths_in_patch,
)


def write(root: Path, files: Mapping[str, str | bytes]) -> None:
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content)


@pytest.fixture
def trees(tmp_path: Path) -> tuple[Path, Path]:
    base, work = tmp_path / "base", tmp_path / "work"
    files = {
        "pkg/a.py": "def a():\n    return 1\n",
        "pkg/b.py": "def b():\n    return 2\n",
        "tests/test_a.py": "def test_a():\n    assert True\n",
        "README.md": "readme\n",
    }
    write(base, files)
    write(work, files)
    return base, work


def same_tree(a: Path, b: Path) -> bool:
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.diff_files:
        return False
    return all(same_tree(a / d, b / d) for d in cmp.common_dirs)


def test_identical_trees_give_an_empty_patch(trees: tuple[Path, Path]) -> None:
    assert make_patch(*trees) == ""
    assert patch_sha("") is None and patch_sha("  \n") is None


def test_modify_add_and_delete_roundtrip_through_git_apply(
    trees: tuple[Path, Path], tmp_path: Path
) -> None:
    base, work = trees
    (work / "pkg/a.py").write_text("def a():\n    return 100\n")  # modify
    write(work, {"pkg/new.py": "NEW = 1\n"})  # add
    (work / "pkg/b.py").unlink()  # delete
    patch = make_patch(base, work)
    assert (
        patch.startswith("diff --git a/")
        and "--- a/pkg/a.py" in patch
        and "+++ b/pkg/a.py" in patch
    )
    assert "--- /dev/null" in patch and "+++ /dev/null" in patch

    fresh = tmp_path / "fresh"
    shutil.copytree(base, fresh)
    apply_to(fresh, patch)
    assert same_tree(fresh, work)


def test_files_without_a_trailing_newline_roundtrip(
    trees: tuple[Path, Path], tmp_path: Path
) -> None:
    base, work = trees
    (base / "pkg/a.py").write_text("x = 1")  # no newline at EOF
    (work / "pkg/a.py").write_text("x = 2")
    patch = make_patch(base, work)
    assert "\\ No newline at end of file" in patch
    fresh = tmp_path / "fresh"
    shutil.copytree(base, fresh)
    apply_to(fresh, patch)
    assert (fresh / "pkg/a.py").read_text() == "x = 2"


def test_include_filter_builds_a_source_only_patch(trees: tuple[Path, Path]) -> None:
    base, work = trees
    (work / "pkg/a.py").write_text("def a():\n    return 5\n")
    write(work, {"tests/test_cascade_repro.py": "def test_r():\n    assert False\n"})
    (work / "tests/test_a.py").write_text("def test_a():\n    assert False\n")
    full = make_patch(base, work)
    src = make_patch(base, work, include=is_source_path)
    assert set(paths_in_patch(full)) == {
        "pkg/a.py",
        "tests/test_cascade_repro.py",
        "tests/test_a.py",
    }
    assert paths_in_patch(src) == ["pkg/a.py"]


def test_build_products_never_appear_in_a_patch(trees: tuple[Path, Path]) -> None:
    base, work = trees
    write(
        work,
        {
            "pkg/__pycache__/a.cpython-311.pyc": b"\x00\x01",
            "pkg/stale.pyc": b"\x00",
            ".pytest_cache/README.md": "cache",
            ".cascade_results.jsonl": "{}\n",
            ".mypy_cache/x.json": "{}",
        },
    )
    assert make_patch(base, work) == ""


@pytest.mark.parametrize(
    ("path", "ignored"),
    [
        ("pkg/__pycache__/x.pyc", True), ("a.pyc", True), (".git/config", True),
        (".pytest_cache/v", True), (".cascade_results.jsonl", True), ("pkg/a.py", False),
        (".github/workflows/ci.yml", False), ("tests/test_a.py", False),
    ],
)  # fmt: skip
def test_is_ignored(path: str, ignored: bool) -> None:
    assert is_ignored(Path(path)) is ignored


def test_binary_changes_are_refused(trees: tuple[Path, Path]) -> None:
    base, work = trees
    write(work, {"pkg/blob.bin": b"\x00\xff\x00\xfe"})
    with pytest.raises(PatchError, match="binary"):
        make_patch(base, work)


@pytest.mark.parametrize(
    ("path", "is_source"),
    [("pkg/a.py", True), ("src/click/core.py", True), ("CHANGES.md", True), ("pyproject.toml", True),
     ("tests/test_a.py", False), ("tests/test_cascade_repro.py", False), ("pkg/conftest.py", False),
     ("pkg/foo_test.py", False)],
)  # fmt: skip
def test_is_source_path(path: str, is_source: bool) -> None:
    assert is_source_path(path) is is_source


def test_paths_in_patch_and_sha_are_stable(trees: tuple[Path, Path]) -> None:
    base, work = trees
    (work / "pkg/a.py").write_text("def a():\n    return 9\n")
    patch = make_patch(base, work)
    assert paths_in_patch(patch) == ["pkg/a.py"]
    assert patch_sha(patch) == patch_sha(make_patch(base, work))
    assert patch_sha(patch) != patch_sha(patch + "\n")


def test_headers_are_canonical_for_modified_added_and_deleted_files(
    trees: tuple[Path, Path],
) -> None:
    """Regression: --no-prefix writes `b/p b/p` for added and `a/p a/p` for deleted files."""
    base, work = trees
    (work / "pkg/a.py").write_text("def a():\n    return 7\n")
    write(work, {"pkg/added.py": "A = 1\n"})
    (work / "pkg/b.py").unlink()
    patch = make_patch(base, work)
    headers = [ln for ln in patch.splitlines() if ln.startswith("diff --git ")]
    assert sorted(headers) == [
        "diff --git a/pkg/a.py b/pkg/a.py",
        "diff --git a/pkg/added.py b/pkg/added.py",
        "diff --git a/pkg/b.py b/pkg/b.py",
    ]
    assert sorted(paths_in_patch(patch)) == ["pkg/a.py", "pkg/added.py", "pkg/b.py"]


def test_paths_are_relative_to_the_tree_not_prefixed_with_the_scratch_dir_names(
    trees: tuple[Path, Path],
) -> None:
    """Regression: plain `git diff --no-index a b` yields `a/a/pkg/x.py`, which cannot apply."""
    base, work = trees
    (work / "pkg/a.py").write_text("def a():\n    return 3\n")
    patch = make_patch(base, work)
    assert "a/a/" not in patch and "b/b/" not in patch
    assert "--- a/pkg/a.py" in patch and "+++ b/pkg/a.py" in patch


def test_paths_with_spaces_roundtrip(trees: tuple[Path, Path], tmp_path: Path) -> None:
    base, work = trees
    write(work, {"pkg/my file.py": "S = 1\n"})
    patch = make_patch(base, work)
    assert paths_in_patch(patch) == ["pkg/my file.py"]
    fresh = tmp_path / "fresh"
    shutil.copytree(base, fresh)
    apply_to(fresh, patch)
    assert (fresh / "pkg/my file.py").read_text() == "S = 1\n"
